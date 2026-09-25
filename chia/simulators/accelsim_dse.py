"""chia.simulators.accelsim_dse — the HW/SW co-design space walk.

The walk scores a *kernel* by the best EDP it can reach anywhere on the
hardware grid::

    score(kernel) = min over h in GRID of EDP(kernel, h)

so no hardware point is ever committed to. That matters because the hardware
verdict depends on the kernel: at 32 CTAs more SMs are worthless, at 256 CTAs
they may not be. A loop that fixed hardware in round 1 would be fitting the
chip to the *worst* kernel it will ever see, with no way to un-commit. Here the
hardware axis is not searched at all -- it is enumerated inside every score.

That is affordable only because the probe wave prunes it first. The raw knob
space is ~50-dimensional; what survives is ~10-30 points at 130 s each, fanned
out across CPU cores while the GPU lane builds and traces the next kernels.

Two asymmetries drive every design choice here:

* **A trace is expensive and reusable; a simulation is cheap and disposable.**
  One build+trace unlocks a whole row of the grid. So the unit of work is a
  row, not a point.
* **The hardware axis has more dimensions but is not the blowup risk.** Probes
  kill ~90% of it empirically and its points are cheap. The software axis has
  fewer dimensions and costs a build plus GPU time per point, so it is sampled
  under a hard per-round cap rather than enumerated.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field

from chia.simulators.accelsim_probes import (
    IDEALIZATION_PROBES,
    STARVE_PROBES,
    Probe,
)

# ======================================================================
# Probes added for CUTLASS
# ======================================================================
#
# The stock twelve were written against a naive SGEMM. CUTLASS lives in shared
# memory and the tensor pipe, and the baseline's own log names its limiter:
#
#     GPGPU-Sim uArch: CTA/core = 1, limited by: shmem regs
#
# One CTA per SM, capped by shared memory *and* registers. No existing probe
# tests either, so the occupancy question the whole study turns on was
# unmeasured.

CUTLASS_EXTRA_PROBES: tuple[Probe, ...] = (
    Probe(
        "big_shmem",
        {"-gpgpu_shmem_size": "671744", "-gpgpu_shmem_sizeDefault": "671744"},
        "If shared memory were 4x larger, how much faster?",
        "Shared-memory capacity is what caps CTA/core. More of it raises "
        "occupancy without touching the kernel.",
        "Cut shared memory per CTA: fewer pipeline stages, or a smaller tile.",
    ),
    Probe(
        "big_regfile",
        {"-gpgpu_shader_registers": "262144", "-gpgpu_registers_per_block": "262144"},
        "If the register file were 4x larger, how much faster?",
        "Register capacity is what caps CTA/core -- the other half of the "
        "'limited by: shmem regs' pair.",
        "Cut registers per thread: smaller warp tile, or fewer accumulators.",
    ),
    Probe(
        "fast_tensor",
        # NOT -gpgpu_num_tensor_core_units. Measured 2026-09-21/22: HMMA maps
        # to SPECIALIZED_UNIT_3_OP (ISA_Def/ampere_opcode.h:49), so every knob
        # named "tensor" drives a pipeline this trace never touches -- 4x the
        # units plus a 4x wider OC_EX_TENSOR_CORE returned bit-identical
        # cycles on two tiles. The live lever is trace_driven.cc:955.
        #
        # Nor is the *unit count* on the real lever a bandwidth: shader.cc:1713
        # tests num_units only as `> 0`, and shader.cc:5176 stamps every
        # operand-collector unit with its scheduler id, so only OC_EX_SPEC
        # slots 0..num_sched-1 are ever written while FU k reads slot k.
        # Usable units are capped at -gpgpu_num_sched_per_core; 4 -> 16 is ten
        # bit-identical points. Aggregate rate is num_units/II, and II is the
        # only half of that a knob can move. Stock is "12,8".
        {},
        "If each tensor core retired an HMMA every cycle, how much faster?",
        "Tensor issue rate is the constraint for a TF32 GEMM, and the stock "
        "probe set has nothing on it. This is the ceiling: 8x the stock "
        "aggregate rate, so any real tensor setting buys less than this.",
        "Raise arithmetic intensity per tensor op, or overlap the epilogue.",
        trace_knobs={"-trace_opcode_latency_initiation_spec_op_3": "12,1"},
    ),
    Probe(
        "big_l1",
        # -gpgpu_adaptive_cache_config recomputes L1D geometry per launch and
        # discards hand-edited -gpgpu_cache:dl1, so this must turn it off and
        # be read against adaptive_off rather than against baseline.
        {
            "-gpgpu_adaptive_cache_config": "0",
            "-gpgpu_unified_l1d_size": "768",
            "-gpgpu_cache:dl1": "S:16:128:64,L:T:m:L:L,A:2048:256,16:0,32",
        },
        "If the L1 were 4x larger with 4x the MSHRs, how much faster?",
        "L1 capacity is worth sweeping. Distinct from ideal_l1, which is an "
        "unbuildable idealization; this one is a real geometry.",
        "Stage more of the tile through shared memory instead of L1.",
        reference="adaptive_off",
    ),
)

L2_SPLIT_PROBES: tuple[Probe, ...] = (
    # `ideal_l2` moves capacity AND -gpgpu_l2_rop_latency (200 -> 20) at once,
    # so a large result cannot be attributed. Measured 10.29% at 4096^2x64
    # where the inputs are only 2 MB and cannot plausibly be capacity-limited,
    # which points at latency -- but pointing is not measuring. These two
    # separate the effects; ideal_l2 stays as the combined upper bound.
    Probe(
        "l2_capacity",
        {"-gpgpu_cache:dl2": "S:512:128:32,L:B:m:L:X,A:1024:32,32:0,32"},
        "If the L2 were 4x larger at stock latency, how much faster?",
        "L2 *capacity* is the constraint -- the working set does not fit.",
        "Shrink the footprint per wave, or block the loop so it fits L2.",
    ),
    Probe(
        "l2_latency",
        {"-gpgpu_l2_rop_latency": "20"},
        "If the L2/ROP round trip were 10x shorter at stock capacity, how "
        "much faster?",
        "L2 *latency* is the constraint, not capacity. Streaming write traffic "
        "is latency-exposed; a bigger cache will not help it.",
        "Raise memory-level parallelism, or overlap the epilogue with compute.",
    ),
)

CUTLASS_PROBES: tuple[Probe, ...] = (
    IDEALIZATION_PROBES + CUTLASS_EXTRA_PROBES + L2_SPLIT_PROBES
)

# What the decision path actually sweeps, and it is deliberately not
# CUTLASS_PROBES.
#
# CUTLASS_PROBES is 21 up, 2 controls, 1 down -- effectively an upward-only
# sweep. It answers "does more of X help?" for every structure and "does less
# of X hurt?" for exactly one (half_sm). That asymmetry is not a cosmetic gap:
# an iso-area design has to pay for what it buys, and the price of a structure
# is its *downward* number, which an upward sweep never collects. Adding the
# starvation half costs no extra wall clock -- 24 concurrent simulations
# becomes 43, still one wave inside the 48 Ray slots, all replaying the same
# trace.
#
# It matters because the two sides disagree. Measured 2026-09-23 at
# 4096x4096x64: L2 doubled is +9.19%, L2 halved is -13.69%, L2 quartered is
# -22.29%. Reading only the upward half ranks L2 the top actionable lever and
# says nothing about the fact that it costs half again as much to sell as it
# pays to buy. A loop that proposes funding an L2 increase by trimming L2
# elsewhere is making that exact mistake, and until this tuple existed it had
# no measurement that would stop it.
CODESIGN_PROBES: tuple[Probe, ...] = CUTLASS_PROBES + STARVE_PROBES


# ======================================================================
# Hardware axis
# ======================================================================

BASE_L2 = "S:128:128:16,L:B:m:L:X,A:192:4,32:0,32"
BASE_L1 = "S:4:128:64,L:T:m:L:L,A:512:64,16:0,32"


def _l2(nsets: int, assoc: int, mshr: int) -> str:
    return f"S:{nsets}:128:{assoc},L:B:m:L:X,A:{mshr}:4,32:0,32"


def _clock(core_mhz: int, dram_mhz: int = 1512) -> str:
    return f"{core_mhz}:{core_mhz}:{core_mhz}:{dram_mhz}"


# Dimensions the grid may vary, each as {label: {knob: value}}.
#
# `always` dimensions go in regardless of what the probes say, because the
# probe verdict is about *cycles* and these are the two knobs that move
# *power*. At 1024^3 only 32 of 108 SMs receive a CTA, so cutting SM count
# should cost almost no cycles while removing a large idle-power term -- a
# real EDP win that a cycles-only reading calls "confounded" and discards.
# They are also the only two knobs McPAT sees without XML re-rendering.
HW_DIMENSIONS: dict[str, dict] = {
    "n_sm": {
        "always": True,
        "knob": "-gpgpu_n_clusters",
        # sm27 dropped at the 4096^2x64 shape: half_sm already measures -64%,
        # so 27 SMs is known-bad, and 512 CTAs over 27 slots is 19 rounds --
        # the slowest cell in the grid, bought for a foregone conclusion.
        # sm432 added because 2x_sm returned +18% un-confounded, so the
        # interesting direction is up and 216 may not be the knee.
        "values": {"sm54": "54", "sm108": "108", "sm216": "216", "sm432": "432"},
        "licensed_by": ("2x_sm", "half_sm"),
    },
    "clock": {
        "always": True,
        "knob": "-gpgpu_clock_domains",
        "values": {
            "clk987": _clock(987),
            "clk1198": _clock(1198),
            "clk1410": _clock(1410),
            "clk1620": _clock(1620),
        },
        "licensed_by": (),
    },
    "l2": {
        "always": False,
        "knob": "-gpgpu_cache:dl2",
        "values": {
            "l2_half": _l2(64, 16, 192),
            "l2_base": BASE_L2,
            "l2_2x": _l2(256, 16, 384),
            "l2_4x": _l2(512, 16, 768),
        },
        "licensed_by": ("ideal_l2",),
    },
    "shmem": {
        "always": False,
        "knob": "-gpgpu_shmem_size",
        "values": {
            "shm_base": "167936",
            "shm_2x": "335872",
            "shm_4x": "671744",
        },
        "licensed_by": ("big_shmem",),
    },
    "regfile": {
        "always": False,
        "knob": "-gpgpu_shader_registers",
        "values": {
            "rf_base": "65536",
            "rf_2x": "131072",
        },
        "licensed_by": ("big_regfile",),
    },
    # Tensor rate, NOT tensor count. "<latency>,<initiation>" in trace.config;
    # stock A100 is "12,8". Latency is nearly free on this workload (1 vs 16 at
    # fixed II differ by 0.03%) so it is pinned at 12 and only II moves.
    # Aggregate HMMA rate is min(num_sched, num_units)/II, and num_units is not
    # a knob that reaches it -- see the ``fast_tensor`` probe comment. Going
    # below II=4 needs more sub-cores, not a different value here, so the
    # column stops at the rate a wider tensor core can actually deliver.
    "tensor": {
        "always": False,
        "trace": True,
        "knob": "-trace_opcode_latency_initiation_spec_op_3",
        "values": {"ii8": "12,8", "ii4": "12,4", "ii2": "12,2"},
        "licensed_by": ("fast_tensor",),
    },
}

BASE_LABELS = {
    "n_sm": "sm108",
    "clock": "clk1410",
    "l2": "l2_base",
    "shmem": "shm_base",
    "regfile": "rf_base",
    "tensor": "ii8",
}


@dataclass
class GridPoint:
    """One hardware configuration: a label and the knobs that make it."""

    name: str
    knobs: dict[str, str]
    dims: dict[str, str]
    trace_knobs: dict[str, str] = field(default_factory=dict)
    """Dimensions whose knob lives in ``trace.config``, not ``gpgpusim.config``.

    Kept separate because the two files are written separately and a knob is
    only ever read back out of the one it was written into. Folding them
    together would send ``-trace_opcode_latency_initiation_*`` to the wrong
    parser, where it is accepted and ignored -- the exact failure that made
    every tensor result in this project unmeasurable until 2026-09-21.
    """

    @property
    def is_baseline(self) -> bool:
        return all(self.dims.get(d) == BASE_LABELS[d] for d in self.dims)


def licensed_dimensions(report: dict, *, bar: float = 5.0) -> list[str]:
    """Which dimensions the probe wave says are worth varying.

    An idealization gives a resource unlimited capacity, so its headroom is an
    *upper bound* on what any real setting of that resource could buy. A
    dimension whose ceiling is under the bar cannot repay a column, and one
    whose probe came back inert was never measured at all. Both are dropped.

    ``always`` dimensions bypass this entirely -- see :data:`HW_DIMENSIONS`.
    """
    live: set[str] = set()
    by_name = {r["name"]: r for r in report.get("ranked", [])}
    for dim, spec in HW_DIMENSIONS.items():
        if spec["always"]:
            live.add(dim)
            continue
        for probe in spec["licensed_by"]:
            r = by_name.get(probe)
            if not r or r.get("inert"):
                continue
            # Either headroom against baseline or against the probe's own
            # control can license a sweep: the first says "we could win here",
            # the second says "this resource is on the critical path but the
            # baseline already realises it" -- which still makes the dimension
            # worth varying downward.
            for key in ("headroom_pct", "control_pct"):
                v = r.get(key)
                if v is not None and abs(v) >= bar:
                    live.add(dim)
                    break
    return [d for d in HW_DIMENSIONS if d in live]


def build_grid(
    dims: list[str], *, cap: int = 30, cross: int = 2
) -> list[GridPoint]:
    """Enumerate hardware points from the licensed dimensions.

    The first ``cross`` dimensions are taken as a full cross product -- that is
    the tradeoff *surface*, and for the two ``always`` dimensions it is exactly
    the area/power frontier. Remaining dimensions are added as one-at-a-time
    variations off the baseline, which keeps the point count linear in the tail
    rather than exponential.
    """
    dims = [d for d in dims if d in HW_DIMENSIONS]
    if not dims:
        dims = [d for d, s in HW_DIMENSIONS.items() if s["always"]]

    head, tail = dims[:cross], dims[cross:]
    points: list[GridPoint] = []
    seen: set[str] = set()

    def add(labels: dict[str, str]) -> None:
        full = dict(BASE_LABELS)
        full.update(labels)
        name = "/".join(full[d] for d in dims)
        if name in seen or len(points) >= cap:
            return
        seen.add(name)
        knobs: dict[str, str] = {}
        trace_knobs: dict[str, str] = {}
        for d in dims:
            spec = HW_DIMENSIONS[d]
            target = trace_knobs if spec.get("trace") else knobs
            target[spec["knob"]] = spec["values"][full[d]]
        points.append(GridPoint(name=name, knobs=knobs,
                                dims={d: full[d] for d in dims},
                                trace_knobs=trace_knobs))

    if head:
        for combo in itertools.product(*[HW_DIMENSIONS[d]["values"] for d in head]):
            add(dict(zip(head, combo)))
    for d in tail:
        for label in HW_DIMENSIONS[d]["values"]:
            add({d: label})
    return points


# ======================================================================
# Software axis
# ======================================================================

SMEM_CAP = 164 * 1024  # SM80 shared memory per CTA
MAX_THREADS = 1024


@dataclass(frozen=True)
class Tile:
    """A CUTLASS threadblock configuration.

    ``warps`` is the warp count per dimension, exactly as CUTLASS's own
    ``TileDescription`` records it; the warp *shape* is derived, not chosen.
    """

    tb: tuple[int, int, int]
    warps: tuple[int, int, int]
    stages: int
    split_k: int = 1

    @property
    def warp_shape(self) -> tuple[int, int, int]:
        return (self.tb[0] // self.warps[0], self.tb[1] // self.warps[1], self.tb[2])

    @property
    def threads(self) -> int:
        return self.warps[0] * self.warps[1] * self.warps[2] * 32

    @property
    def smem_bytes(self) -> int:
        """Multistage pipeline holds ``stages`` copies of both operand tiles."""
        return self.stages * (self.tb[0] + self.tb[1]) * self.tb[2] * 4

    def blocks(self, n: int) -> int:
        return math.ceil(n / self.tb[0]) * math.ceil(n / self.tb[1]) * self.split_k

    @property
    def name(self) -> str:
        base = f"tb{self.tb[0]}x{self.tb[1]}x{self.tb[2]}_s{self.stages}"
        return base + (f"_k{self.split_k}" if self.split_k > 1 else "")

    def feasible(self) -> tuple[bool, str]:
        """Reject illegal tiles with arithmetic, before spending a compile."""
        if self.smem_bytes > SMEM_CAP:
            return False, f"shmem {self.smem_bytes} > {SMEM_CAP}"
        if self.threads > MAX_THREADS:
            return False, f"threads {self.threads} > {MAX_THREADS}"
        if any(self.tb[i] % self.warps[i] for i in (0, 1)):
            return False, "warp count does not divide threadblock shape"
        wm, wn, _ = self.warp_shape
        if wm % 16 or wn % 8:
            return False, f"warp shape {wm}x{wn} not a multiple of the 16x8x8 MMA"
        return True, ""

    def defines(self, n: int) -> list[str]:
        wm, wn, wk = self.warp_shape
        return [
            f"-DTB_M={self.tb[0]}", f"-DTB_N={self.tb[1]}", f"-DTB_K={self.tb[2]}",
            f"-DWP_M={wm}", f"-DWP_N={wn}", f"-DWP_K={wk}",
            f"-DSTAGES={self.stages}", f"-DSPLIT_K={self.split_k}",
            f"-DGEMM_N={n}",
        ]


# Exactly the (tile, stages, warps) triples CUTLASS's own generator emits for
# SM80 TF32 (`GenerateSM80_TensorOp_1688`, min_cc=80). Inventing shapes outside
# this list mostly produces configurations that fail `can_implement`, so the
# search space is CUTLASS's, not ours.
LEGAL_TILES: tuple[Tile, ...] = (
    Tile((256, 128, 16), (4, 2, 1), 3),
    Tile((128, 256, 16), (2, 4, 1), 3),
    Tile((256, 64, 16), (4, 1, 1), 4),
    Tile((64, 256, 16), (1, 4, 1), 4),
    Tile((128, 128, 16), (2, 2, 1), 5),
    Tile((128, 128, 16), (2, 2, 1), 4),
    Tile((128, 128, 16), (2, 2, 1), 3),
    Tile((128, 64, 16), (2, 2, 1), 6),
    Tile((64, 128, 16), (2, 2, 1), 6),
    Tile((64, 64, 16), (2, 2, 1), 10),
    Tile((256, 128, 32), (4, 2, 1), 3),
    Tile((128, 256, 32), (2, 4, 1), 3),   # <-- stock CUTLASS default; baseline
    Tile((256, 64, 32), (4, 1, 1), 4),
    Tile((64, 256, 32), (1, 4, 1), 4),
    Tile((128, 128, 32), (2, 2, 1), 4),
    Tile((128, 128, 32), (2, 2, 1), 3),
    Tile((128, 64, 32), (2, 2, 1), 3),
    Tile((64, 128, 32), (2, 2, 1), 3),
    Tile((64, 64, 32), (2, 2, 1), 5),
)

BASELINE_TILE = Tile((128, 256, 32), (2, 4, 1), 3)


def resident_ctas(tile: Tile, *, smem_cap: int = SMEM_CAP) -> int:
    """How many CTAs of this tile fit on one SM, shared memory being the cap.

    This is the quantity the stages/occupancy tradeoff turns on: pipeline
    stages buy latency hiding *inside* a CTA and cost shared memory, which is
    what limits how many CTAs hide latency *across* each other.
    """
    return max(1, smem_cap // tile.smem_bytes) if tile.smem_bytes else 1


def suggest_tiles(
    report: dict, *, n: int, k: int = 6, exclude: set[str] | None = None
) -> list[Tile]:
    """Pick the next software candidates from the probe verdict.

    The probe table already pairs each resource with the software move that
    relieves it, so this is a lookup rather than a search. When the verdict is
    "not enough parallelism" -- which is what a flat SM probe means -- the
    ranking key is block count, and shared memory per CTA breaks ties because
    it decides how many of those blocks are co-resident.
    """
    exclude = exclude or set()
    # kernel_hints is a list of {probe, headroom_pct, hint} dicts, not of
    # strings -- joining it directly raises TypeError the first time any probe
    # fires a hint, which is exactly when this branch matters. The recorded
    # runs never hit it because the hint list came back empty; that is luck,
    # not correctness. Found 2026-09-23.
    hints = " ".join(
        (h.get("hint", "") if isinstance(h, dict) else str(h))
        for h in (report.get("kernel_hints") or ())
    ).lower()
    quant = report.get("quantization", {})
    want_parallelism = (
        quant.get("confounded")
        or "parallelism" in hints
        or "split-k" in hints
        or "smaller" in hints
    )

    pool = [t for t in LEGAL_TILES if t.name not in exclude and t.feasible()[0]]
    if not want_parallelism:
        pool.sort(key=lambda t: (t.smem_bytes, -t.blocks(n)))
        return pool[:k]

    # Stratify across block count rather than taking the extreme.
    #
    # The hint gives a *direction*, not a magnitude, and following it greedily
    # is wrong. Measured at 1024^3 against a 32-block baseline of 124,322
    # cycles: 64 blocks is 91,566 (-26%), 128 blocks is 122,795 (-1%), and 256
    # blocks is 135,019 (+9%) -- worse than the baseline the move was meant to
    # fix, because the smaller tile costs 2x the instructions and that
    # overwhelms the occupancy it buys. Sorting by -blocks and taking the top k
    # picks the 256-block tile first: the worst kernel in the set.
    #
    # So sample one tile per distinct block count, cheapest-instruction first
    # within each, and let the screen find the interior optimum.
    by_blocks: dict[int, list[Tile]] = {}
    for t in pool:
        by_blocks.setdefault(t.blocks(n), []).append(t)
    for group in by_blocks.values():
        group.sort(key=lambda t: (-resident_ctas(t), t.smem_bytes))

    picked: list[Tile] = []
    levels = sorted(by_blocks, reverse=True)
    depth = 0
    while len(picked) < k and any(len(by_blocks[b]) > depth for b in levels):
        for b in levels:
            if len(picked) >= k:
                break
            if len(by_blocks[b]) > depth:
                picked.append(by_blocks[b][depth])
        depth += 1
    return picked


# ======================================================================
# Scoring
# ======================================================================

@dataclass
class Cell:
    """One (kernel, hardware) evaluation."""

    kernel: str
    point: str
    cycles: float | None = None
    power_w: float | None = None
    joules: float | None = None
    edp: float | None = None
    occupancy: float | None = None
    issued_ctas: float | None = None
    cta_per_core: float | None = None
    ok: bool = False
    note: str = ""


@dataclass
class KernelRow:
    """A kernel and every hardware point it was evaluated on."""

    kernel: str
    tile: Tile | None
    cells: list[Cell] = field(default_factory=list)

    @property
    def best(self) -> Cell | None:
        live = [c for c in self.cells if c.ok and c.edp is not None]
        return min(live, key=lambda c: c.edp) if live else None

    @property
    def score(self) -> float:
        b = self.best
        return b.edp if b else float("inf")


def pareto_front(cells: list[Cell]) -> list[Cell]:
    """Non-dominated points on (cycles, joules).

    EDP collapses the tradeoff to one number so the search can rank; the front
    keeps it open so a reader can see what was traded. Both are reported
    because they answer different questions.
    """
    live = [c for c in cells if c.ok and c.cycles and c.joules]
    front: list[Cell] = []
    for c in live:
        if not any(
            o is not c and o.cycles <= c.cycles and o.joules <= c.joules
            and (o.cycles < c.cycles or o.joules < c.joules)
            for o in live
        ):
            front.append(c)
    return sorted(front, key=lambda c: c.cycles)


__all__ = [
    "BASELINE_TILE",
    "BASE_LABELS",
    "CUTLASS_EXTRA_PROBES",
    "CUTLASS_PROBES",
    "CODESIGN_PROBES",
    "L2_SPLIT_PROBES",
    "Cell",
    "GridPoint",
    "HW_DIMENSIONS",
    "KernelRow",
    "LEGAL_TILES",
    "SMEM_CAP",
    "Tile",
    "build_grid",
    "licensed_dimensions",
    "pareto_front",
    "resident_ctas",
    "simulable",
    "suggest_tiles",
]


# ======================================================================
# Reporting
# ======================================================================

def cells_from_records(records: list[dict]) -> list[Cell]:
    """Rehydrate :class:`Cell` objects from the driver's JSON records."""
    out = []
    for r in records:
        out.append(Cell(
            kernel=r.get("kernel", ""), point=r.get("point", ""),
            cycles=r.get("cycles"), power_w=r.get("power_w"),
            joules=r.get("joules"), edp=r.get("edp"),
            occupancy=r.get("occupancy"), issued_ctas=r.get("issued_ctas"),
            cta_per_core=r.get("cta_per_core"),
            ok=bool(r.get("ok")), note=r.get("note", ""),
        ))
    return out


def format_grid(cells: list[Cell], baseline_point: str = "sm108/clk1410") -> str:
    """The grid as a table, normalized to the baseline design point.

    Normalizing is not cosmetic. Absolute watts carry the Volta-XML caveat and
    absolute EDP is in units nobody has intuition for, but a *ratio* between two
    points rendered from the same XML is the quantity the walk actually decides
    on, and it survives the caveat intact.
    """
    live = [c for c in cells if c.ok and c.edp]
    if not live:
        return "no successful cells"
    # Point names gain a component per licensed dimension ("sm108/clk1410"
    # becomes "sm108/clk1410/l2_base"), so a fixed baseline string silently
    # missed and fell through to a *lexicographic* min, normalizing the whole
    # table to an arbitrary cell. A prefix match is no better -- "sm108/clk1410/"
    # matches every L2 variant. The grid centre is the point whose every
    # component is a BASE_LABELS value, whatever the dimension count.
    stock = set(BASE_LABELS.values())
    base = (next((c for c in live if c.point == baseline_point), None)
            or next((c for c in live
                     if all(part in stock for part in c.point.split("/"))), None))
    fellback = base is None
    if fellback:
        base = min(live, key=lambda c: c.point)
    front = {id(c) for c in pareto_front(live)}

    warn = ("" if not fellback else
            f"!! baseline {baseline_point!r} not in grid -- normalized to "
            f"{base.point!r} instead; ratios are NOT vs stock\n\n")

    rows = [
        f"{'design point':<22}{'cycles':>10}{'cyc':>8}{'power':>9}{'energy':>9}"
        f"{'EDP':>9}{'occ':>8}{'CTA':>5}  ",
        f"{'':<22}{'':>10}{'vs base':>8}{'W':>9}{'mJ':>9}{'vs base':>9}{'%':>8}"
        f"{'/SM':>5}  pareto",
        "-" * 88,
    ]
    for c in sorted(live, key=lambda c: c.edp):
        rows.append(
            f"{c.point:<22}{c.cycles:>10.0f}"
            f"{c.cycles / base.cycles:>8.3f}"
            f"{c.power_w:>9.2f}"
            f"{c.joules * 1e3:>9.4f}"
            f"{c.edp / base.edp:>9.3f}"
            f"{(c.occupancy or 0):>8.2f}"
            f"{(c.cta_per_core or 0):>5.0f}"
            f"  {'*' if id(c) in front else ''}"
            f"{'  <- baseline' if c is base else ''}"
        )
    dead = [c for c in cells if not c.ok]
    if dead:
        rows.append("")
        rows.append(f"failed ({len(dead)}): " + ", ".join(
            f"{c.point} [{c.note[:40]}]" for c in dead[:6]))
    return "\n".join(rows)


def format_rows(rows: list[KernelRow], baseline_kernel: str = "") -> str:
    """Software axis: each kernel scored by the best cell in its row."""
    live = [r for r in rows if r.best]
    if not live:
        return "no scored kernels"
    base = next((r for r in live if r.kernel == baseline_kernel), None) or max(
        live, key=lambda r: r.score)
    out = [
        f"{'kernel':<22}{'blocks':>8}{'best point':<22}{'cycles':>10}"
        f"{'mJ':>9}{'EDP':>9}",
        f"{'':<22}{'':>8}{'':<22}{'':>10}{'':>9}{'vs base':>9}",
        "-" * 80,
    ]
    for r in sorted(live, key=lambda r: r.score):
        b = r.best
        blocks = r.tile.blocks(1024) if r.tile else 0
        out.append(
            f"{r.kernel:<22}{blocks:>8}{b.point:<22}{b.cycles:>10.0f}"
            f"{b.joules * 1e3:>9.4f}{r.score / base.score:>9.3f}"
            f"{'  <- baseline' if r is base else ''}"
        )
    return "\n".join(out)


# ======================================================================
# Admission gate
# ======================================================================

def simulable(node, kernelslist: str, outdir: str, *, arch: str = "SM80_A100",
              max_cycle: int = 0, timeout_s: int = 1800) -> tuple[bool, str]:
    """Can this trace be simulated at all? Run one cheap sim and find out.

    Correct and simulable are different properties, and the correctness gate
    cannot tell them apart. A CUTLASS kernel built with ``AlignmentA=1``
    computes bit-identical results to the aligned version -- it passes a
    reference check without complaint -- and then every simulation of it dies
    with ``ERROR ** deadlock detected`` after ~68k idle cycles, because scalar
    global loads generate enough outstanding misses to wedge the memory model.

    So this runs once, with the power model **off**, before the trace is
    admitted to a probe wave or a grid row. The economics are the same as the
    software screen: ~130 s here against 16-30 cells at ~6 min each, every one
    of which would fail identically.

    ``max_cycle`` bounds a runaway; 0 leaves it unbounded and relies on
    GPGPU-Sim's own deadlock detector, which is on by default and fires long
    before a timeout would.
    """
    from chia.simulators.accelsim import AccelSimNode

    # A trace whose kernelslist is missing or kernel-free is not a failing
    # experiment, it is an absent one -- and it is indistinguishable from
    # success downstream: the simulator initialises, finds nothing to run,
    # exits rc=0 in about a second, and `run_accelsim` fingerprints that as a
    # good result. Every later run then hits the cache and repeats the verdict.
    # Post-processing writes kernelslist.g last, so this also catches the race
    # against a tracer that has not finished.
    n_kernels = 0
    if os.path.exists(kernelslist):
        with open(kernelslist, errors="ignore") as fh:
            n_kernels = sum(
                1 for ln in fh
                if ln.strip() and not ln.strip().lower().startswith("memcpy")
            )
    if n_kernels == 0:
        return False, ("trace incomplete: no kernels in kernelslist "
                       f"({kernelslist}) -- tracer may still be post-processing")

    knobs = {"-power_simulation_enabled": "0"}
    if max_cycle:
        knobs["-gpgpu_max_cycle"] = str(max_cycle)
    r = AccelSimNode.run_accelsim(
        node, kernelslist=kernelslist, arch=arch, outdir=outdir,
        knob_overrides=knobs, collect_counters=False, per_kernel=False,
        timeout_s=timeout_s,
    )
    log = os.path.join(outdir, "sim.log")
    text = open(log, errors="ignore").read() if os.path.exists(log) else ""
    if "deadlock detected" in text:
        return False, "simulator deadlock (check operand alignment / MSHR pressure)"
    if r.timed_out:
        return False, f"timed out after {timeout_s}s"
    if not r.ok:
        return False, (r.stderr_tail or "simulation failed")[:160]
    cycles = r.stats.get("cycles", 0)
    if not cycles:
        # rc==0 with no cycles means nothing was simulated. Drop the
        # fingerprint so the next attempt actually re-runs instead of
        # inheriting this verdict from cache.
        fp = os.path.join(outdir, "fingerprint.txt")
        if os.path.exists(fp):
            os.remove(fp)
        return False, "simulator ran but produced no cycles (cache entry purged)"
    return True, f"ok, {cycles:.0f} cycles"
