"""Autonomous hardware/software co-design loop over the Accel-Sim backend.

``accelsim_probes`` decides which *single* knob is worth measuring.  Everything
that decides what a *set* of measurements means used to live in driver scripts
outside CHIA: the reservation-failure attribution, the composition tests, the
sign-stability band, the marginal-return knee.  Those are the steps that turn a
table into a decision, so they are the steps that have to be in here for the
loop to close without a human reading the table.

The loop is deterministic.  No LLM is consulted: every branch is a threshold on
a measured number, and the thresholds are the module constants below.  Two
runs against the same traces produce the same design.

Chain::

    baseline -> idealization probes -> attribute the limiter
             -> propose hardware knob (from the attribution, not the ranking)
             -> propose software tiles (same attribution)
             -> cross product, because the pair's sign is not the sum of signs
             -> sign-stability band over SM count
             -> area ledger (pay for what was added by removing what is dead)
             -> select

Why the attribution step is load-bearing: at 4096x4096x64 every idealization
probe that touches MSHRs reads exactly 0.00%.  Ranking probes by headroom
therefore says "MSHRs are irrelevant" and stops.  Splitting reservation
failures by *reason* says 100% of them are MISS_QUEUE_FULL -- a different
structure in the same cache, reachable by a different knob.  The ranking finds
the cache; only the attribution finds the structure.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import time
from dataclasses import dataclass, field

try:  # Ray is optional: the pure decision helpers import without it.
    from chia.base.ChiaFunction import ChiaFunction
except Exception:  # pragma: no cover
    ChiaFunction = None  # type: ignore[assignment]

from chia.simulators.accelsim_dse import Tile, LEGAL_TILES, BASELINE_TILE
from chia.simulators.accelsim_probes import HW_TARGET_PCT, NOISE_PCT

# ---------------------------------------------------------------------------
# Thresholds.  Every decision in this module is one of these compared against a
# measured number; there is no other source of authority.
# ---------------------------------------------------------------------------

#: A reason owns a failure class when it accounts for at least this share.
#: Set high because the interesting case is total domination (100% observed),
#: and a split cause means the knob will only buy its share of the headroom.
ATTRIB_DOMINANT_PCT = 60.0

#: Composition: two knobs compose super-additively when the pair beats the
#: better single by more than noise.  Below -NOISE they interfere.
COMPOSE_PCT = NOISE_PCT

#: Marginal return: stop scaling a knob when one more doubling buys less
#: than this.  The knee, not the asymptote -- the asymptote is unaffordable.
KNEE_PCT = 0.5

#: A resource is free to surrender when removing it costs less than this.
FREE_PCT = 1.0

#: SM counts the band sweeps.  The SM axis is deterministically chaotic at
#: +/-5-8% per 2-SM step, so a claim measured at one SM count is not a claim.
#: A band that never crosses zero is quotable; one that straddles it is not.
BAND_SM = (104, 106, 108, 110, 112)

# ---------------------------------------------------------------------------
# Mechanism attribution.  Ported from the standalone driver so the loop can
# read it; this is the only part of the chain that looks inside the cache.
# ---------------------------------------------------------------------------

REASONS = ("RESERVATION_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
           "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING", "LINE_ALLOC_FAIL")
MSHR_REASONS = ("MSHR_ENRTY_FAIL", "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING")
UNITS = ("L1D", "L2")
RW = ("R", "W")


@dataclass
class Attribution:
    """Reservation failures split by (cache, access type, reason).

    ``limiter`` is the decision this class exists to produce: the name of the
    structure to spend area on.  ``write_share`` is reported alongside because
    a write-dominated failure profile and a read-dominated one call for the
    same knob but a different software response.
    """
    counts: dict = field(default_factory=dict)   # (unit, rw, reason) -> float
    total: float = 0.0
    write_share: float = 0.0
    limiter: str = ""            # "miss_queue" | "mshr" | "line_alloc" | ""
    limiter_share: float = 0.0
    dominant_unit: str = ""
    ok: bool = False
    note: str = ""

    def get(self, unit: str, rw: str, reason: str) -> float:
        return self.counts.get((unit, rw, reason), 0.0)


def attribute_failures(run_dir: str) -> Attribution:
    """Sum the final cumulative row of ``perf_counter.csv.gz`` by reason.

    Column names carry a bank or SM index (``L2_bank_007_GLOBAL_ACC_W_...``,
    ``L1D_042_GLOBAL_ACC_R_...``), so matching is prefix-on-unit and
    suffix-on-reason, which collapses the indices together.
    """
    path = os.path.join(run_dir, "perf_counter.csv.gz")
    if not os.path.exists(path):
        return Attribution(ok=False, note=f"no perf_counter.csv.gz in {run_dir}")
    try:
        with gzip.open(path, "rt") as fh:
            rdr = csv.reader(fh)
            head = next(rdr)
            last = None
            for row in rdr:
                if len(row) >= len(head):
                    last = row
    except Exception as exc:
        return Attribution(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
    if last is None:
        return Attribution(ok=False, note=f"{run_dir}: no data rows")

    want: dict = {}
    for i, c in enumerate(head):
        for u in UNITS:
            if not c.startswith(u + "_"):
                continue
            for rw in RW:
                for r in REASONS:
                    if c.endswith(f"GLOBAL_ACC_{rw}_{r}"):
                        want.setdefault((u, rw, r), []).append(i)

    counts: dict = {}
    for k, idx in want.items():
        tot = 0.0
        for i in idx:
            try:
                tot += float(last[i])
            except (ValueError, IndexError):
                pass
        counts[k] = tot

    a = Attribution(counts=counts, ok=True)
    wf = sum(a.get(u, "W", "RESERVATION_FAIL") for u in UNITS)
    rf = sum(a.get(u, "R", "RESERVATION_FAIL") for u in UNITS)
    a.total = wf + rf
    a.write_share = (100.0 * wf / a.total) if a.total else 0.0

    # Which structure? Sum each candidate reason across every (unit, rw) and
    # score it against total RESERVATION_FAIL.  Deliberately global: a knob
    # raises the queue in both caches at once, so the decision is global too.
    mq = sum(a.get(u, w, "MISS_QUEUE_FULL") for u in UNITS for w in RW)
    mshr = sum(a.get(u, w, r) for u in UNITS for w in RW for r in MSHR_REASONS)
    line = sum(a.get(u, w, "LINE_ALLOC_FAIL") for u in UNITS for w in RW)
    ranked = sorted((("miss_queue", mq), ("mshr", mshr), ("line_alloc", line)),
                    key=lambda kv: -kv[1])
    name, val = ranked[0]
    share = (100.0 * val / a.total) if a.total else 0.0
    if a.total and share >= ATTRIB_DOMINANT_PCT:
        a.limiter, a.limiter_share = name, share
    else:
        a.limiter, a.limiter_share = "", share
        a.note = (f"no reason clears {ATTRIB_DOMINANT_PCT:.0f}% "
                  f"(top {name} at {share:.1f}%)")

    per_unit = {u: sum(a.get(u, w, "RESERVATION_FAIL") for w in RW)
                for u in UNITS}
    a.dominant_unit = max(per_unit, key=lambda u: per_unit[u]) if a.total else ""
    return a


def format_attribution(a: Attribution) -> str:
    if not a.ok:
        return f"attribution failed: {a.note}"
    L = [f"  {'cache':<6}{'op':<4}{'RESERVATION_FAIL':>18}"
         f"{'MISS_QUEUE_FULL':>18}{'share':>8}{'MSHR-related':>14}{'share':>8}"]
    for u in UNITS:
        for rw in RW:
            rf = a.get(u, rw, "RESERVATION_FAIL")
            mq = a.get(u, rw, "MISS_QUEUE_FULL")
            ms = sum(a.get(u, rw, r) for r in MSHR_REASONS)
            if rf == 0 and mq == 0:
                continue
            L.append(f"  {u:<6}{rw:<4}{rf:>18,.0f}{mq:>18,.0f}"
                     f"{(100 * mq / rf if rf else 0):>7.1f}%"
                     f"{ms:>14,.0f}{(100 * ms / rf if rf else 0):>7.1f}%")
    L.append(f"  -> write path is {a.write_share:.1f}% of all reservation "
             f"failures ({a.total:,.0f} total)")
    if a.limiter:
        L.append(f"  -> LIMITER: {a.limiter} at {a.limiter_share:.1f}% "
                 f"of failures, worst in {a.dominant_unit}")
    else:
        L.append(f"  -> LIMITER: undecided -- {a.note}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Knob vocabulary.  Probes are idealizations ("what if this were infinite");
# these are buildable design *points*.  They are kept here rather than in a
# driver because the proposer below has to be able to name one.
# ---------------------------------------------------------------------------

#: Stock SM80_A100 values, read from the shipped config.  Every builder below
#: edits exactly one field of these so a diff of the emitted knobs against
#: stock is a diff of the intended field only.
STOCK_DL1 = "S:4:128:64,L:T:m:L:L,A:512:64,16:0,32"
STOCK_DL2 = "S:128:128:16,L:B:m:L:X,A:192:4,32:0,32"
STOCK_PIPE = "4,4,4,4,4,4,4,4,4,4,8,4,4"
EX_WB_INDEX = 10          # writeback width; the only field wbq ever moved
STOCK_MQ_DL1, STOCK_MQ_DL2 = 16, 32

#: Adaptive repartitioning silently undoes an occupancy knob by handing the
#: capacity back.  Measured inert on its own (bit-identical cycles), so it is
#: safe to pin off in every point and removes the confound.
ADAPTIVE_OFF = {"-gpgpu_adaptive_cache_config": "0"}


def dl1_spec(miss_queue: int) -> str:
    """Stock L1D with field 4 (miss queue depth) replaced.  MSHRs untouched."""
    p = STOCK_DL1.split(",")
    p[3] = f"{int(miss_queue)}:0"
    return ",".join(p)


def dl2_spec(miss_queue: int) -> str:
    p = STOCK_DL2.split(",")
    p[3] = f"{int(miss_queue)}:0"
    return ",".join(p)


def queue_knobs(mult: int = 16) -> dict:
    """Deepen both miss queues by ``mult``x.

    This is *not* an MSHR change.  The MSHR groups (``A:512:64`` in L1,
    ``A:192:4`` in L2) are field 3 and are left alone; the miss queue is the
    FIFO a miss sits in after its MSHR is allocated, waiting for an outbound
    port slot.  The distinction is the finding: every MSHR knob measured
    0.00% while 100% of write-path failures were MISS_QUEUE_FULL.
    """
    return {"-gpgpu_cache:dl1": dl1_spec(STOCK_MQ_DL1 * mult),
            "-gpgpu_cache:dl2": dl2_spec(STOCK_MQ_DL2 * mult)}


#: Measured OCC2/OCC4 bundles.  Tabulated rather than computed: the shared
#: memory figure is the hardware's next legal partition size, not a multiple
#: of the stock one (393216 is 384 KiB, not 2 x 167936), and the option list
#: must name every partition the config is allowed to select.
OCCUPANCY_BUNDLE = {
    2: {"-gpgpu_shader_registers": "131072",
        "-gpgpu_shmem_size": "393216",
        "-gpgpu_shmem_sizeDefault": "393216",
        "-gpgpu_unified_l1d_size": "384",
        "-gpgpu_shmem_option": "0,8,16,32,64,164,288,352"},
    4: {"-gpgpu_shader_registers": "262144",
        "-gpgpu_shmem_size": "786432",
        "-gpgpu_shmem_sizeDefault": "786432",
        "-gpgpu_unified_l1d_size": "768",
        "-gpgpu_shmem_option": "0,8,16,32,64,164,288,352,608,736"},
}


def occupancy_knobs(mult: int = 2) -> dict:
    """Scale the four resources that jointly cap CTAs per SM.

    They are co-limiting: raising any one alone moves nothing, because the
    next one binds at the same CTA count.  Hence one bundle, not four knobs --
    the control points that raise only shared memory or only registers both
    measured flat.
    """
    if int(mult) not in OCCUPANCY_BUNDLE:
        raise ValueError(f"no measured occupancy bundle for {mult}x")
    return dict(OCCUPANCY_BUNDLE[int(mult)])


def writeback_knobs(width: int = 4) -> dict:
    """Set EX_WB (writeback bandwidth) only; the other 12 stages stay stock."""
    p = STOCK_PIPE.split(",")
    p[EX_WB_INDEX] = str(int(width))
    return {"-gpgpu_pipeline_widths": ",".join(p)}


def sm_knobs(n_sm: int) -> dict:
    return {"-gpgpu_n_clusters": str(int(n_sm))}


#: Maps an attributed limiter onto the knob that actually reaches it.  This is
#: the step a headroom ranking cannot perform: the ranking says "L1/L2", the
#: attribution says "the queue inside it", and only this table knows which
#: string field that is.
# -- knobs the counter audit found the loop was not driving -------------
# Every one of these was named by an agent reading the baseline counters as
# the structure it actually wanted to change, and every one was refused
# because the hand-written menu did not list it.  The simulator has accepted
# them all along (see ``accelsim_knobs``: 110 architectural options
# registered, 21 reachable from the probe layer, 3 from ``LIMITER_KNOB``).
# They are defined here but deliberately NOT added to ``LIMITER_KNOB``: the
# deterministic chain is the control arm and must keep producing exactly the
# result it produced.  ``EXTENDED_LIMITER_KNOB`` below is what the agent arm
# gets.

def write_alloc_spec(spec: str, wap: str) -> str:
    """Replace the write-allocate policy character in a cache config string.

    The config is ``ct:nset:line:assoc,rp:wp:ap:wap:sif,mshr...`` -- the
    policy is the 4th field of the second group, one character.  The whole
    "we cannot express write policy" limitation was this character.
    """
    parts = spec.split(",")
    grp = parts[1].split(":")
    grp[3] = wap
    parts[1] = ":".join(grp)
    return ",".join(parts)


#: ``N`` no-write-allocate, ``W`` write-allocate, ``F`` fetch-on-write,
#: ``L`` lazy-fetch-on-read (the stock L1D setting).
def cache_type_spec(spec: str, ct: str) -> str:
    """Swap a cache's line/sector type -- the 1st field of the config string.

    Stock L1 and L2 are sectored (`S`), so a warp storing a full 128 B line
    produces four 32 B transactions, each taking its own miss-queue entry.
    `N` (NORMAL) makes the line the unit of transfer, which is the change
    that actually reduces transaction *count*. Removing write-allocate does
    not: the L1 is write-through, so the entry is spent forwarding the data
    to L2 whether or not a line is allocated. That was measured -- 0.00%.
    """
    if ct not in ("N", "S"):
        raise ValueError(f"cache type must be N (line) or S (sector), not {ct!r}")
    parts = spec.split(",")
    parts[0] = ct + parts[0][1:]
    return ",".join(parts)


def line_cache_knobs(unit: str = "N", *, l2: bool = False) -> dict:
    """De-sector the L1 (and optionally the L2)."""
    out = {"-gpgpu_cache:dl1": cache_type_spec(STOCK_DL1, unit)}
    if l2:
        out["-gpgpu_cache:dl2"] = cache_type_spec(STOCK_DL2, unit)
    return out


def mem_port_knobs(ports: int = 2) -> dict:
    """Memory transactions the LSU may issue per core cycle. Stock is 1."""
    if int(ports) < 1:
        raise ValueError("mem_unit_ports must be >= 1")
    return {"-gpgpu_mem_unit_ports": str(int(ports))}


def ldst_buffer_knobs(n: int = 8) -> dict:
    """Response packets buffered in the LD/ST unit's ejection port. Stock 2."""
    if int(n) < 1:
        raise ValueError("ldst response buffer must be >= 1")
    return {"-gpgpu_n_ldst_response_buffer_size": str(int(n))}


def lrc_knobs(entries: int = 256, *, merged: int = 32) -> dict:
    """Turn on the load request coalescer. Stock is off."""
    return {"-gpgpu_lrc_enabled": "1",
            "-gpgpu_lrc_max_entries": str(int(entries)),
            "-gpgpu_lrc_max_merged": str(int(merged))}


WRITE_ALLOC_MODES = ("N", "W", "F", "L")


def write_alloc_knobs(mode: str = "N", *, l2: bool = False) -> dict:
    """Change how a store that misses is allocated.

    ``N`` is the interesting one for a write-once output stream: the stock
    ``L`` allocates a line for every store, and since the L1D here is already
    write-through the allocate buys nothing and costs miss-queue entries.

    ``F`` and ``L`` assert out when the cache allocates ON_FILL rather than
    ON_MISS (gpu-cache.h), so only ``N`` and ``W`` are unconditionally safe
    to ask for on a cache whose allocation policy has also been changed.
    """
    mode = str(mode).upper()
    if mode not in WRITE_ALLOC_MODES:
        raise ValueError(f"write-allocate mode must be one of "
                         f"{', '.join(WRITE_ALLOC_MODES)}, not {mode!r}")
    out = {"-gpgpu_cache:dl1": write_alloc_spec(STOCK_DL1, mode)}
    if l2:
        out["-gpgpu_cache:dl2"] = write_alloc_spec(STOCK_DL2, mode)
    return out


def l1_bank_knobs(banks: int = 4, *, hashing: int = 1,
                  interleave: int = 32) -> dict:
    """Bank the L1 differently.

    Stock is a single bank (``-gpgpu_l1_banks 1``), which is why a perfectly
    coalesced 128-byte store stream still measured 0.64 L1 bank conflicts per
    access.  ``hashing`` selects the bank index function; ``interleave`` is
    the byte granularity at which consecutive addresses change bank.
    """
    if int(banks) < 1:
        raise ValueError("L1 must have at least one bank")
    return {"-gpgpu_l1_banks": str(int(banks)),
            "-gpgpu_l1_banks_hashing_function": str(int(hashing)),
            "-gpgpu_l1_banks_byte_interleaving": str(int(interleave))}


def dram_queue_knobs(size: str = "64:56:32", *, separate: bool = True) -> dict:
    """Deepen the DRAM write queue, optionally splitting it from reads.

    Stock is ``32:28:16`` with a shared queue.  Relevant when the L2 is
    rejecting on a full downstream queue rather than on bandwidth.
    """
    out = {"-dram_write_queue_size": str(size)}
    if separate:
        out["-dram_seperate_write_queue_enable"] = "1"
    return out


def l2_subpartition_knobs(n: int = 2) -> dict:
    """More memory sub-partitions per channel: more independent queues."""
    if int(n) < 1:
        raise ValueError("need at least one sub-partition per channel")
    return {"-gpgpu_n_sub_partition_per_mchannel": str(int(n))}


def gmem_skip_l1_knobs(on: bool = True) -> dict:
    """Send global accesses straight past the L1D.

    The bluntest version of "stop allocating for a stream that is never
    re-read": it removes the L1 write path from the problem entirely, at the
    cost of every global read that *was* hitting.
    """
    return {"-gpgpu_gmem_skip_L1D": "1" if on else "0"}


LIMITER_KNOB = {
    "miss_queue": ("queue", queue_knobs),
    "mshr": ("mshr", lambda mult=4: {
        "-gpgpu_cache:dl1": STOCK_DL1.replace("A:512:64", f"A:{512 * mult}:64"),
        "-gpgpu_cache:dl2": STOCK_DL2.replace("A:192:4", f"A:{192 * mult}:4")}),
    "line_alloc": ("l1_capacity", lambda mult=2: {
        "-gpgpu_unified_l1d_size": str(192 * mult)}),
}

#: What the *agent* arm may propose.  A superset of ``LIMITER_KNOB``, kept
#: separate so the deterministic chain's behaviour is untouched and the two
#: arms stay comparable.  The added entries are the ones a counter-reading
#: agent asked for and was refused.
EXTENDED_LIMITER_KNOB = {
    **LIMITER_KNOB,
    "write_alloc": ("write_alloc", write_alloc_knobs),
    "l1_banks": ("l1_banks", l1_bank_knobs),
    "dram_queue": ("dram_queue", dram_queue_knobs),
    "l2_subpart": ("l2_subpart", l2_subpartition_knobs),
    "gmem_skip_l1": ("gmem_skip_l1", gmem_skip_l1_knobs),
    "line_cache": ("line_cache", line_cache_knobs),
    "mem_ports": ("mem_ports", mem_port_knobs),
    "ldst_buffer": ("ldst_buffer", ldst_buffer_knobs),
    "lrc": ("lrc", lrc_knobs),
}


# ---------------------------------------------------------------------------
# Cross-point decision rules.  Each takes measurements and returns a verdict
# string plus the number that justifies it -- never a bare verdict, so the
# loop's log is auditable after the fact.
# ---------------------------------------------------------------------------

def pct_gain(cycles: float, ref: float) -> float:
    """Positive means faster than ``ref``."""
    return 100.0 * (ref - cycles) / ref if ref else 0.0


@dataclass
class Composition:
    a_pct: float
    b_pct: float
    both_pct: float
    verdict: str          # super-additive | independent | redundant | interfering
    margin: float         # both - best single
    note: str = ""


def composition_verdict(ref: float, a: float, b: float, both: float,
                        *, names: tuple = ("A", "B")) -> Composition:
    """Do two knobs compose?

    Asked because the pair's sign is not the sum of the singles' signs: a
    knob that is a small loss alone can be a win in combination, and the
    reverse.  A sequential search that fixes one axis then the other cannot
    see either case.
    """
    pa, pb, pboth = (pct_gain(a, ref), pct_gain(b, ref), pct_gain(both, ref))
    best = max(pa, pb)
    margin = pboth - best
    if margin > COMPOSE_PCT:
        v = "super-additive"
        note = (f"{names[0]}+{names[1]} beats the better single "
                f"({best:+.2f}%) by {margin:+.2f}%")
    elif margin < -COMPOSE_PCT:
        v = "interfering"
        note = (f"the pair is {margin:+.2f}% behind the better single "
                f"({best:+.2f}%) -- they contend")
    else:
        v = "redundant"
        note = (f"the pair lands within noise of the better single "
                f"({best:+.2f}%); the second knob adds nothing")
    return Composition(pa, pb, pboth, v, margin, note)


@dataclass
class Band:
    points: dict                      # n_sm -> pct gain
    lo: float
    hi: float
    median: float
    stable: bool
    verdict: str                      # WIN | LOSS | STRADDLES
    note: str = ""


def sign_stable_band(gains: dict) -> Band:
    """Is a claim a claim, or an artefact of the SM count it was measured at?

    The SM-count axis is deterministically chaotic: +/-5-8% per 2-SM step,
    which swamps every knob in this study.  So a knob is quoted as the band
    over ``BAND_SM``, and only when the band does not cross zero.  A single
    SM count proves nothing, including when it agrees with the hypothesis.
    """
    vals = sorted(gains.values())
    if not vals:
        return Band({}, 0.0, 0.0, 0.0, False, "STRADDLES", "no points")
    lo, hi = vals[0], vals[-1]
    med = vals[len(vals) // 2]
    if lo > 0:
        v, stable = "WIN", True
        note = f"never negative across {len(vals)} SM counts"
    elif hi < 0:
        v, stable = "LOSS", True
        note = f"never positive across {len(vals)} SM counts -- a stable loss"
    else:
        v, stable = "STRADDLES", False
        note = ("band crosses zero: the sign is set by the SM count, "
                "not by the knob -- not quotable")
    return Band(dict(gains), lo, hi, med, stable, v, note)


@dataclass
class Knee:
    depths: dict                      # mult -> pct gain vs 1x
    knee: int
    note: str = ""


def marginal_knee(gains_by_mult: dict) -> Knee:
    """Smallest multiplier past which one more doubling buys < KNEE_PCT.

    Scaling a structure to infinity is a probe, not a design.  The knee is
    the depth worth paying area for; beyond it the curve is flat and the
    area is wasted.
    """
    ms = sorted(gains_by_mult)
    knee = ms[0] if ms else 1
    for prev, cur in zip(ms, ms[1:]):
        knee = cur
        if gains_by_mult[cur] - gains_by_mult[prev] < KNEE_PCT:
            knee = prev
            break
    return Knee(dict(gains_by_mult), knee,
                f"past {knee}x the next doubling buys < {KNEE_PCT}%")


def free_to_surrender(starve_gains: dict) -> tuple:
    """Split resources into those whose removal is free and those that bind.

    The upward probes only answer "does more of X help".  An iso-area
    reallocation needs "does less of X hurt", which is a different question
    with a different answer: several resources here are inert upward *and*
    free downward, and those are the ones that pay for the knob.
    """
    free = sorted(k for k, v in starve_gains.items() if abs(v) < FREE_PCT)
    bearing = sorted(k for k, v in starve_gains.items() if abs(v) >= FREE_PCT)
    return free, bearing


# ---------------------------------------------------------------------------
# Proposers.  Both are pure functions of measurements: same inputs, same
# proposal.  Neither consults a model.
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    name: str
    knobs: dict
    because: str
    family: str = ""     # knobs from the same family do not compose; see below


def bottleneck_brief(m: int, n: int, k: int, baseline: dict,
                     report: dict, attrib: "Attribution",
                     diag: dict | None = None, *, budget: int = 8,
                     hw_note: str = "") -> str:
    """The hardware analysis, written out as a problem statement for an agent.

    Everything here is read off the run that just finished rather than
    written down in advance.  That is the whole point of calling the kernel
    agent from inside the loop instead of ahead of it: a brief composed by
    hand freezes one run's conclusions into the source, and the next shape --
    K=256 inverts both the hardware pair and the software tile -- then briefs
    the agent with a bottleneck that is no longer there.

    Three things go in, in descending order of how much they constrain the
    answer: what the problem is and how much arithmetic it owes (fixed by the
    shape, and the gate the agent must pass), what the measured limiter is
    (the attribution), and what the idealization probes imply for a *kernel*
    rather than for a knob (``kernel_hints``, each already filtered to fire
    only when its probe reads the way that makes the hint meaningful).

    No tile shape is suggested.  The deterministic arm proposes tiles from
    ``LEGAL_TILES`` and will keep doing so; an agent told which tile to write
    adds nothing that arm does not already cover.
    """
    ffma = m * n * k // 32
    cyc = float(baseline.get("cycles") or 0)
    occ = baseline.get("occupancy")
    cta = baseline.get("cta_per_core")
    L = [f"""Target: C = A x B, single-precision, row-major, on an NVIDIA A100
(sm_80). A is {m}x{k}, B is {k}x{n}, C is {m}x{n}. You write
`void launch(const float *A, const float *B, float *C, int M, int N, int K)`;
the harness owns allocation, initialisation and checking.

The arithmetic the problem requires is fixed: M*N*K = {m * n * k} fused
multiply-adds = {ffma} FFMA warp-instructions. A kernel that produces the
right numbers with materially fewer than that is rejected as not having done
the work.

This project has just measured the baseline and attributed its stalls. What
follows is that measurement, not a guess."""]

    L.append(f"""
Measured baseline (simulated, cycle-accurate):
  cycles        {cyc:,.0f}
  occupancy     {occ}
  resident CTAs {cta} per SM""")

    if attrib is not None and getattr(attrib, "limiter", ""):
        L.append(f"""
Measured limiter: {attrib.limiter}, {attrib.limiter_share:.1f}% of """
                 f"""{attrib.total:,.0f} reservation failures, worst in
{attrib.dominant_unit}. The write path is {attrib.write_share:.1f}% of all
reservation failures.""")
        if attrib.write_share >= 60.0:
            L.append(f"""C is {m}x{n} = {m * n} floats and every one of them is
written exactly once. With K only {k}, the reduction is short, so store
traffic is a large fraction of all traffic and the store path is where the
stalls are -- not the load path.""")

    hints = report.get("kernel_hints") or ()
    if hints:
        L.append("""
What the idealization probes imply for a kernel. Each comes from making one
structure effectively infinite and measuring what that bought, so the
percentage is an upper bound on what any change to that resource can pay:""")
        for h in hints:
            L.append(f"  * {h.get('probe')} ({h.get('headroom_pct')}% "
                     f"headroom): {h.get('hint')}")

    def _f(row, key):
        return row.get(key) if isinstance(row, dict) else getattr(row, key, None)

    # Only the upward probes.  A ``starve_*`` that read flat says the resource
    # can be sold -- a hardware conclusion -- and listing it here as something
    # a kernel must not depend on inverts what it measured.
    up = {_f(r, "name") for r in (report.get("ranked") or ())
          if _f(r, "direction") == "up"}
    flat = sorted(x for x in (report.get("no_headroom") or ())
                  if x in up and not x.startswith("starve"))
    if flat:
        L.append(f"""
Structures measured to be OFF the critical path entirely (idealizing them
changed nothing): {', '.join(flat[:12])}. Do not write a kernel whose
advantage depends on one of these.""")

    d = (diag or {}).get("diagnosis") or {}
    if d.get("mechanism"):
        L.append(f"\nA second agent read the raw counter file and concluded: "
                 f"{str(d['mechanism'])[:600]}")

    if hw_note:
        L.append(f"""
You are being graded on a modified part, not a stock A100: {hw_note}
Write for that machine. A kernel whose advantage is that it keeps that
structure busy is exactly what is wanted here.""")

    L.append(f"""
You are not restricted to any particular decomposition: threadblock tile
shape, register blocking, vectorised access, split-K, a different epilogue
layout -- anything that is correct.

Budget: {budget} simulations. Submit a kernel, read its cycles, and let the
counters choose the next one.""")
    return "\n".join(L)


def propose_hw_points(attrib: "Attribution", report: dict,
                      *, mults=(4, 16)) -> list:
    """Hardware candidates, chosen by the attribution rather than the ranking.

    The idealization ranking is used only as a veto: a knob family whose
    probe measured inert is not proposed, because inert means the simulator
    never applied it and the number is not a measurement.
    """
    out: list = []
    inert = set(report.get("inert", ()) or ())

    if attrib.limiter and attrib.limiter in LIMITER_KNOB:
        label, builder = LIMITER_KNOB[attrib.limiter]
        if attrib.limiter == "mshr" and "many_mshr" in inert:
            pass                       # probe never applied; refuse to price it
        else:
            for m in mults:
                out.append(Proposal(
                    f"{label}{m}", {**ADAPTIVE_OFF, **builder(m)},
                    f"{attrib.limiter} is {attrib.limiter_share:.1f}% of "
                    f"reservation failures (worst in {attrib.dominant_unit})",
                    family=label))

    # An occupancy bundle is proposed only when the limiter is a *per-SM*
    # queueing structure: a second resident CTA is what gives the queue
    # independent work to overlap with.  It is not proposed for a capacity
    # limiter, where a second CTA only adds contention.
    if attrib.limiter in ("miss_queue", "mshr"):
        out.append(Proposal("occ2", {**ADAPTIVE_OFF, **occupancy_knobs(2)},
                            "per-SM queue limiter: a second resident CTA "
                            "gives it independent work to overlap",
                            family="occupancy"))

    # Writeback width is proposed only when the failures are write-dominated.
    if attrib.write_share >= ATTRIB_DOMINANT_PCT:
        out.append(Proposal("wb4", {**ADAPTIVE_OFF, **writeback_knobs(4)},
                            f"write path is {attrib.write_share:.1f}% of "
                            f"failures; EX_WB is the drain",
                            family="writeback"))
    return out


def propose_tiles(attrib: "Attribution", *, baseline_ctas: int,
                  exclude=(), k: int = 4) -> list:
    """Software candidates: synthesise ``-D`` vectors against the template.

    Not free-form codegen.  ``gemm_tpl.cu`` varies in 11 integers and
    self-screens illegal shapes, so the software axis is a search over legal
    ``Tile`` tuples -- which ``Tile.feasible()`` already filters for free,
    before any compiler runs.

    The rule, when the limiter is a per-SM queue: raise resident CTAs above
    the baseline, because the queue stalls on a single CTA's dependent miss
    stream and a second CTA is independent work.  Rank the survivors by
    shared memory ascending, since shared memory is what caps residency.
    """
    from chia.simulators.accelsim_dse import resident_ctas
    cands = []
    for t in LEGAL_TILES:
        if t.name in exclude or t.name == BASELINE_TILE.name:
            continue
        ok, _why = t.feasible()
        if not ok:
            continue
        rc = resident_ctas(t)
        if attrib.limiter in ("miss_queue", "mshr") and rc <= baseline_ctas:
            continue                   # would not add independent work
        cands.append((rc, t))
    cands.sort(key=lambda rt: (-rt[0], rt[1].smem_bytes))
    why = (f"{attrib.limiter} limiter at {baseline_ctas} CTA/SM: raise "
           f"residency so the queue has independent work")
    # Named by trace directory, not by ``Tile.name``: an existing trace is
    # reused only if the name the loop asks for is the name on disk.
    return [Proposal(trace_name(t), {}, why, family="tile")
            for _rc, t in cands[:k]]


# ---------------------------------------------------------------------------
# Evaluation.  One simulated design point, fault-tolerant, flattened to a
# JSON-serialisable record so a failed cell degrades the sweep by one row
# instead of killing the loop.
# ---------------------------------------------------------------------------

def _evaluate_point(accelsim_root: str, cuda_path: str, kernel: str,
                    point: str, kernelslist: str, outdir: str, knobs: dict,
                    power: bool = True, trace_knobs=None,
                    collect_counters: bool = False,
                    timeout_s: int = 10800) -> dict:
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_power import run_with_power
    t0 = time.time()
    try:
        r, v, s = run_with_power(
            AccelSimNode(accelsim_root, cuda_path), kernelslist, outdir=outdir,
            knob_overrides=dict(knobs), sync_xml=power,
            trace_knob_overrides=dict(trace_knobs or {}),
            collect_counters=collect_counters, per_kernel=False,
            timeout_s=timeout_s)
    except Exception as exc:
        return {"kernel": kernel, "point": point, "ok": False, "outdir": outdir,
                "note": f"{type(exc).__name__}: {exc}"[:160],
                "wall_s": time.time() - t0}
    st = r.stats
    ok = bool(r.ok) and bool(st.get("cycles"))
    if power and not v.avg_power_w:
        ok = False
    log = os.path.join(outdir, "sim.log")
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    deadlock = "deadlock detected" in txt
    if deadlock:
        ok = False
    # A knob the simulator did not apply is not a measurement.  Recording the
    # effective value here is what lets the loop tell "no effect" from
    # "never happened" without re-reading the config later.
    #
    # Both maps have to be checked, and against the right config: ``knobs``
    # land in gpgpusim.config, ``trace_knobs`` in trace.config, and a name is
    # only ever read back out of the file it was written into.  Checking only
    # ``knobs`` -- as this did until 2026-09-22 -- left every trace knob in the
    # project silently unverified, which is exactly how a sweep of
    # ``-trace_opcode_latency_initiation_spec_op_3`` got interpreted against
    # the option-parser default instead of the applied value and came out
    # sign-inverted.  ``accelsim_probes.py:669`` has always done it this way.
    _tk = dict(trace_knobs or {})
    unapplied = tuple(
        k for k, want in {**knobs, **_tk}.items()
        if str((r.trace_knobs if k in _tk else r.knobs).get(k, "")).strip()
        != str(want).strip())
    return {
        "kernel": kernel, "point": point, "ok": ok, "outdir": outdir,
        "cycles": st.get("cycles"), "insts": st.get("insts"),
        "occupancy": st.get("occupancy"), "issued_ctas": st.get("issued_ctas"),
        "cta_per_core": st.get("cta_per_core"),
        "dram_reads": st.get("dram_reads"), "dram_writes": st.get("dram_writes"),
        "l2_bw_gbps": st.get("l2_bw_gbps"),
        "power_w": v.avg_power_w if power else None,
        "components": v.components if power else {},
        "inert_power": v.inert if power else [],
        "joules": s.joules if power else None,
        "edp": s.edp if power else None,
        "seconds": s.seconds if power else None,
        "cached": bool(r.cached),
        "stall_mix": (r.stall_mix if collect_counters else {}) or {},
        "effective": {**{k: r.knobs.get(k) for k in knobs},
                      **{k: r.trace_knobs.get(k) for k in _tk}},
        "unapplied": list(unapplied),
        "wall_s": time.time() - t0,
        "note": "" if ok else ("deadlock" if deadlock
                               else (r.stderr_tail or "no cycles")[:150]),
    }


if ChiaFunction is not None:
    #: One simulation per slot.  ``accelsim`` is a CPU-only resource: the
    #: trace already exists by this point, so no GPU is held.
    evaluate_point = ChiaFunction(resources={"accelsim": 1.0},
                                  num_cpus=1)(_evaluate_point)
else:  # pragma: no cover
    evaluate_point = _evaluate_point


# ---------------------------------------------------------------------------
# Software axis: synthesise, screen, build, trace.
# ---------------------------------------------------------------------------

#: Every ``-D`` the template accepts.  The loop may set only these, and only
#: to integers -- which is what makes the software axis safe to drive
#: automatically: the source is fixed and screened once, and the varying part
#: cannot express anything a compiler would execute.
TEMPLATE_DEFINES = ("TB_M", "TB_N", "TB_K", "WP_M", "WP_N", "WP_K",
                    "STAGES", "SPLIT_K", "ALIGN_A", "ALIGN_B",
                    "GEMM_M", "GEMM_N", "GEMM_K")


def trace_name(tile: "Tile") -> str:
    """Directory name a tile's trace lives under.

    The baseline keeps the name ``base`` for continuity with traces that
    already exist; every other tile drops the ``b`` from ``Tile.name``.
    """
    if tile.name == BASELINE_TILE.name:
        return "base"
    return tile.name.replace("tb", "t", 1)


#: The shape the legacy ``traces4k`` directories were captured at, verified
#: from the template's element types (all four tensors are ``float``) against
#: the recorded memcpy sizes: A = 4096*K*4 = 1 MiB gives K = 64.
LEGACY_TRACE_SHAPE = (4096, 4096, 64)


def trace_dir_name(tname: str, m: int, n: int, k: int) -> str:
    """Directory for a tile's trace, qualified by GEMM shape.

    ``trace_name`` encodes the *tile*, not the problem, so two shapes would
    otherwise collide in one directory and a K=256 run would silently replay
    K=64 instructions.  The legacy shape keeps its bare name so the existing
    captures stay reusable; every other shape gets its own directory.
    """
    if (m, n, k) == LEGACY_TRACE_SHAPE:
        return tname
    return f"{tname}__{m}x{n}x{k}"


def write_trace_manifest(trace_root: str, dname: str,
                         m: int, n: int, k: int) -> None:
    """Record the shape next to the trace so reuse can be checked, not assumed."""
    try:
        d = os.path.join(trace_root, dname)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "shape.json"), "w") as fh:
            json.dump({"m": m, "n": n, "k": k}, fh)
    except OSError:
        pass


def trace_shape_ok(trace_root: str, dname: str,
                   m: int, n: int, k: int) -> tuple:
    """(usable, note) for an existing trace directory.

    A manifest that disagrees with the requested shape is a hard refusal --
    replaying the wrong problem size produces numbers that look fine and mean
    nothing.  A missing manifest is only trusted for the legacy shape.
    """
    mf = os.path.join(trace_root, dname, "shape.json")
    if os.path.exists(mf):
        try:
            g = json.load(open(mf))
        except (OSError, ValueError):
            return False, "unreadable shape.json"
        if (g.get("m"), g.get("n"), g.get("k")) != (m, n, k):
            return False, (f"trace is {g.get('m')}x{g.get('n')}x{g.get('k')}, "
                           f"need {m}x{n}x{k}")
        return True, ""
    if (m, n, k) == LEGACY_TRACE_SHAPE:
        return True, "no manifest; legacy dir, shape verified out-of-band"
    return False, "no shape.json and not the legacy shape"


def screen_defines(defines: dict) -> tuple:
    """Refuse anything that is not a known define set to an integer.

    The kernel source never varies, so this is the entire attack surface of
    automated kernel generation here.  Screening it means an ``nvcc`` in this
    loop only ever compiles text the repository already contains.
    """
    bad = []
    for k, v in defines.items():
        if k not in TEMPLATE_DEFINES:
            bad.append(f"unknown define {k!r}")
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            bad.append(f"{k}={v!r} is not an integer")
            continue
        if not (0 < iv <= 1 << 20):
            bad.append(f"{k}={iv} out of range")
    return (not bad), tuple(bad)


def _build_and_trace(accelsim_root: str, cuda_path: str, cutlass_root: str,
                     template: str, tile_json: str, m: int, n: int, k: int,
                     work_dir: str, trace_root: str,
                     timeout_s: int = 3600) -> dict:
    """Compile one synthesised tile, gate it on correctness, then trace it.

    Three stages, each reported separately so a failure names the stage that
    produced it rather than just the tile.  The correctness gate is the
    template's own sentinel: it prints ``OK ...`` on a verified result and
    ``CANNOT_IMPLEMENT`` / ``INIT_FAIL`` / ``RUN_FAIL`` otherwise, so an
    illegal shape costs a compile and no simulation.
    """
    import subprocess
    from chia.simulators.accelsim import AccelSimNode
    t0 = time.time()
    tile = Tile(**json.loads(tile_json))
    name = trace_name(tile)

    ok, why = tile.feasible()
    if not ok:
        return {"tile": name, "ok": False, "stage": "feasibility",
                "note": why, "wall_s": time.time() - t0}

    defines = {"TB_M": tile.tb[0], "TB_N": tile.tb[1], "TB_K": tile.tb[2],
               "WP_M": tile.warp_shape[0], "WP_N": tile.warp_shape[1],
               "WP_K": tile.warp_shape[2], "STAGES": tile.stages,
               "SPLIT_K": tile.split_k,
               "GEMM_M": m, "GEMM_N": n, "GEMM_K": k}
    ok, bad = screen_defines(defines)
    if not ok:
        return {"tile": name, "ok": False, "stage": "screen",
                "note": "; ".join(bad)[:200], "wall_s": time.time() - t0}

    os.makedirs(work_dir, exist_ok=True)
    binary = os.path.join(work_dir, name)
    flags = [f"-D{d}={int(v)}" for d, v in sorted(defines.items())]
    cmd = [os.path.join(cuda_path, "bin", "nvcc"), "-arch=sm_80",
           f"-I{cutlass_root}/include", f"-I{cutlass_root}/tools/util/include",
           "-O3", "--expt-relaxed-constexpr", "--use_fast_math",
           *flags, template, "-o", binary]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"tile": name, "ok": False, "stage": "nvcc",
                "note": "compile timeout", "wall_s": time.time() - t0}
    if r.returncode != 0:
        return {"tile": name, "ok": False, "stage": "nvcc",
                "note": (r.stderr or "")[-300:], "wall_s": time.time() - t0}

    try:
        c = subprocess.run([binary], capture_output=True, text=True,
                           timeout=timeout_s, cwd=work_dir)
    except subprocess.TimeoutExpired:
        return {"tile": name, "ok": False, "stage": "correctness",
                "note": "run timeout", "wall_s": time.time() - t0}
    if c.returncode != 0 or not c.stdout.startswith("OK "):
        return {"tile": name, "ok": False, "stage": "correctness",
                "note": ((c.stdout or "") + (c.stderr or ""))[:300],
                "wall_s": time.time() - t0}

    node = AccelSimNode(accelsim_root, cuda_path)
    dname = trace_dir_name(name, m, n, k)
    tr = AccelSimNode.trace_app(node, app_cmd=[binary],
                                trace_dir=os.path.join(trace_root, dname),
                                cwd=work_dir)
    if tr.ok:
        write_trace_manifest(trace_root, dname, m, n, k)
    return {"tile": name, "ok": bool(tr.ok), "stage": "trace",
            "kernelslist": tr.kernelslist, "ref": (c.stdout or "").strip(),
            "n_kernels": tr.n_kernels,
            "note": "" if tr.ok else (tr.aborted_reason
                                      or (tr.stderr_tail or "")[:150]),
            "wall_s": time.time() - t0}


if ChiaFunction is not None:
    #: Holds a GPU: the tracer runs the kernel on real hardware to record it.
    build_and_trace = ChiaFunction(resources={"accelsim_gpu": 1.0},
                                   num_gpus=1, num_cpus=8)(_build_and_trace)
else:  # pragma: no cover
    build_and_trace = _build_and_trace


# ---------------------------------------------------------------------------
# Area ledger.  A knob that costs area is not free because it made the kernel
# faster; it is free when something dead pays for it.
# ---------------------------------------------------------------------------

#: Miss-queue depth has no home in the AccelWattch XML, so McPAT prices a
#: deeper queue at exactly zero.  That is a blind spot, not a free lunch: the
#: structure is a real FIFO and costs real area.  Measured separately by
#: sweeping McPAT's buffer_sizes, which is non-monotonic in depth -- so what
#: is carried here is the *upper bound* over the sweep, per SM.  A ledger
#: that silently omitted this would report a favourable net area for free.
MISS_QUEUE_MM2_PER_SM_BOUND = 0.0866


@dataclass
class Ledger:
    baseline_mm2: float = 0.0
    design_mm2: float = 0.0
    added_mm2: float = 0.0
    unpriced_mm2: float = 0.0        # area McPAT cannot see; charged anyway
    credit_mm2: float = 0.0
    net_mm2: float = 0.0
    removed: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    ok: bool = False
    note: str = ""


def area_ledger(baseline_dir: str, design_dir: str, *, n_sm: int, n_l2: int,
                xml_name: str = "accelwattch_sass_sim.xml") -> Ledger:
    """Price the design against the baseline, then look for something dead.

    Added area comes from re-running McPAT on the XML the power step already
    rendered for each point, so the price is the model's own answer rather
    than a hand-kept table.  The credit side is certified by counters: a
    structure is removable only when every counter naming it is present and
    exactly zero, which is a stronger test than "it looked idle".
    """
    from chia.simulators.accelsim_area import (
        area_report, read_counters, removal_is_justified, removal_credit,
        REMOVABLE,
    )
    try:
        base = area_report(os.path.join(baseline_dir, xml_name), n_sm, n_l2)
        dsgn = area_report(os.path.join(design_dir, xml_name), n_sm, n_l2)
    except Exception as exc:
        return Ledger(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
    if not (base.ok and dsgn.ok):
        return Ledger(ok=False, note=f"mcpat: {base.note or dsgn.note}")

    lg = Ledger(baseline_mm2=base.total_mm2, design_mm2=dsgn.total_mm2,
                added_mm2=dsgn.total_mm2 - base.total_mm2, ok=True)

    # Counters come from the *design* run: a structure must be dead under the
    # configuration actually being shipped, not under the baseline.
    rep = os.path.join(design_dir, "accelwattch_power_report.log")
    counters = read_counters(rep) if os.path.exists(rep) else {}
    knobs = {}
    cfg = os.path.join(design_dir, "gpgpusim.config")
    if os.path.exists(cfg):
        from chia.simulators.accelsim import read_config_knobs
        knobs = read_config_knobs(cfg)

    keep = []
    for part in REMOVABLE:
        ok, why = removal_is_justified(part, counters, knobs)
        (lg.removed if ok else lg.refused).append(why)
        if ok:
            keep.append(part)
    if keep:
        _per_sm, chip = removal_credit(dsgn, keep)
        lg.credit_mm2 = chip
    # Charge for what McPAT cannot see.  Compare the miss-queue field of each
    # cache spec against stock; if it moved, McPAT priced it at zero and the
    # bound has to be added by hand or the ledger flatters the design.
    # The bound was measured on the L1 structure, so the per-instance count
    # differs by cache: one queue per SM for the L1D, one per L2 bank for the
    # L2.  Carrying the L1-measured bound over to the L2 is an extrapolation
    # and is labelled as one.
    for key, stock_spec, n_inst, how in (
            ("-gpgpu_cache:dl1", STOCK_DL1, n_sm, "measured"),
            ("-gpgpu_cache:dl2", STOCK_DL2, n_l2, "extrapolated from L1")):
        got = str(knobs.get(key, stock_spec))
        try:
            got_q = int(got.split(",")[3].split(":")[0])
            ref_q = int(stock_spec.split(",")[3].split(":")[0])
        except (IndexError, ValueError):
            continue
        if got_q != ref_q:
            charge = MISS_QUEUE_MM2_PER_SM_BOUND * n_inst
            lg.unpriced_mm2 += charge
            lg.warnings.append(
                f"{key} miss queue {ref_q}->{got_q}: McPAT prices this at 0 "
                f"(no XML field); charged the bound ({how}) "
                f"{MISS_QUEUE_MM2_PER_SM_BOUND} mm2 x {n_inst} = "
                f"{charge:.2f} mm2")
    lg.net_mm2 = lg.added_mm2 + lg.unpriced_mm2 - lg.credit_mm2
    return lg


def format_ledger(lg: Ledger) -> str:
    if not lg.ok:
        return f"  area ledger unavailable: {lg.note}"
    L = [f"  baseline {lg.baseline_mm2:>10.2f} mm2",
         f"  design   {lg.design_mm2:>10.2f} mm2   (added {lg.added_mm2:+.3f})"]
    for r in lg.removed:
        L.append(f"  CREDIT   {r}")
    for r in lg.refused:
        L.append(f"  refused  {r}")
    for w in lg.warnings:
        L.append(f"  UNPRICED {w}")
    if lg.unpriced_mm2:
        L.append(f"  unpriced {lg.unpriced_mm2:>10.3f} mm2  (McPAT blind spot)")
    L.append(f"  credit   {lg.credit_mm2:>10.3f} mm2")
    L.append(f"  NET      {lg.net_mm2:>+10.3f} mm2  "
             f"({'pays for itself' if lg.net_mm2 <= 0 else 'unpaid'})")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Fan-out.  Dispatch everything, report as each lands.
# ---------------------------------------------------------------------------

def fan(refs: list, labels: list, desc: str, *, log=print) -> list:
    """Collect ObjectRefs as they complete rather than in submission order.

    ``ray.wait(num_returns=1)`` in a loop so a 40-minute cell does not hide
    the 39 three-minute cells behind it, and so a failure is visible while
    the rest of the wave is still running.
    """
    import ray
    log(f"\n[{desc}] {len(refs)} jobs")
    out, pend, n = [], list(refs), 0
    by_ref = dict(zip(refs, labels))
    t0 = time.time()
    while pend:
        ready, pend = ray.wait(pend, num_returns=1)
        for rf in ready:
            try:
                r = ray.get(rf)
            except Exception as exc:
                r = {"point": by_ref.get(rf, "?"), "ok": False,
                     "note": f"{type(exc).__name__}: {exc}"[:160]}
            out.append(r)
            n += 1
            c = f"{r['cycles']:.0f}" if r.get("cycles") else "-"
            p = f"{r['power_w']:.1f}W" if r.get("power_w") else ""
            cached = " (cached)" if r.get("cached") else ""
            bad = "" if r.get("ok") else f"  FAILED {str(r.get('note',''))[:45]}"
            log(f"  [{n:>3}/{len(refs)}] {str(r.get('point') or r.get('tile')):<26}"
                f" cyc={c:>10} {p:>8}{cached}{bad}")
    log(f"[{desc}] done in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    accelsim_root: str
    cuda_path: str
    cutlass_root: str
    template: str
    trace_root: str
    out_root: str
    work_dir: str
    arch: str = "SM80_A100"
    m: int = 4096
    n: int = 4096
    k: int = 64
    n_sm: int = 108
    max_new_tiles: int = 4
    band_sm: tuple = BAND_SM
    # Reporting only.  EDP differences smaller than this, measured at one SM
    # count, are not resolvable: the SM axis moves every design by +-5..8%
    # per 2-SM step, so a fraction of a percent at sm108 says nothing about
    # which point is better anywhere else.  Points inside this band of the
    # best EDP are printed so that margin is visible, but they do not change
    # the winner -- minimum EDP does, on its own.  See step 8.
    edp_tie_pct: float = 1.0


    # Net area a design may add over the baseline, in mm^2, before it is
    # removed from the ranking.  Zero means iso-area: a point must pay for
    # every structure it grows.  Raise it to explore at a stated budget; the
    # gate reports the figure it enforced either way.
    area_budget_mm2: float = 0.0
    # Arm B.  ``attribute_failures`` reads 24 fixed counters out of 157
    # families and maps the winner to one of three knobs; the agent reads
    # whichever it likes, tests its own hypothesis with simulations, and
    # names its own design.  Both run at step 4 and both designs go into the
    # step-6 cross product, so the comparison is made by measurement rather
    # than by argument.  Set False to get the deterministic chain alone,
    # which is the control arm and what the recorded result was produced by.
    agent_step4: bool = True
    agent_model: str = "claude-sonnet-4-6"
    # 0 means unlimited: the agent decides how many simulations the question
    # needs, and they dispatch in parallel waves.
    agent_sim_budget: int = 0
    agent_timeout_s: int = 7200
    #: Tile names written by the kernel-writing agent (``forge_mnk.py``) and
    #: published into ``trace_root``.  Purely additive: they are appended to
    #: whatever ``propose_tiles`` returns, every existing tile still runs, and
    #: the winner is still chosen by the same EDP-then-cycles rule over the
    #: union.  A forge tile can therefore only win by being faster than
    #: everything already in the grid -- it cannot displace a result, and an
    #: empty tuple reproduces the deterministic run exactly.
    forge_tiles: tuple = ()
    #: Run the kernel-writing agent *inside* step 4, briefed from this run's
    #: own hardware analysis, and append whatever it gets past the gate to the
    #: software proposals.  ``forge_tiles`` above is the same mechanism fed
    #: from a previous, separate run; this is the closed-loop version, and the
    #: difference matters because the brief is then composed from the limiter
    #: that was just measured rather than from one written down months ago.
    agent_forge: bool = False
    forge_budget: int = 8
    #: Where the harness and reference sources live (``harness_mnk.cu``,
    #: ``reference_mnk.cu``).  Empty disables the forge regardless of
    #: ``agent_forge``: there is nothing to build a reference against.
    forge_dir: str = ""
    forge_timeout_s: int = 3600
    #: Which machine candidate kernels are timed on.  ``"proposal"`` uses the
    #: hardware point step 4 just proposed, so the agent is writing for the
    #: part the loop is actually considering; ``"stock"`` times on the
    #: unmodified A100.  A named proposal is used verbatim if it exists.
    forge_grade_on: str = "proposal"


def _rec(records: list, kernel: str, point: str):
    for r in records:
        if r.get("kernel") == kernel and r.get("point") == point:
            return r
    return None


def codesign_loop(cfg: LoopConfig, *, log=print) -> dict:
    """Derive a design from measurements, with no human in the loop.

    Returns the full decision record: every proposal, the measurement that
    licensed it, and the verdict rule that accepted or rejected it.  The
    return value is the audit trail -- a design without the chain that
    produced it is not reproducible.
    """
    import ray
    A, C = cfg.accelsim_root, cfg.cuda_path
    os.makedirs(cfg.out_root, exist_ok=True)
    journal: dict = {"config": cfg.__dict__.copy(), "steps": []}

    def step(name, **kw):
        journal["steps"].append({"step": name, **kw})
        return kw

    def kl(tname: str) -> str:
        dname = trace_dir_name(tname, cfg.m, cfg.n, cfg.k)
        return os.path.join(cfg.trace_root, dname, "traces", "kernelslist.g")

    def trace_usable(tname: str) -> bool:
        """A trace counts as present only if it is the right *problem*.

        Existence of kernelslist.g is not enough: the directory name encodes
        the tile, so a different GEMM shape would replay here unnoticed.
        """
        dname = trace_dir_name(tname, cfg.m, cfg.n, cfg.k)
        if not os.path.exists(kl(tname)):
            return False
        ok, why = trace_shape_ok(cfg.trace_root, dname, cfg.m, cfg.n, cfg.k)
        if not ok:
            log(f"  [trace] {tname}: REFUSED for reuse -- {why}")
        elif why:
            log(f"  [trace] {tname}: reused ({why})")
        return ok

    def ev(kernel, point, knobs, *, power=True, sub="pts", trace_knobs=None):
        outdir = os.path.join(cfg.out_root, kernel, sub,
                              point.replace("/", "_"))
        # ``trace_knobs`` land in trace.config rather than gpgpusim.config.
        # Passing None here -- as this did until 2026-09-23 -- silently
        # dropped them for the five probes that need them (fast_alu,
        # fast_tensor, 2x_subcore, starve_tensor, starve_subcore_half). The
        # probe still ran, it just measured stock hardware, and the report
        # correctly called it INERT: five simulations spent to learn nothing.
        return evaluate_point.chia_remote(
            A, C, kernel, f"{kernel}/{point}", kl(kernel), outdir, knobs,
            power, dict(trace_knobs or {}), False, 10800)

    base_t = trace_name(BASELINE_TILE)
    stock = {**ADAPTIVE_OFF, **sm_knobs(cfg.n_sm)}

    # -- 1. baseline ------------------------------------------------------
    log("\n=== 1. baseline")
    b = fan([ev(base_t, "baseline", stock)], ["baseline"], "baseline", log=log)[0]
    if not b.get("ok"):
        raise SystemExit(f"baseline failed: {b.get('note')}")
    ref_cycles = float(b["cycles"])
    log(f"  {ref_cycles:,.0f} cycles  {b['power_w']:.1f} W  "
        f"{b['joules']*1e3:.2f} mJ  occ {b.get('occupancy')}  "
        f"{b.get('cta_per_core')} CTA/SM")
    step("baseline", cycles=ref_cycles, record=b)

    # -- 2. idealization probes ------------------------------------------
    log("\n=== 2. idealization probes (what is even on the critical path)")
    from chia.simulators.accelsim_dse import CODESIGN_PROBES
    from chia.simulators.accelsim_probes import probe_report, format_probe_report
    prefs, plabels = [], []
    for p in CODESIGN_PROBES:
        prefs.append(ev(base_t, f"probe_{p.name}",
                        {**stock, **p.knobs}, power=False, sub="probe",
                        trace_knobs=p.trace_knobs))
        plabels.append(p.name)
    pres = fan(prefs, plabels, "probes", log=log)

    class _R:                    # probe_report wants AccelSimRunResult-alikes
        def __init__(self, r):
            self.stats = {"cycles": r.get("cycles")}
            self.knobs = r.get("effective") or {}
            # _evaluate_point flattens knob and trace-knob effective values
            # into one "effective" map, so both lookups read the same dict.
            # Leaving this {} made every trace-knob probe look inert.
            self.trace_knobs = r.get("effective") or {}
            self.ok = bool(r.get("ok"))
            self.stderr_tail = r.get("note", "")
            # probe_report reads stdout_tail alongside stderr_tail when a
            # probe fails, and stall_mix unconditionally.  Neither survives
            # the trip through the Ray boundary, so both are supplied here:
            # empty is the honest value, since probes run with
            # collect_counters=False and there is no stall breakdown to carry.
            self.stdout_tail = r.get("stdout_tail", "")
            self.counters = {}
            self.stall_mix = r.get("stall_mix") or {}
    shim = {"baseline": _R(b)}
    for r in pres:
        shim[str(r["point"]).split("/", 1)[-1][len("probe_"):]] = _R(r)
    rep = probe_report(shim, baseline="baseline", probes=CODESIGN_PROBES)
    log(format_probe_report(rep))
    step("probes", report=rep)

    # -- 3. attribute the limiter ----------------------------------------
    log("\n=== 3. attribute reservation failures by reason")
    attrib = attribute_failures(b["outdir"])
    log(format_attribution(attrib))
    step("attribution", limiter=attrib.limiter, share=attrib.limiter_share,
         write_share=attrib.write_share, total=attrib.total, note=attrib.note)
    if not attrib.limiter:
        log("  no dominant reason -- the loop has nothing to target; stopping")
        journal["design"] = None
        return journal

    # -- 4. propose, from the attribution --------------------------------
    log("\n=== 4. proposals")
    hw = propose_hw_points(attrib, rep)
    base_ctas = int(float(b.get("cta_per_core") or 1))
    sw = propose_tiles(attrib, baseline_ctas=base_ctas,
                       k=cfg.max_new_tiles)
    # Agent-written kernels join the software side here.  They are appended,
    # never substituted: ``propose_tiles`` keeps proposing exactly what it
    # proposed before, so the deterministic result stays reproducible and the
    # grid simply gets wider.  A name with no usable trace is dropped with a
    # reason rather than failing the run -- the forge publishes traces
    # independently and may not have finished.
    for tname in (cfg.forge_tiles or ()):
        if any(p.name == tname for p in sw) or tname == base_t:
            continue
        if not trace_usable(tname):
            log(f"  [forge] {tname}: no usable trace, not entered in the grid")
            continue
        sw.append(Proposal(tname, {}, "written by the kernel-forge agent",
                           family="tile"))
        log(f"  [forge] {tname}: entered in the grid")

    # -- 4b. the agent reads the same file, open-endedly ------------------
    # Arm A above is a fixed ratio over 24 counters.  It is fast, it is
    # reproducible, and on this kernel its share metric is *anti-correlated*
    # with cycles at the best design found so far -- so it would rank the
    # winner last.  Arm B is a sandboxed model with the whole counter file,
    # a simulation budget to falsify itself with, and no obligation to pick
    # a knob at all.  Neither arbitrates; step 6 runs both.
    diag = None
    if cfg.agent_step4:
        log("\n--- 4b. agent diagnosis (sandboxed, open reading)")
        try:
            from chia.simulators.accelsim_counters import (
                agent_diagnose, default_knob_menu)
            diag = agent_diagnose(
                os.path.join(b["outdir"], "perf_counter.csv.gz"),
                os.path.join(cfg.out_root, "agent_step4"),
                knob_menu=default_knob_menu(families=EXTENDED_LIMITER_KNOB),
                cycles=ref_cycles, model=cfg.agent_model,
                accelsim_root=A, cuda_path=C, kernelslist=kl(base_t),
                budget=cfg.agent_sim_budget,
                knob_families=EXTENDED_LIMITER_KNOB,
                timeout_seconds=cfg.agent_timeout_s, log=log)
        except Exception as exc:                 # arm B must never take arm A down
            log(f"  agent arm failed: {exc}")
            diag = {"error": str(exc)}
        d = (diag or {}).get("diagnosis") or {}
        runs = (diag or {}).get("runs") or []
        log(f"  agent: {len((diag or {}).get('calls') or [])} reads, "
            f"{len(runs)} simulations, "
            f"limiter={d.get('limiter_family', '(none)')!r}")
        if d.get("mechanism"):
            log(f"  mechanism: {d['mechanism'][:200]}")
        if d.get("wanted_but_undriven"):
            log(f"  wanted but undriven: {d['wanted_but_undriven'][:160]}")
        # Its design joins the cross product only if it named one that
        # builds.  A diagnosis with no runnable proposal is still recorded.
        pk = d.get("proposed_knobs") or {}
        if pk:
            nm = f"agent_{d.get('proposed_family')}{d.get('proposed_mult') or ''}"
            nm = nm.replace("/", "_")
            if nm not in {q.name for q in hw}:
                hw = hw + [Proposal(
                    nm, {**ADAPTIVE_OFF, **pk},
                    "agent: " + (d.get("rationale")
                                 or d.get("mechanism") or "")[:200],
                    family=str(d.get("proposed_family") or "agent"))]
        elif cfg.agent_step4:
            log("  agent named no runnable design; arm A proceeds alone")
        step("agent_diagnosis",
             diagnosis=d, runs=runs,
             report=(diag or {}).get("report", "")[-4000:],
             error=(diag or {}).get("error"))

    # -- 4c. the kernel agent, briefed from this run's own analysis --------
    # The forge used to run ahead of the loop, against a brief whose numbers
    # were literals in a source file.  That is backwards: the loop measures
    # the limiter in step 3 and proposes hardware in step 4, and a kernel
    # written before either happened cannot be a response to either.  Here it
    # is briefed from ``rep``/``attrib``/``diag`` as they stand, and timed on
    # the hardware point the loop just proposed -- because a kernel whose
    # value is that it keeps a widened queue busy measures no better than one
    # that does not, on a part where the queue was never widened.
    #
    # Everything it produces is *appended*.  ``propose_tiles`` still proposes
    # exactly what it proposed before, the grid just gets wider, and a forged
    # kernel can only win by being faster than everything already in it.
    if cfg.agent_forge and cfg.forge_dir:
        log("\n--- 4c. kernel forge (sandboxed, briefed from steps 1-4b)")
        grade, hw_note, brief = {}, "", ""
        if cfg.forge_grade_on != "stock" and hw:
            pick = next((q for q in hw if q.name == cfg.forge_grade_on), hw[0])
            grade, hw_note = dict(pick.knobs), f"{pick.name} -- {pick.because}"
            log(f"  grading candidates on {pick.name}")
        try:
            from chia.simulators.accelsim import AccelSimNode
            from chia.simulators.accelsim_sandbox import forge_kernels
            brief = bottleneck_brief(cfg.m, cfg.n, cfg.k, b, rep, attrib, diag,
                                     budget=cfg.forge_budget, hw_note=hw_note)
            fg = forge_kernels(
                AccelSimNode(A, C),
                forge_dir=cfg.forge_dir,
                out_dir=os.path.join(cfg.out_root, "forge"),
                trace_root=cfg.trace_root, brief=brief,
                mnk=(cfg.m, cfg.n, cfg.k), budget=cfg.forge_budget,
                model=cfg.agent_model, timeout_s=cfg.forge_timeout_s,
                accelsim_root=A, cuda_path=C, sim_sm=cfg.n_sm,
                sim_knobs=grade, hw_note=hw_note, name="forge_loop", log=log)
        except Exception as exc:        # the forge must never take the loop down
            log(f"  forge failed: {type(exc).__name__}: {exc}")
            fg = {"tiles": [], "error": f"{type(exc).__name__}: {exc}"}
        for tname in (fg.get("tiles") or ()):
            if any(q.name == tname for q in sw) or tname == base_t:
                continue
            if not trace_usable(tname):
                log(f"  [forge] {tname}: no usable trace, not entered")
                continue
            sw.append(Proposal(tname, {}, "written by the kernel agent from "
                               f"the step-3 limiter ({attrib.limiter})",
                               family="tile"))
            log(f"  [forge] {tname}: entered in the grid")
        step("forge", brief=brief,
             graded_on=hw_note, tiles=list(fg.get("tiles") or ()),
             accepted=fg.get("accepted"), submissions=fg.get("submissions"),
             agent=fg.get("agent"), error=fg.get("error"))

    for p in hw:
        log(f"  HW  {p.name:<12} because {p.because}")
    for p in sw:
        log(f"  SW  {p.name:<12} because {p.because}")
    step("proposals", hw=[(p.name, p.knobs, p.because) for p in hw],
         sw=[(p.name, p.because) for p in sw])

    # -- 5. trace any software proposal that has no trace yet ------------
    by_name = {trace_name(t): t for t in LEGAL_TILES}
    need = [p.name for p in sw if not trace_usable(p.name)]
    if need:
        log(f"\n=== 5. build+trace {len(need)} new tile(s) (screened, sandboxed)")
        brefs = [build_and_trace.chia_remote(
            A, C, cfg.cutlass_root, cfg.template,
            json.dumps(by_name[nm].__dict__, default=list),
            cfg.m, cfg.n, cfg.k, cfg.work_dir, cfg.trace_root, 3600)
            for nm in need]
        bres = fan(brefs, need, "build+trace", log=log)
        step("build_trace", results=bres)
        dead = {r["tile"] for r in bres if not r.get("ok")}
        sw = [p for p in sw if p.name not in dead]
    else:
        log("\n=== 5. every software proposal already has a trace (reused)")

    # -- 6. cross product -------------------------------------------------
    # Not HW-then-SW: the pair's sign is not the sum of the singles' signs,
    # and a sequential search cannot see the cross term at all.
    log("\n=== 6. cross product: every tile x every hardware point")
    tiles = [base_t] + [p.name for p in sw]
    points = [("stock", stock)] + [(p.name, {**stock, **p.knobs}) for p in hw]
    # Compose across knob *families*.  Two depths of the same structure are
    # the same knob twice: their "composition" is just the deeper one, and
    # reading it as a synergy would be an artefact of the labelling.
    pair = None
    for i, a in enumerate(hw):
        for bq in hw[i + 1:]:
            if a.family and bq.family and a.family != bq.family:
                pair = (a, bq)
                break
        if pair:
            break
    if pair:
        a, bq = pair
        points.append((f"{a.name}+{bq.name}",
                       {**stock, **a.knobs, **bq.knobs}))
    refs, labels = [], []
    for t in tiles:
        if not os.path.exists(kl(t)):
            log(f"  skip {t}: no trace")
            continue
        for pn, pk in points:
            refs.append(ev(t, pn, pk, sub="cross"))
            labels.append(f"{t}/{pn}")
    cross = fan(refs, labels, "cross product", log=log)
    step("cross", records=cross)

    # -- 7. read the cross product ---------------------------------------
    log("\n=== 7. verdicts")
    grid = {}
    for r in cross:
        if r.get("ok"):
            grid[(r["kernel"], str(r["point"]).split("/", 1)[-1])] = r
    log(f"  {'tile':<20}" + "".join(f"{pn:>18}" for pn, _ in points))
    for t in tiles:
        row = f"  {t:<20}"
        for pn, _ in points:
            g = grid.get((t, pn))
            row += (f"{pct_gain(g['cycles'], ref_cycles):>17.2f}%"
                    if g else f"{'-':>18}")
        log(row)

    comps = []
    if pair:
        an, bn = pair[0].name, pair[1].name
        for t in tiles:
            need_all = [grid.get((t, x)) for x in ("stock", an, bn, f"{an}+{bn}")]
            if all(need_all):
                s, ga, gb, gab = (x["cycles"] for x in need_all)
                c = composition_verdict(s, ga, gb, gab, names=(an, bn))
                log(f"  {t:<20} {an}/{bn}: {c.verdict.upper():<15} {c.note}")
                comps.append((t, c.__dict__))
    step("composition", verdicts=comps)

    # -- 8. rank on EDP, then band the winner over the SM axis -----------
    # Minimum EDP is the selection rule, on its own: the design with the
    # lowest energy-delay product wins, with no secondary criterion.  One
    # measurement at one SM count cannot resolve a gap of a few tenths of a
    # percent -- the SM axis swings +-5..8% per 2-SM step -- so every point
    # within `edp_tie_pct` of the best is printed alongside it, with its
    # cycles, to show how much of the margin is real.  That listing is a
    # report, not a vote: it never moves the winner.  Step 8 then bands the
    # selected design over the SM axis, which is where an unresolvable gap
    # actually gets tested.
    ranked = sorted((r for r in cross if r.get("ok") and r.get("edp")),
                    key=lambda r: r["edp"])
    if not ranked:
        log("  nothing scored; stopping")
        journal["design"] = None
        return journal
    # The iso-area constraint used to bind only on the way in -- a point
    # could be *built* only by pairing an increase with a certified-dead
    # removal -- and was audited on the way out, after the winner had already
    # been taken.  Nothing dropped a point that priced out.  This gate closes
    # that: every distinct hardware point is priced against the baseline and
    # anything over budget leaves the ranking before the winner is read off.
    # Area is a property of the configuration, not of the tile, so each point
    # is priced once and the result reused across the tiles that share it.
    base_dir = os.path.join(cfg.out_root, base_t, "pts", "baseline")
    pled: dict = {}
    for r in ranked:
        pn = str(r["point"]).split("/", 1)[-1]
        if pn not in pled:
            pled[pn] = area_ledger(base_dir, r["outdir"],
                                   n_sm=cfg.n_sm, n_l2=160)
    log(f"\n=== 7b. area gate  (budget {cfg.area_budget_mm2:+.2f} mm2 net)")
    priced = {pn: lg for pn, lg in pled.items() if lg.ok}
    if not priced:
        # McPAT answered for nothing at all: that is a broken tool, not a
        # verdict on the designs.  Say so and leave the ranking alone rather
        # than rejecting a whole grid on the strength of a missing binary.
        log("  McPAT priced no point; gate skipped -- results are UNGATED")
        rejected = {}
    else:
        # A point McPAT could not price is rejected, not waived.  The credit
        # side of this ledger is already a lower bound, so an unpriced point
        # is an unsupported area claim and the pessimistic reading is the
        # only one that keeps the constraint meaningful.
        rejected = {pn: (lg.net_mm2 if lg.ok else None)
                    for pn, lg in pled.items()
                    if (not lg.ok) or lg.net_mm2 > cfg.area_budget_mm2}
    for pn in sorted(pled):
        lg = pled[pn]
        if not lg.ok:
            log(f"  {pn:<22} unpriced -- {lg.note}")
        else:
            log(f"  {pn:<22} added {lg.added_mm2:>+9.2f}  "
                f"unpriced {lg.unpriced_mm2:>8.2f}  "
                f"credit {lg.credit_mm2:>8.2f}  "
                f"net {lg.net_mm2:>+10.2f} mm2   "
                f"{'REJECTED' if pn in rejected else 'ok'}")
    gated = [r for r in ranked
             if str(r["point"]).split("/", 1)[-1] not in rejected]
    if not gated:
        # Every point priced out.  Reporting the cheapest design and saying
        # plainly that it is unpaid beats reporting nothing.
        log("  every point is over budget; ranking kept, design is UNPAID")
    else:
        if len(gated) < len(ranked):
            log(f"  {len(ranked) - len(gated)} of {len(ranked)} grid points "
                f"removed; {len(gated)} remain")
        ranked = gated
    step("area_gate", budget_mm2=cfg.area_budget_mm2,
         ledgers={pn: lg.__dict__ for pn, lg in pled.items()},
         rejected=sorted(rejected), enforced=bool(priced and gated),
         kept=len(ranked))

    best_edp = ranked[0]["edp"]
    tied = [r for r in ranked
            if (r["edp"] - best_edp) / best_edp * 100.0 <= cfg.edp_tie_pct]
    win = ranked[0]
    wk, wp = win["kernel"], str(win["point"]).split("/", 1)[-1]
    log(f"\n  best by EDP: {ranked[0]['kernel']}/"
        f"{str(ranked[0]['point']).split('/', 1)[-1]}  "
        f"{ranked[0]['cycles']:,.0f} cyc  {ranked[0]['joules']*1e3:.3f} mJ")
    if len(tied) > 1:
        log(f"  {len(tied)} points within {cfg.edp_tie_pct:.2f}% EDP of it "
            f"-- reported, not selected:")
        for r in sorted(tied, key=lambda r: r["cycles"]):
            log(f"    {r['kernel']}/{str(r['point']).split('/', 1)[-1]:<20}"
                f"{r['cycles']:>10,.0f} cyc  EDP {r['edp']:>9.1f}  "
                f"({(r['edp'] - best_edp) / best_edp * 100.0:+.2f}% EDP)")
    fastest = min(tied, key=lambda r: r["cycles"])
    if fastest is not win:
        # Named, not taken.  A point inside the EDP band can be faster than
        # the winner; saying so is what keeps minimum-EDP an honest rule
        # rather than one that hides what it cost.
        log(f"  not selected: {fastest['kernel']}/"
            f"{str(fastest['point']).split('/', 1)[-1]} is "
            f"{pct_gain(fastest['cycles'], win['cycles']):+.2f}% faster at "
            f"{(fastest['edp'] - best_edp) / best_edp * 100.0:+.2f}% EDP")
    log(f"  selected: {wk}/{wp}  {win['cycles']:,.0f} cyc  "
        f"{win['joules']*1e3:.3f} mJ  EDP {win['edp']:.1f}")

    log(f"\n=== 8. sign-stability band for {wk}/{wp} over SM {cfg.band_sm}")
    wknobs = dict(next(pk for pn, pk in points if pn == wp))
    # Three arms, not two.  The reference for a *design* claim is the
    # baseline tile on stock hardware -- the chip the user already has.
    # Banding the winning tile against itself puts the tile inside both
    # arms, where it cancels: the result is the knob's marginal
    # contribution and the tile's gain is silently discarded.
    #   base = baseline tile + stock knobs    (the chip you have)
    #   ref  = winning  tile + stock knobs    (software change alone)
    #   win  = winning  tile + winning knobs  (the full co-designed point)
    same_tile = (wk == base_t)
    brefs, blabels = [], []
    for nsm in cfg.band_sm:
        arms = [("base", base_t, stock)]
        if not same_tile:                 # else ref is base by construction
            arms.append(("ref", wk, stock))
        arms.append(("win", wk, wknobs))
        for tag, kern, kn in arms:
            kk = {**kn, **sm_knobs(nsm)}
            brefs.append(ev(kern, f"band_{tag}_sm{nsm}", kk, sub="band"))
            blabels.append(f"{tag}_sm{nsm}")
    bres = fan(brefs, blabels, "band", log=log)
    bg = {}
    for r in bres:
        if r.get("ok"):
            bg[str(r["point"]).split("/", 1)[-1]] = r["cycles"]
    if same_tile:
        for nsm in cfg.band_sm:
            if f"band_base_sm{nsm}" in bg:
                bg[f"band_ref_sm{nsm}"] = bg[f"band_base_sm{nsm}"]

    def _band(num, den) -> Band:
        return sign_stable_band({
            nsm: pct_gain(bg[f"band_{num}_sm{nsm}"], bg[f"band_{den}_sm{nsm}"])
            for nsm in cfg.band_sm
            if f"band_{num}_sm{nsm}" in bg and f"band_{den}_sm{nsm}" in bg})

    band = _band("win", "base")           # the design, vs the chip you have
    band_tile = _band("ref", "base")      # software alone
    band_knob = _band("win", "ref")       # hardware alone, on the new tile

    log(f"  {'sm':<6}{'tile':>11}{'knob':>11}{'FULL':>11}")
    for nsm in cfg.band_sm:
        row = f"  {nsm:<6}"
        for bd in (band_tile, band_knob, band):
            v = bd.points.get(nsm)
            row += f"{v:>10.2f}%" if v is not None else f"{'-':>11}"
        log(row)
    for nm, bd in (("tile", band_tile), ("knob", band_knob), ("FULL", band)):
        log(f"  {nm:>5}: {bd.verdict:<10} [{bd.lo:+.2f}%, {bd.hi:+.2f}%]"
            f"  -- {bd.note}")
    step("band", gains=band.points, verdict=band.verdict, lo=band.lo,
         hi=band.hi, stable=band.stable, note=band.note,
         reference=f"{base_t}/stock", tile_band=band_tile.__dict__,
         knob_band=band_knob.__dict__)

    # -- 9. area ledger ---------------------------------------------------
    log(f"\n=== 9. area ledger for {wk}/{wp}")
    lg = area_ledger(os.path.join(cfg.out_root, base_t, "pts", "baseline"),
                     win["outdir"], n_sm=cfg.n_sm, n_l2=160)
    log(format_ledger(lg))
    step("area", ledger=lg.__dict__)

    journal["design"] = {
        "tile": wk, "hw_point": wp, "knobs": wknobs,
        "selection": {
            "rule": "min EDP",
            "edp_best": f"{ranked[0]['kernel']}/"
                        f"{str(ranked[0]['point']).split('/', 1)[-1]}",
            "tied": [{"tile": r["kernel"],
                      "point": str(r["point"]).split("/", 1)[-1],
                      "cycles": r["cycles"], "edp": r["edp"],
                      "edp_pct_over_best":
                          (r["edp"] - best_edp) / best_edp * 100.0}
                     for r in sorted(tied, key=lambda r: r["cycles"])],
            "tie_pct_reported": cfg.edp_tie_pct,
            "fastest_in_tie": f"{fastest['kernel']}/"
                              f"{str(fastest['point']).split('/', 1)[-1]}",
            "fastest_in_tie_is_winner": fastest is win,
        },
        "area": lg.__dict__,
        "area_gate": {"budget_mm2": cfg.area_budget_mm2,
                      "rejected": sorted(rejected),
                      "enforced": bool(priced and gated)},
        "cycles": win["cycles"], "joules": win["joules"], "edp": win["edp"],
        "vs_baseline_pct": pct_gain(win["cycles"], ref_cycles),
        "band": band.__dict__, "quotable": band.stable,
        "band_reference": f"{base_t}/stock",
        "tile_band": band_tile.__dict__, "knob_band": band_knob.__dict__,
        "limiter": attrib.limiter,
    }
    log("\n=== design")
    if band.stable and band.verdict == "WIN":
        log(f"  {wk} + {wp} vs {base_t}/stock: "
            f"{band.lo:+.2f}%..{band.hi:+.2f}% over the SM band, "
            f"sign-stable -- quotable")
        log(f"    of which tile {band_tile.lo:+.2f}%..{band_tile.hi:+.2f}% "
            f"({band_tile.verdict}), knob {band_knob.lo:+.2f}%.."
            f"{band_knob.hi:+.2f}% ({band_knob.verdict})")
    else:
        log(f"  {wk} + {wp} is best at sm{cfg.n_sm} but the band is "
            f"{band.verdict}: {band.note}")
    return journal
