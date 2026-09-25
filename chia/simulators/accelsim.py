"""chia.simulators.accelsim — Accel-Sim / GPGPU-Sim 4.x trace-driven GPU simulation.

Mirrors :mod:`chia.simulators.gem5`. Accel-Sim splits into phases with very
different placement requirements, and the split is the whole reason to put it
in CHIA:

* **Build** (``build_cuda``) — nvcc. CPU only, no GPU.
* **Verify** (``verify_kernel``) — runs a candidate against a reference dump on
  real silicon. Needs a GPU, takes seconds. Gates everything downstream.
* **Trace** (``trace_app``) — runs a CUDA binary under an NVBit ``LD_PRELOAD``
  tool. Needs a *physical GPU*; ~20 s for a small kernel.
* **Simulate** (``run_accelsim``) — replays that trace on a cycle-level model.
  Pure CPU, no GPU, ~3 min. This is the expensive half and the only one worth
  fanning out: measured ~507k thread-instructions/sec, ~7.8e6x slowdown.

One GPU worker produces a trace once; an arbitrary number of CPU-only workers
sweep configurations against it. ``trace_app``/``verify_kernel`` declare
``{"accelsim_gpu": 1.0}``, ``run_accelsim``/``build_cuda`` declare
``{"accelsim": 1.0}``; bind them to different logical workers in the cluster
YAML.

Traces are far too large for the Ray object store (a 256^2 SGEMM is 6.7 MB
compressed; real workloads reach hundreds of GB), so trace *paths* move between
nodes, not trace bytes. Trace and simulate workers must share a filesystem.

This module must be importable on the worker image: ``@ChiaFunction`` bodies
resolve module-level helpers by import on the worker, not by cloudpickle.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import lzma
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:  # ray not installed — node unavailable, dataclasses still usable
    ChiaFunction = None  # type: ignore[assignment]


# Accel-Sim prints aggregate stats to stdout. Each entry maps a logical name to
# a regex whose first group is the value. Table-driven and overridable per call
# so a loop can ask for counters this module never anticipated.
#
# This is the *coarse* surface. The fine surface is ``perf_counter.csv.gz``
# (15,448 counters at 500-cycle resolution) — see ``read_perf_counters``.
DEFAULT_STATS_PATTERNS: dict[str, str] = {
    # Top-line performance
    "cycles": r"gpu_tot_sim_cycle\s*=\s*([0-9]+)",
    "insts": r"gpu_tot_sim_insn\s*=\s*([0-9]+)",
    "ipc": r"gpu_tot_ipc\s*=\s*([0-9.]+)",
    "occupancy": r"gpu_occupancy\s*=\s*([0-9.]+)%",
    "warp_insts": r"gpgpu_n_tot_w_icount\s*=\s*([0-9]+)",
    # CTA supply vs resident capacity. Without these the SM-count probes cannot
    # be told apart from wave quantization -- see wave_quantization().
    "issued_ctas": r"gpu_tot_issued_cta\s*=\s*([0-9]+)",
    "cta_per_core": r"CTA/core\s*=\s*([0-9]+)",
    # Memory system
    "l2_bw_gbps": r"L2_BW\s*=\s*([0-9.]+)\s*GB/Sec",
    "dram_reads": r"total dram reads\s*=\s*([0-9]+)",
    "dram_writes": r"total dram writes\s*=\s*([0-9]+)",
    "l1_bank_conflicts": r"gpgpu_n_l1cache_bkconflict\s*=\s*([0-9]+)",
    "shmem_bank_conflicts": r"gpgpu_n_shmem_bkconflict\s*=\s*([0-9]+)",
    # Stall breakdown
    "stall_gl_mem_resource": r"gpgpu_stall_shd_mem\[gl_mem\]\[resource_stall\]\s*=\s*([0-9]+)",
    "stall_gl_mem_coal": r"gpgpu_stall_shd_mem\[gl_mem\]\[coal_stall\]\s*=\s*([0-9]+)",
    "stall_dram_full": r"gpu_stall_dramfull\s*=\s*([0-9]+)",
    "stall_icnt2sh": r"gpu_stall_icnt2sh\s*=\s*([0-9]+)",
    # Simulation cost — lets a loop budget its own remaining iterations
    "sim_rate_ips": r"gpgpu_simulation_rate\s*=\s*([0-9]+)\s*\(inst/sec\)",
    "silicon_slowdown": r"gpgpu_silicon_slowdown\s*=\s*([0-9]+)x",
}

# Opcodes that constitute *algorithmic* work for a floating-point kernel. The
# work-preservation gate counts these and nothing else: address arithmetic
# (IMAD.WIDE), loads (LDG) and control flow are exactly what a legitimate
# optimization is supposed to remove, so gating on total instruction count
# would reject the optimizations we are trying to find.
DEFAULT_WORK_OPCODES = (
    "FFMA", "FADD", "FMUL", "FMNMX", "MUFU",
    "DFMA", "DADD", "DMUL",
    "HFMA2", "HADD2", "HMUL2",
    "HMMA", "IMMA",
)

# Opcode *forms* that carry an arithmetic mnemonic but do no arithmetic, and so
# must be dropped before the suffix collapsing in work_preserved() folds them
# into the family above. There is one, and it matters: ptxas materialises an
# immediate with `HFMA2.MMA Rd, -RZ, RZ, imm, imm` -- a half-precision FMA
# whose multiplicands are both the zero register, i.e. a MOV wearing an FMA's
# opcode. Verified on disassembly; every HFMA2.MMA in both the reference and a
# tiled candidate has that operand pattern.
#
# It appears a few thousand times in any kernel and the count is a compiler
# detail, so it moves with the code even when the arithmetic does not. That is
# exactly the false-negative direction this gate cannot afford: a shared-memory
# tiled SGEMM whose FFMA count matched the reference *exactly* (524288 vs
# 524288) was rejected at ratio 0.9961 because it materialised 4096 immediates
# where the reference materialised 6144. With this set applied the same pair
# scores 1.0000.
#
# Caveat this is narrow: ptxas also emits HFMA2.MMA for genuine half2 FMAs, and
# the histogram keeps no operands, so a real fp16 kernel would be under-counted
# here. Pass `ignore_forms=()` when the reference itself is half-precision.
NON_WORK_FORMS = frozenset({"HFMA2.MMA"})

# Columns of perf_counter.csv.gz that are global scalars rather than members of
# a per-unit family. Everything not matching _PER_UNIT_RE is treated as global.
_PER_UNIT_RE = re.compile(
    r"^(L2_bank_\d+|L1D_\d+|LRC_subpartition|chiplet|dram_reads_per_mc_\d+|"
    r"dram_writes_per_mc_\d+|L2_dram_queue_full_\d+|m_num_sim_winsn_\d+|"
    r"gpgpu_n_tensor_core_inst_issued_\d+)_?(.*)$"
)


# ======================================================================
# Result types
# ======================================================================

@dataclass
class CudaBuildResult:
    """One nvcc invocation."""

    binary: str
    sources: list[str] = field(default_factory=list)
    cmd: str = ""
    returncode: int = 0
    timed_out: bool = False
    wall_s: float = 0.0
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and os.path.exists(self.binary)


@dataclass
class VerifyResult:
    """Correctness of a candidate kernel against a reference output dump.

    ``ok`` is the gate: a candidate that is not ``ok`` must never be traced or
    simulated, because a kernel that skips work looks like a speedup.
    """

    ok: bool = False
    reason: str = ""
    dump: str = ""
    n_elems: int = 0
    n_mismatch: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    nan_or_inf: bool = False
    returncode: int = 0
    timed_out: bool = False
    wall_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class AccelSimTraceResult:
    """A trace directory produced on a GPU worker.

    ``kernelslist`` is the path handed to ``accel-sim.out -trace``; every other
    field exists so a loop can decide whether simulating this trace is
    affordable *before* it commits a CPU worker for hours.
    """

    kernelslist: str
    trace_dir: str
    app_cmd: str
    n_kernels: int = 0
    trace_bytes: int = 0
    gpu_id: str = ""
    opcodes: dict[str, int] = field(default_factory=dict)
    est_thread_insts: int = 0
    est_sim_seconds: float = 0.0
    returncode: int = 0
    timed_out: bool = False
    aborted_reason: str = ""
    wall_s: float = 0.0
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.aborted_reason
            and self.n_kernels > 0
        )

    def work_ops(self, opcodes: tuple[str, ...] = DEFAULT_WORK_OPCODES) -> int:
        """Warp-instruction count of arithmetic work in this trace.

        This is the invariant a kernel rewrite must preserve. For a 256^3 SGEMM
        it is exactly 256^3/32 = 524288 FFMA, independent of tiling, blocking or
        vectorization — all of which change the *load* count instead.
        """
        return sum(n for op, n in self.opcodes.items() if op.split(".")[0] in opcodes)


@dataclass
class AccelSimRunResult:
    """One simulated configuration."""

    config_dir: str
    kernelslist: str
    stats: dict[str, float] = field(default_factory=dict)
    knobs: dict[str, str] = field(default_factory=dict)
    trace_knobs: dict[str, str] = field(default_factory=dict)
    per_kernel: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    returncode: int = 0
    timed_out: bool = False
    wall_s: float = 0.0
    cached: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def cycles(self) -> float | None:
        return self.stats.get("cycles")

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and "cycles" in self.stats

    @property
    def stall_mix(self) -> dict[str, float]:
        """Fraction of scheduler cycles spent idle / dependency-stalled /
        structurally-stalled. The single most direct bottleneck signal the
        simulator produces; see ``read_perf_counters``."""
        return self.counters.get("stall_mix", {})


# ======================================================================
# Process plumbing
# ======================================================================

def _stderr_digest(stderr: str, head_lines: int = 6) -> str:
    """Keep the head of stderr as well as the tail.

    A failed assertion writes its message as the *first* line and then dumps a
    ~15-frame backtrace, so ``_tail`` alone preserves the useless
    ``=== CRASH: SIGABRT ===`` banner and discards the one line that says which
    invariant was violated.
    """
    if not stderr.strip():
        return ""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    head = lines[:head_lines]
    tail_txt = _tail("\n".join(lines[head_lines:])) if len(lines) > head_lines else ""
    return "\n".join(head) + (f"\n...\n{tail_txt}" if tail_txt.strip() else "")


def _run_logged(
    cmd: list[str] | str,
    cwd: str | None,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, bool, float]:
    """Run *cmd* in its own process group; return
    ``(returncode, stdout, stderr, timed_out, wall_s)``.

    ``start_new_session=True`` puts the subprocess tree in one process group so
    the timeout path can ``killpg`` every descendant — a long accel-sim run that
    must be abandoned otherwise leaves children holding the captured pipes open.
    """
    t0 = time.time()
    full_env = {**os.environ, **env} if env else None
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=full_env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stdout, stderr, False, time.time() - t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return -1, stdout, stderr, True, time.time() - t0


def _tail(text: str, max_bytes: int = 3000) -> str:
    """Keep the end of a stream — errors surface last, and this has to fit an
    LLM context window."""
    if len(text) <= max_bytes:
        return text
    return "...[truncated]...\n" + text[-max_bytes:]


# ======================================================================
# Stat extraction
# ======================================================================

def parse_accelsim_stats(
    stdout: str, patterns: dict[str, str] | None = None
) -> dict[str, float]:
    """Pull aggregate counters out of accel-sim stdout.

    Takes the *last* match for each pattern: aggregate counters are reprinted
    after every kernel, and the final print is the whole-run total.
    """
    pats = {**DEFAULT_STATS_PATTERNS, **(patterns or {})}
    out: dict[str, float] = {}
    for name, pat in pats.items():
        matches = re.findall(pat, stdout)
        if matches:
            try:
                out[name] = float(matches[-1])
            except ValueError:
                pass
    return out


def parse_per_kernel_stats(
    stdout: str, patterns: dict[str, str] | None = None
) -> list[dict]:
    """Split stdout into per-kernel blocks and parse each one.

    Accel-Sim prints a full stat dump at every kernel boundary, led by
    ``kernel_name = ...`` / ``kernel_launch_uid = ...``. Aggregate-only parsing
    is fine for a single-launch kernel but hides everything in a multi-kernel
    app — and a tiling rewrite can turn one launch into several.

    Important caveat baked into the output: ``gpu_sim_cycle`` and
    ``gpu_sim_insn`` are reset per kernel by ``gpgpu_sim::update_stats()``, but
    ``shader_core_stats`` (every ``gpgpu_n_*``, every cache counter) is memset
    once at construction and never reset. So the ``gpgpu_n_*`` family is
    cumulative-from-boot. This function therefore reports both the raw value and
    a ``_delta`` against the previous kernel, which is the real per-kernel
    figure.
    """
    blocks = re.split(r"(?=kernel_name\s*=\s*)", stdout)
    per_kernel: list[dict] = []
    prev: dict[str, float] = {}
    for blk in blocks:
        m = re.search(r"kernel_name\s*=\s*(\S+)", blk)
        if not m:
            continue
        entry: dict = {"kernel_name": m.group(1)}
        uid = re.search(r"kernel_launch_uid\s*=\s*([0-9]+)", blk)
        if uid:
            entry["kernel_launch_uid"] = int(uid.group(1))
        # Per-kernel (reset) counters.
        for key, pat in (
            ("cycles", r"gpu_sim_cycle\s*=\s*([0-9]+)"),
            ("insts", r"gpu_sim_insn\s*=\s*([0-9]+)"),
            ("ipc", r"gpu_ipc\s*=\s*([0-9.]+)"),
            ("occupancy", r"gpu_occupancy\s*=\s*([0-9.]+)%"),
        ):
            mm = re.search(pat, blk)
            if mm:
                entry[key] = float(mm.group(1))
        # Cumulative counters -> report the delta, which is the per-kernel value.
        stats = parse_accelsim_stats(blk, patterns)
        for key, val in stats.items():
            if key in ("cycles", "insts", "ipc", "occupancy"):
                continue
            entry[key] = val
            if key in prev:
                entry[f"{key}_delta"] = val - prev[key]
        prev = {**prev, **stats}
        per_kernel.append(entry)
    return per_kernel


def read_perf_counters(
    run_dir: str,
    *,
    series_keys: tuple[str, ...] = (
        "gpu_sim_cycle", "gpu_sim_insn", "occupancy",
        "shader_cycle_distro_0", "shader_cycle_distro_1", "shader_cycle_distro_2",
    ),
    max_series_points: int = 24,
) -> dict:
    """Digest ``perf_counter.csv.gz`` — the fine-grained counter surface.

    This Accel-Sim fork emits 15,448 counters every 500 cycles (registered in
    ``gpgpu_sim``'s constructor, ``gpu-sim.cc:1161-1276``) for ~169 KB gzipped
    per run. Almost none of it is on stdout. Ignoring it and regexing 17 numbers
    off stdout, which is what this module used to do, throws away the only
    signal that localizes a bottleneck rather than merely totalling it.

    Returns a *digest*, not the raw table, because the consumer is an LLM
    context window:

    ``final``
        every global scalar counter at the last sample (44 of them).
    ``stall_mix``
        ``shader_cycle_distro_{0,1,2}`` normalized. These are mutually exclusive
        per-scheduler-per-cycle buckets incremented at ``shader.cc:1786``:
        ``idle`` (no warp had a valid instruction — front-end/latency bound),
        ``dependency`` (scoreboard collision, i.e. waiting on an outstanding
        memory op), ``structural`` (ready but no free functional unit).
        This is the closest thing the simulator has to a direct answer to
        "what is the bottleneck".
    ``per_unit``
        each per-unit family (108 L1D caches, 160 L2 banks, ...) collapsed to
        ``sum``/``max``/``mean``/``imbalance``. ``imbalance`` = max/mean - 1,
        which is how load skew across SMs or L2 banks shows up; a hot bank is
        invisible in a sum.
    ``series``
        a downsampled time series of ``series_keys`` as per-interval deltas
        (counters are cumulative), so a loop can see a phase change instead of
        one whole-run average.
    """
    path = os.path.join(run_dir, "perf_counter.csv.gz")
    if not os.path.exists(path):
        cands = sorted(glob.glob(os.path.join(run_dir, "perf_counter*.csv.gz")))
        if not cands:
            return {}
        path = cands[-1]

    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        rows = [ln.rstrip("\n").split(",") for ln in fh if ln.strip()]
    if not rows:
        return {}

    def fval(s: str) -> float:
        try:
            v = float(s)
        except ValueError:
            return 0.0
        # occupancy is emitted as "-nan" until the first warp launches (0/0);
        # a NaN here poisons every downstream mean/any()/max().
        return v if v == v and v not in (float("inf"), float("-inf")) else 0.0

    idx = {name: i for i, name in enumerate(header)}
    last = rows[-1]

    global_names = [n for n in header if not _PER_UNIT_RE.match(n)]
    final = {n: fval(last[idx[n]]) for n in global_names if idx[n] < len(last)}

    # Stall mix.
    d0 = final.get("shader_cycle_distro_0", 0.0)
    d1 = final.get("shader_cycle_distro_1", 0.0)
    d2 = final.get("shader_cycle_distro_2", 0.0)
    tot = d0 + d1 + d2
    stall_mix = (
        {"idle": d0 / tot, "dependency": d1 / tot, "structural": d2 / tot}
        if tot
        else {}
    )

    # Per-unit families -> {family: {stat: {sum,max,mean,imbalance}}}
    fams: dict[str, dict[str, list[float]]] = {}
    for n in header:
        m = _PER_UNIT_RE.match(n)
        if not m:
            continue
        # The unit index lives in the prefix for most families
        # (``L1D_000_GLOBAL_ACC_R_MISS``) but in the suffix for the two that
        # have a bare prefix (``LRC_subpartition_num_icnt_to_lrc_sectors_0``).
        # Strip only the trailing index, never every digit -- a blanket
        # re.sub(r"\d+", "N", ...) turns L2_bank_000 into LN_bank_N.
        prefix, rest = m.group(1), m.group(2) or ""
        if prefix[-1].isdigit():
            fam = re.sub(r"_?\d+$", "", prefix) + "_N"
            stat = rest or "value"
        else:
            fam = prefix + "_N"
            stat = re.sub(r"_?\d+$", "", rest) or "value"
        if idx[n] < len(last):
            fams.setdefault(fam, {}).setdefault(stat, []).append(fval(last[idx[n]]))

    per_unit: dict[str, dict] = {}
    for fam, stats in fams.items():
        for stat, vals in stats.items():
            if not any(vals):
                continue  # drop the enormous tail of all-zero access-type columns
            s, mx = sum(vals), max(vals)
            mean = s / len(vals)
            per_unit.setdefault(fam, {})[stat] = {
                "sum": s,
                "max": mx,
                "mean": mean,
                "imbalance": (mx / mean - 1.0) if mean else 0.0,
                "n_units": len(vals),
            }

    # Downsampled per-interval deltas.
    step = max(1, len(rows) // max_series_points)
    picked = rows[::step]
    if picked[-1] is not rows[-1]:
        picked.append(rows[-1])
    series: dict[str, list[float]] = {}
    for key in series_keys:
        if key not in idx:
            continue
        vals = [fval(r[idx[key]]) for r in picked if idx[key] < len(r)]
        if key == "occupancy":
            series[key] = [round(v, 4) for v in vals]  # a ratio, not cumulative
        else:
            # gpu_sim_cycle/gpu_sim_insn are reset per kernel in update_stats()
            # (gpu-sim.cc:1456), so a negative delta means a kernel boundary --
            # there the progress within the interval is the new value itself.
            series[key] = [round(vals[0], 2)] + [
                round(b - a if b >= a else b, 2) for a, b in zip(vals, vals[1:])
            ]
        if not any(series[key]):
            # Registered but never advanced (gpu_tot_sim_cycle is like this).
            # Returning a zero series invites a consumer to conclude "no work".
            del series[key]

    return {
        "source": os.path.basename(path),
        "n_samples": len(rows),
        "final": final,
        "stall_mix": stall_mix,
        "per_unit": per_unit,
        "series": series,
    }


# ======================================================================
# Trace introspection — the work-preservation measurement
# ======================================================================

_OPCODE_SKIP_RE = re.compile(r"^(R[0-9]+|RZ|P[0-9]+|PT|UR[0-9]+|URZ|UP[0-9]+|NoVal|Val)$")
_OPCODE_TOK_RE = re.compile(r"^[A-Z][A-Z0-9_.]*$")


def opcode_histogram(trace_dir: str, max_bytes: int = 0) -> dict[str, int]:
    """Count SASS opcodes across every raw trace in *trace_dir*.

    Counts *warp* instructions, one per trace line, which is what the trace
    records. This is the measurement that makes "did the rewrite skip work?" a
    fact rather than a judgement call: for SGEMM the FFMA count is exactly
    ``N^3 / 32`` regardless of tiling, blocking or vectorization, while the
    LDG count is what a good rewrite collapses. A gate on *total* instructions
    would reject precisely the optimizations we are searching for.

    Run this before the raw ``.trace.xz`` files are deleted — the post-processed
    ``.tracez`` is not in this format.
    """
    hist: dict[str, int] = {}
    files = [
        p for p in sorted(glob.glob(os.path.join(trace_dir, "*.trace*")))
        if not p.endswith(".tracez")
    ]
    read = 0
    for path in files:
        opener = lzma.open if path.endswith(".xz") else open
        with opener(path, "rt", errors="ignore") as fh:  # type: ignore[operator]
            for line in fh:
                read += len(line)
                if max_bytes and read > max_bytes:
                    return hist
                if not line or line[0] in "-#\n":
                    continue
                for tok in line.split():
                    if _OPCODE_TOK_RE.match(tok) and not _OPCODE_SKIP_RE.match(tok):
                        hist[tok] = hist.get(tok, 0) + 1
                        break
    return hist


def work_preserved(
    candidate: dict[str, int],
    reference: dict[str, int],
    *,
    opcodes: tuple[str, ...] = DEFAULT_WORK_OPCODES,
    min_ratio: float = 0.999,
    ignore_forms: frozenset[str] | tuple[str, ...] = NON_WORK_FORMS,
) -> dict:
    """Did a rewritten kernel keep doing the arithmetic the original did?

    Compares only the arithmetic opcodes in *opcodes*, collapsed over SASS
    suffixes (``FFMA.FTZ`` counts as ``FFMA``). Output equality already proves
    the kernel computes the right answer; this proves it got there by doing the
    work rather than by exploiting something about the harness. The two checks
    fail in different ways and both are cheap, so run both.

    ``min_ratio`` defaults to 0.999 rather than 1.0 to tolerate tail/predication
    differences at block boundaries, not to tolerate algorithmic shortcuts.
    Setting it below ~0.9 makes the gate meaningless.
    """
    ignored = frozenset(ignore_forms)

    def counts(h: dict[str, int]) -> dict[str, int]:
        return {op: n for op, n in h.items() if op not in ignored}

    candidate, reference = counts(candidate), counts(reference)

    def total(h: dict[str, int]) -> int:
        return sum(n for op, n in h.items() if op.split(".")[0] in opcodes)

    cand, ref = total(candidate), total(reference)
    ratio = (cand / ref) if ref else 0.0
    by_op = {}
    for op in sorted({o.split(".")[0] for o in list(candidate) + list(reference)}):
        if op not in opcodes:
            continue
        c = sum(n for o, n in candidate.items() if o.split(".")[0] == op)
        r = sum(n for o, n in reference.items() if o.split(".")[0] == op)
        if c or r:
            by_op[op] = {"candidate": c, "reference": r}
    return {
        "ok": ratio >= min_ratio,
        "work_ops_candidate": cand,
        "work_ops_reference": ref,
        "ratio": ratio,
        "by_opcode": by_op,
        "total_insts_candidate": sum(candidate.values()),
        "total_insts_reference": sum(reference.values()),
    }


# ======================================================================
# Config files
# ======================================================================

def read_config_knobs(config_path: str) -> dict[str, str]:
    """Parse a ``gpgpusim.config`` / ``trace.config`` into ``{knob: value}``."""
    knobs: dict[str, str] = {}
    if not os.path.exists(config_path):
        return knobs
    with open(config_path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line.startswith("-"):
                continue
            parts = line.split(None, 1)
            knobs[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return knobs


def write_config_knobs(src_config: str, dst_config: str, overrides: dict[str, str]) -> dict[str, str]:
    """Copy *src_config* to *dst_config*, replacing the given knobs.

    Returns the knobs that were actually applied. A knob absent from the source
    is appended rather than dropped, so a caller can introduce one the base
    config never set.

    Rewriting the knob in place (rather than appending a duplicate) matters:
    GPGPU-Sim's option parser takes the *last* occurrence, so appending works
    but leaves the file self-contradictory and hard for a model to read back.
    """
    remaining = dict(overrides)
    lines_out: list[str] = []
    with open(src_config) as fh:
        for line in fh:
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith("-"):
                knob = stripped.split(None, 1)[0]
                if knob in remaining:
                    lines_out.append(f"{knob} {remaining.pop(knob)}\n")
                    continue
            lines_out.append(line)
    for knob, value in remaining.items():
        lines_out.append(f"{knob} {value}\n")
    os.makedirs(os.path.dirname(os.path.abspath(dst_config)), exist_ok=True)
    with open(dst_config, "w") as fh:
        fh.writelines(lines_out)
    return overrides


def config_fingerprint(
    kernelslist: str, arch: str, knobs: dict | None, trace_knobs: dict | None
) -> str:
    """Stable hash of everything that determines a simulation's result.

    Simulation is deterministic given a trace, so two runs with this
    fingerprint produce bit-identical output. LLM-driven loops re-propose the
    same configuration constantly; at ~3 minutes a run, deduplicating is worth
    more than most of the knobs being swept.
    """
    payload = json.dumps(
        {
            "trace": os.path.abspath(kernelslist),
            "arch": arch,
            "knobs": dict(sorted((knobs or {}).items())),
            "trace_knobs": dict(sorted((trace_knobs or {}).items())),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _trace_newer_than(kernelslist: str, log_path: str) -> bool:
    """True if the trace was rewritten after *log_path* was written.

    :func:`config_fingerprint` keys on the trace's **path**, which is not a
    stable identity for the trace itself. The kernel forge traces every
    candidate into a directory named after its label, so a model that revises
    a kernel and resubmits it under the same label writes a new trace at the
    same path. On the fingerprint alone that is a cache hit, and the previous
    kernel's cycle count comes back for the new kernel -- silently, flagged
    ``cached``, with nothing in the record to show the two differ.

    Comparing mtimes is the ordinary make-style staleness test and settles it
    without re-hashing traces that reach 139 MB. It also leaves every existing
    cache entry valid, since those traces predate the runs that read them.
    Unreadable or missing files answer True: re-simulating costs minutes,
    while trusting a stale result costs the measurement.
    """
    try:
        ref = os.stat(log_path).st_mtime_ns
        base = os.path.dirname(os.path.abspath(kernelslist))
        paths = [kernelslist]
        with open(kernelslist) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln.endswith(".tracez"):
                    paths.append(ln if os.path.isabs(ln)
                                 else os.path.join(base, ln))
        return any(os.stat(f).st_mtime_ns > ref for f in paths)
    except OSError:
        return True


# ======================================================================
# Nodes
# ======================================================================

class AccelSimNode:
    """Build, verify, trace and simulate CUDA kernels.

    Operational paths are bound at construction so callers (and, later, an LLM
    driving a tool server) choose *what* to run, never where the install lives.
    """

    _MEMBER_FNS = ("build_cuda", "verify_kernel", "trace_app", "run_accelsim")
    _DEFAULT_BUNDLE = {"CPU": 1, "accelsim": 1.0}

    def __init__(self, accelsim_root: str, cuda_path: str = "/usr/local/cuda"):
        self.accelsim_root = os.path.abspath(accelsim_root)
        self.cuda_path = cuda_path

    @property
    def sim_bin(self) -> str:
        return os.path.join(self.accelsim_root, "gpu-simulator/bin/release/accel-sim.out")

    @property
    def tracer_so(self) -> str:
        return os.path.join(
            self.accelsim_root, "util/tracer_nvbit/tracer_tool/tracer_tool.so"
        )

    @property
    def post_processor(self) -> str:
        return os.path.join(
            self.accelsim_root,
            "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing",
        )

    @property
    def nvcc(self) -> str:
        return os.path.join(self.cuda_path, "bin/nvcc")

    def config_pair(self, arch: str) -> tuple[str, str]:
        """Return ``(gpgpusim.config, trace.config)`` for e.g. ``SM80_A100``.

        The two live in different repos — the microarchitecture model ships with
        GPGPU-Sim, the SASS opcode latencies with Accel-Sim — and both are
        required.
        """
        return (
            os.path.join(
                self.accelsim_root,
                "gpu-simulator/gpgpu-sim/configs/tested-cfgs",
                arch,
                "gpgpusim.config",
            ),
            os.path.join(
                self.accelsim_root, "gpu-simulator/configs/tested-cfgs", arch, "trace.config"
            ),
        )

    # ------------------------------------------------------------------
    # Phase 0 — build (CPU only; nvcc needs no GPU)
    # ------------------------------------------------------------------
    @staticmethod
    def build_cuda(
        node: "AccelSimNode",
        sources: list[str],
        binary: str,
        *,
        arch_sm: str = "sm_80",
        extra_flags: tuple[str, ...] = ("-O3", "-lineinfo"),
        cwd: str | None = None,
        timeout_s: int = 600,
    ) -> CudaBuildResult:
        """Compile CUDA sources to *binary*.

        A node rather than a driver-side ``subprocess.run`` because generated
        kernels fail to compile routinely, and a compile failure has to be a
        readable result the loop can feed back to the model, not an exception
        that kills the driver. ``stderr_tail`` is the nvcc diagnostic, which is
        exactly what the next prompt needs.

        ``-lineinfo`` is kept because the NVBit tracer records line numbers when
        it is present, which makes a trace attributable back to source.
        """
        binary = os.path.abspath(binary)
        os.makedirs(os.path.dirname(binary), exist_ok=True)
        cmd = [
            node.nvcc, f"-arch={arch_sm}", *extra_flags,
            *[os.path.abspath(s) for s in sources],
            "-o", binary,
        ]
        rc, _out, err, timed_out, wall = _run_logged(cmd, cwd=cwd, timeout_s=timeout_s)
        return CudaBuildResult(
            binary=binary,
            sources=list(sources),
            cmd=" ".join(cmd),
            returncode=rc,
            timed_out=timed_out,
            wall_s=wall,
            stderr_tail=_tail(err),
        )

    # ------------------------------------------------------------------
    # Phase 0.5 — verify on silicon (needs a GPU; seconds, not minutes)
    # ------------------------------------------------------------------
    @staticmethod
    def verify_kernel(
        node: "AccelSimNode",
        binary: str,
        dump: str,
        *,
        args: tuple[str, ...] = (),
        reference_dump: str | None = None,
        dtype: str = "float32",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        max_mismatch_frac: float = 0.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int = 600,
    ) -> VerifyResult:
        """Run *binary* on a real GPU and compare its output to a reference.

        **This is the gate on the whole software-side search.** An agent asked
        to reduce cycles can reduce them to zero by computing less, and every
        such "optimization" looks like a win in the simulator. Nothing may be
        traced or simulated until it has passed here.

        Calling convention: ``binary <args...> <dump>`` — the harness writes a
        raw little-endian ``dtype`` array to the last argument. Inputs are the
        harness's business, not the candidate's, so a rewrite cannot change the
        problem it is being measured on.

        The comparison is **tolerance-based, not bitwise, on purpose.** A tiled
        SGEMM reassociates the accumulation, so the correct answer legitimately
        differs in the low bits; a bitwise check would reject every real
        optimization. ``rtol=1e-4`` is roughly the fp32 slack for a length-256
        reassociated dot product — tighten it for shorter reductions, loosen it
        only with a reason.

        With ``reference_dump=None`` this *produces* the reference instead of
        checking against one.

        Output equality is necessary but not sufficient: pair it with
        :func:`work_preserved` on the traces, which catches a kernel that got
        the right answer without doing the arithmetic.
        """
        import numpy as np

        dump = os.path.abspath(dump)
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        if os.path.exists(dump):
            os.remove(dump)  # never let a stale dump pass as a fresh result

        rc, out, err, timed_out, wall = _run_logged(
            [os.path.abspath(binary), *args, dump], cwd=cwd, timeout_s=timeout_s, env=env
        )
        res = VerifyResult(
            dump=dump,
            returncode=rc,
            timed_out=timed_out,
            wall_s=wall,
            stdout_tail=_tail(out),
            stderr_tail=_tail(err),
        )
        if rc != 0 or timed_out:
            res.reason = "timeout" if timed_out else f"exit {rc}"
            return res
        if not os.path.exists(dump):
            res.reason = "no output dump written"
            return res

        cand = np.fromfile(dump, dtype=dtype)
        res.n_elems = int(cand.size)
        res.nan_or_inf = bool(not np.all(np.isfinite(cand)))
        if res.nan_or_inf:
            res.reason = "output contains NaN or Inf"
            return res
        if cand.size == 0:
            res.reason = "empty output dump"
            return res

        if reference_dump is None:
            res.ok = True
            res.reason = "reference produced"
            return res

        ref = np.fromfile(reference_dump, dtype=dtype)
        if ref.size != cand.size:
            res.reason = f"size mismatch: {cand.size} vs reference {ref.size}"
            return res

        adiff = np.abs(cand - ref)
        denom = np.maximum(np.abs(ref), np.finfo(np.dtype(dtype)).tiny)
        rdiff = adiff / denom
        bad = adiff > (atol + rtol * np.abs(ref))
        res.n_mismatch = int(bad.sum())
        res.max_abs_err = float(adiff.max())
        res.max_rel_err = float(rdiff.max())
        if res.n_mismatch > max_mismatch_frac * cand.size:
            res.reason = (
                f"{res.n_mismatch}/{cand.size} elements outside "
                f"rtol={rtol} atol={atol} (max_rel_err={res.max_rel_err:.3e})"
            )
            return res
        res.ok = True
        res.reason = "match"
        return res

    # ------------------------------------------------------------------
    # Phase 1 — tracing (needs a physical GPU)
    # ------------------------------------------------------------------
    @staticmethod
    def trace_app(
        node: "AccelSimNode",
        app_cmd: list[str],
        trace_dir: str,
        *,
        device: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        post_process_jobs: int = 8,
        max_trace_bytes: int = 64 * 1024 ** 3,
        max_est_sim_seconds: float = 0.0,
        count_opcodes: bool = True,
        keep_raw: bool = False,
        timeout_s: int = 3600,
    ) -> AccelSimTraceResult:
        """Trace *app_cmd* under NVBit, then post-process into ``.tracez``.

        Works on any CUDA binary — the tracer attaches via ``LD_PRELOAD`` and
        needs no recompilation or source access.

        **GPU selection is Ray's job, not this function's.** The node is
        declared ``num_gpus=1``, so Ray has already set ``CUDA_VISIBLE_DEVICES``
        to the one GPU it assigned this task; verified: four concurrent tasks
        receive ``['0','1','2','3']``. Overwriting it here — which this function
        used to do, unconditionally, with a default of ``0`` — silently funnels
        every concurrent trace onto physical GPU 0. Nothing errors, tracing just
        serializes and nobody notices. ``device`` is therefore ``None`` by
        default and should stay that way outside single-process debugging.

        Note the deliberate omission: this does **not** source
        ``gpu-simulator/setup_environment.sh``. That script puts GPGPU-Sim's
        stub ``libcudart`` ahead of the real one, and tracing under it segfaults.
        Build and trace environments must stay separate.

        ``max_trace_bytes`` / ``max_est_sim_seconds`` are the guard against a
        generated kernel whose loop bounds make simulation unaffordable. A 1024^2
        tiled SGEMM is ~5e9 thread-instructions, which at the measured 507k
        inst/s is 2.7 *hours* per configuration. Refusing here costs one trace;
        not refusing costs a whole sweep.
        """
        trace_dir = os.path.abspath(trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        # Clear any previous trace before writing a new one. The tracer names
        # its files after the host process's context pointer, which is
        # different every run, and the post-processor rebuilds from *every*
        # ``kernelslist_ctx_*`` it finds in the directory. So tracing twice
        # into one directory makes the post-processor re-derive the earlier
        # run's ``.tracez`` from raw ``.trace`` files that the cleanup at the
        # end of this function already deleted -- it writes a 21-byte stub --
        # and then points the canonical ``kernelslist.g`` at that stub rather
        # than at the trace just recorded. GPGPU-Sim rejects the result with
        # "unsupported binary version: 0", and the good trace is gone too.
        # Found when a forge self-test re-run silently destroyed the trace
        # that forge_band.py reads. A stale trace is never the wanted one.
        shutil.rmtree(os.path.join(trace_dir, "traces"), ignore_errors=True)
        for stale in (glob.glob(os.path.join(trace_dir, "kernelslist*"))
                      + glob.glob(os.path.join(trace_dir, "*.trace*"))):
            try:
                os.remove(stale)
            except OSError:
                pass

        trace_env = {
            "NVBIT_INSTRUMENTATION_ENABLED": "1",
            "TRACES_FOLDER": trace_dir,
            "LD_PRELOAD": node.tracer_so,
            "CUDA_INSTALL_PATH": node.cuda_path,
            **(env or {}),
        }
        if device is not None:
            trace_env["CUDA_VISIBLE_DEVICES"] = str(device)
        gpu_id = trace_env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

        rc, _stdout, stderr, timed_out, wall = _run_logged(
            app_cmd, cwd=cwd, timeout_s=timeout_s, env=trace_env
        )

        # The tracer writes into a "traces" subdirectory of TRACES_FOLDER.
        raw_dir = os.path.join(trace_dir, "traces")
        if not os.path.isdir(raw_dir):
            raw_dir = trace_dir

        # Opcode histogram must be taken from the raw trace, before
        # post-processing and before cleanup: the .tracez is a different format.
        opcodes: dict[str, int] = {}
        if rc == 0 and not timed_out and count_opcodes:
            try:
                opcodes = opcode_histogram(raw_dir)
                with open(os.path.join(raw_dir, "opcodes.json"), "w") as fh:
                    json.dump(opcodes, fh, indent=1, sort_keys=True)
            except Exception as exc:  # never let bookkeeping kill a good trace
                stderr += f"\n[opcode_histogram failed: {exc}]"

        if rc == 0 and not timed_out:
            pp_rc, _pp_out, pp_err, pp_timeout, pp_wall = _run_logged(
                [node.post_processor, raw_dir, "-j", str(post_process_jobs)],
                cwd=cwd,
                timeout_s=timeout_s,
            )
            wall += pp_wall
            if pp_rc != 0 or pp_timeout:
                rc, stderr, timed_out = pp_rc, pp_err, pp_timeout

        if rc == 0 and not timed_out and not keep_raw:
            # The raw .trace.xz is ~3x the .tracez the simulator actually reads
            # (18 MB vs 6.7 MB for one small kernel) and nothing downstream
            # needs it once the histogram is taken.
            for p in glob.glob(os.path.join(raw_dir, "*.trace*")):
                if not p.endswith(".tracez"):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        kernelslist = os.path.join(raw_dir, "kernelslist.g")
        n_kernels = 0
        if os.path.exists(kernelslist):
            with open(kernelslist) as fh:
                n_kernels = sum(1 for ln in fh if ln.strip().endswith(".tracez"))

        trace_bytes = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _dn, fn in os.walk(raw_dir)
            for f in fn
        )
        # Warp instructions x 32 lanes, an upper bound on thread instructions.
        est_thread_insts = sum(opcodes.values()) * 32
        est_sim_seconds = est_thread_insts / 507_000.0

        aborted = ""
        if trace_bytes > max_trace_bytes:
            aborted = f"trace {trace_bytes/1e9:.1f} GB exceeds cap {max_trace_bytes/1e9:.1f} GB"
        elif max_est_sim_seconds and est_sim_seconds > max_est_sim_seconds:
            aborted = (
                f"estimated {est_sim_seconds/60:.0f} min/config exceeds cap "
                f"{max_est_sim_seconds/60:.0f} min"
            )
        if aborted:
            shutil.rmtree(raw_dir, ignore_errors=True)

        return AccelSimTraceResult(
            kernelslist=kernelslist,
            trace_dir=raw_dir,
            app_cmd=" ".join(app_cmd),
            n_kernels=n_kernels,
            trace_bytes=trace_bytes,
            gpu_id=gpu_id,
            opcodes=opcodes,
            est_thread_insts=est_thread_insts,
            est_sim_seconds=est_sim_seconds,
            returncode=rc,
            timed_out=timed_out,
            aborted_reason=aborted,
            wall_s=wall,
            stderr_tail=_tail(stderr),
        )

    # ------------------------------------------------------------------
    # Phase 2 — simulation (CPU only; this is what gets swept in parallel)
    # ------------------------------------------------------------------
    @staticmethod
    def run_accelsim(
        node: "AccelSimNode",
        kernelslist: str,
        arch: str = "SM80_A100",
        *,
        outdir: str | None = None,
        knob_overrides: dict[str, str] | None = None,
        trace_knob_overrides: dict[str, str] | None = None,
        stats_patterns: dict[str, str] | None = None,
        collect_counters: bool = True,
        per_kernel: bool = True,
        reuse_existing: bool = True,
        timeout_s: int = 7200,
    ) -> AccelSimRunResult:
        """Replay a trace under one configuration and parse the counters.

        ``knob_overrides`` edits ``gpgpusim.config``, ``trace_knob_overrides``
        edits ``trace.config``; both are written to a fresh directory so the
        blessed configs are never mutated and every run stays reproducible from
        its own ``outdir``.

        The returned ``knobs`` and ``trace_knobs`` are the *effective* configs
        read back off disk, so a caller can diff intent against what the
        simulator was actually given. That check is not cosmetic: several knobs
        are silently inert — with ``-gpgpu_adaptive_cache_config 1`` the L1D
        geometry is recomputed at every kernel launch and overrides hand-edited
        ``-gpgpu_cache:dl1`` fields, and in trace-driven mode HMMA maps to a
        specialized unit rather than the modelled tensor cores, so
        ``-gpgpu_tensor_core_avail`` does nothing. Comparing the effective knobs
        catches the first class of mistake; only a cycle-count delta catches the
        second.

        ``reuse_existing`` dedupes on :func:`config_fingerprint`. Simulation is
        deterministic given a trace, so a repeat is guaranteed identical, and
        LLM-driven loops re-propose configurations constantly. At ~3 minutes a
        run this is worth more than most of the knobs being swept.
        """
        base_gpgpusim, base_trace = node.config_pair(arch)
        outdir = os.path.abspath(outdir or os.path.join(os.getcwd(), f"asim_{arch}"))
        os.makedirs(outdir, exist_ok=True)

        fp = config_fingerprint(kernelslist, arch, knob_overrides, trace_knob_overrides)
        fp_path = os.path.join(outdir, "fingerprint.txt")
        log_path = os.path.join(outdir, "sim.log")

        gpgpusim_cfg = os.path.join(outdir, "gpgpusim.config")
        trace_cfg = os.path.join(outdir, "trace.config")

        cached = False
        stdout = ""
        rc, stderr, timed_out, wall = 0, "", False, 0.0
        if (
            reuse_existing
            and os.path.exists(fp_path)
            and os.path.exists(log_path)
            and open(fp_path).read().strip() == fp
            and not _trace_newer_than(kernelslist, log_path)
        ):
            stdout = open(log_path, errors="ignore").read()
            cached = True
        else:
            write_config_knobs(base_gpgpusim, gpgpusim_cfg, knob_overrides or {})
            write_config_knobs(base_trace, trace_cfg, trace_knob_overrides or {})
            cmd = [
                node.sim_bin,
                "-trace", kernelslist,
                "-config", gpgpusim_cfg,
                "-config", trace_cfg,
            ]
            rc, stdout, stderr, timed_out, wall = _run_logged(
                cmd, cwd=outdir, timeout_s=timeout_s
            )
            with open(log_path, "w") as fh:
                fh.write(stdout)
            if stderr.strip():
                # The simulator asserts on illegal configs and prints the
                # assertion to stderr; stdout just stops mid-config-dump. Losing
                # stderr turns "L1 size must divide unified_l1d_size" into an
                # unexplained failure.
                with open(os.path.join(outdir, "sim.err"), "w") as fh:
                    fh.write(stderr)
            if rc == 0 and not timed_out:
                with open(fp_path, "w") as fh:
                    fh.write(fp)

        return AccelSimRunResult(
            config_dir=outdir,
            kernelslist=kernelslist,
            stats=parse_accelsim_stats(stdout, stats_patterns),
            knobs=read_config_knobs(gpgpusim_cfg),
            trace_knobs=read_config_knobs(trace_cfg),
            per_kernel=parse_per_kernel_stats(stdout, stats_patterns) if per_kernel else [],
            counters=read_perf_counters(outdir) if collect_counters else {},
            returncode=rc,
            timed_out=timed_out,
            wall_s=wall,
            cached=cached,
            stdout_tail=_tail(stdout),
            stderr_tail=_stderr_digest(stderr),
        )


if ChiaFunction is not None:
    # Decorate after the class body so the module still imports (with plain,
    # locally-callable functions) on a machine without ray. Class attribute
    # access unwraps the staticmethod to the underlying function, which is what
    # ChiaFunction wants; in 3.10 that unwrapped object has no ``__func__``.
    #
    # Placement is the point of the split: only the two GPU phases contend for
    # silicon, and they are the cheap ones.
    AccelSimNode.build_cuda = staticmethod(  # type: ignore[method-assign]
        ChiaFunction(resources={"accelsim": 1.0})(AccelSimNode.build_cuda)
    )
    AccelSimNode.verify_kernel = staticmethod(  # type: ignore[method-assign]
        ChiaFunction(resources={"accelsim_gpu": 1.0}, num_gpus=1)(AccelSimNode.verify_kernel)
    )
    AccelSimNode.trace_app = staticmethod(  # type: ignore[method-assign]
        ChiaFunction(resources={"accelsim_gpu": 1.0}, num_gpus=1)(AccelSimNode.trace_app)
    )
    AccelSimNode.run_accelsim = staticmethod(  # type: ignore[method-assign]
        ChiaFunction(resources={"accelsim": 1.0})(AccelSimNode.run_accelsim)
    )


# ======================================================================
# Fan-out
# ======================================================================

def sweep_configs(
    node: AccelSimNode,
    kernelslist: str,
    candidates: dict[str, dict[str, str]] | list[dict[str, str]],
    arch: str = "SM80_A100",
    *,
    outroot: str | None = None,
    trace_candidates: dict[str, dict[str, str]] | None = None,
    gather: bool = True,
    **run_kwargs,
):
    """Dispatch one simulation per candidate config against a *single* trace.

    Deliberately a plain function, not a ``@ChiaFunction``: a node that blocked
    waiting on its own children would hold a worker slot idle for the whole
    sweep. Dispatch happens here, the work happens on the cluster.

    Pinning every candidate to one ``kernelslist`` is the point. Simulation is
    deterministic given a trace, but re-tracing the same source shifts the
    baseline by ~0.4% — larger than some real knob effects — so candidates
    compared across different traces are not comparable at all. Taking the trace
    as a single argument makes that mistake hard to express.

    With ``gather=True`` returns ``{name: AccelSimRunResult}``; with
    ``gather=False`` returns ``{name: ObjectRef}`` so the caller can overlap
    other work before collecting. The pipeline wants ``gather=False``: the
    next kernel's trace and baseline should start while this sweep is still
    running.
    """
    if isinstance(candidates, list):
        candidates = {f"cfg{i}": c for i, c in enumerate(candidates)}
    outroot = os.path.abspath(outroot or os.path.join(os.getcwd(), "accelsim_sweep"))
    trace_candidates = trace_candidates or {}

    refs = {
        name: AccelSimNode.run_accelsim.chia_remote(
            node,
            kernelslist=kernelslist,
            arch=arch,
            outdir=os.path.join(outroot, name),
            knob_overrides=knobs,
            trace_knob_overrides=trace_candidates.get(name),
            **run_kwargs,
        )
        for name, knobs in candidates.items()
    }
    if not gather:
        return refs

    from chia.base.ChiaFunction import get

    return {name: get(ref) for name, ref in refs.items()}


def compare_runs(
    results: dict[str, AccelSimRunResult],
    baseline: str = "baseline",
    metric: str = "cycles",
) -> dict[str, dict]:
    """Rank a sweep against its baseline and flag candidates that did nothing.

    ``inert`` means the metric came back *bit-identical* to the baseline — the
    knob was accepted by the parser and changed nothing observable. That is
    common enough to deserve first-class reporting rather than being read as a
    failed optimization: ``-gpgpu_adaptive_cache_config 1`` recomputes L1D
    geometry per launch and discards ``-gpgpu_cache:dl1`` edits, and in
    trace-driven mode HMMA dispatches to a specialized unit, so tensor-core
    knobs are no-ops. An agent that cannot tell "inert" from "tried and didn't
    help" will keep re-proposing the same dead knobs.
    """
    base = results.get(baseline)
    base_val = base.stats.get(metric) if base and base.ok else None

    out: dict[str, dict] = {}
    for name, r in results.items():
        val = r.stats.get(metric) if r.ok else None
        entry: dict = {"value": val, "ok": r.ok, "wall_s": r.wall_s, "cached": r.cached}
        if r.stall_mix:
            entry["stall_mix"] = {k: round(v, 3) for k, v in r.stall_mix.items()}
        if val is not None and base_val:
            entry["delta_pct"] = 100.0 * (val - base_val) / base_val
            entry["inert"] = name != baseline and val == base_val
        out[name] = entry
    return out
