"""chia.simulators.accelsim_power — AccelWattch power, energy, and EDP.

Three things this module exists to do, in order of how much trouble they save:

1. **Co-generate the AccelWattch XML with ``gpgpusim.config``.** This is the
   important one. ``gpgpu_sim_wrapper::init_mcpat`` passes exactly four values
   out of the simulator config into McPAT::

       p->sys.total_cycles          = gpu_stat_sample_freq;
       p->sys.target_core_clockrate = clock_freq;
       p->sys.number_of_cores       = num_shaders;
       p->sys.core[0].clock_rate    = clock_freq;

   Everything else structural -- cache capacity, associativity, MSHR counts,
   shared-memory size, register-file size -- is hardcoded in the XML. So a
   design point that doubles L2 in ``gpgpusim.config`` gets a bigger cache in
   the *performance* model and the original cache in the *power* model: it
   collects the hit-rate win without ever paying the access-energy cost. EDP
   then has a degenerate optimum ("grow every array") in exactly the way a
   cycles-only objective does. :func:`render_power_xml` closes that hole by
   deriving the XML's geometry from the same knobs.

   Only SM count and clock are safe without it. Those two reach McPAT directly;
   everything in :data:`XML_DERIVED_KNOBS` does not.

2. **Parse the power report and refuse to trust it blindly.** Several
   components read exactly ``0``. Some of those are physically right for a
   given kernel (no FP64 in a TF32 GEMM) and some are the power-model version
   of the inert-knob problem. :func:`classify_components` separates them the
   only way that works -- by cross-checking each component against an
   *independent* activity counter from the performance model, never against the
   value itself. Measured on CUTLASS 1024^3: ``DRAM = 0 W`` is legitimate
   because the performance model independently reports ``total dram reads = 0``
   (a 12 MB working set that never leaves L2), while ``IC = 0 W`` is not, since
   146 M instructions were executed and every one had to be fetched.

3. **Compute energy and EDP.** ``EDP = P * t^2``, with ``t = cycles / clock``.
   Power alone is degenerate as an objective -- it is a rate, so the cheapest
   way to lower it is to do less per second, and an optimizer handed raw watts
   will remove SMs until the kernel crawls. Note also that for a *fixed* amount
   of work, maximizing perf-per-watt and minimizing energy are the same
   objective: ``FLOPs/t / P = FLOPs / (P*t) = FLOPs / E``.

Validity caveat that belongs next to every number this module returns: no
AccelWattch XML ships for ``SM80_A100``. :data:`DEFAULT_XML_ARCH` is Volta, so
absolute watts are **not** A100-validated. Deltas between design points are
comparable because every point is rendered from the same base XML.
"""

from __future__ import annotations

import math
import shutil
import os
import re
from dataclasses import dataclass, field

# ======================================================================
# Config <-> XML bridge
# ======================================================================

DEFAULT_XML_ARCH = "SM7_GV100"
"""Which tested-cfg to borrow ``accelwattch_sass_sim.xml`` from.

A100 ships none. Volta is the closest relative with a validated model: same
datacenter line, HBM2, tensor cores. The performance config stays SM80_A100 --
only the circuit parameters are borrowed, and the ISA comes from the trace's
``binary_verion`` field rather than from either config, so there is no opcode
mismatch (``trace_driven.cc``, ``OpcodeMap`` selection).
"""

POWER_KNOBS = {
    "-power_simulation_enabled": "1",
    "-power_simulation_mode": "0",  # 0=Sim: driven by simulator counters alone
    "-accelwattch_xml_file": "accelwattch_sass_sim.xml",
    "-power_per_cycle_dump": "0",
    "-aggregate_power_stats": "1",
}
"""Knobs that turn the power model on. Added to every run's overrides so that
enabling power changes the config fingerprint and invalidates pre-power caches.
"""

XML_DERIVED_KNOBS = (
    "-gpgpu_cache:dl1",
    "-gpgpu_cache:dl2",
    "-gpgpu_unified_l1d_size",
    "-gpgpu_shmem_size",
    "-gpgpu_shader_registers",
)
"""Knobs whose effect the power model cannot see unless the XML is regenerated.

A design point touching any of these and *not* re-rendering the XML produces an
EDP that is biased in a known direction: cheaper than reality. :func:`edp`
takes a ``xml_synced`` flag so a caller cannot forget.
"""

_CACHE_SPEC_RE = re.compile(
    r"^\s*[A-Z]:(?P<nsets>\d+):(?P<line>\d+):(?P<assoc>\d+)\s*,"  # geometry
    r"[^,]*,"                                                     # policy field
    r"\s*[A-Z]:(?P<mshr>\d+):(?P<merge>\d+)"                      # MSHR field
)


def parse_cache_spec(spec: str) -> dict | None:
    """Decode a GPGPU-Sim cache string into geometry McPAT can consume.

    Format is ``<type>:<nsets>:<linesize>:<assoc>,<policy...>,<mshr_type>:
    <entries>:<merge>,<miss_queue>:<fifo>,<port_width>``. Only the fields the
    XML has a slot for are returned.
    """
    m = _CACHE_SPEC_RE.match(spec or "")
    if not m:
        return None
    nsets = int(m.group("nsets"))
    line = int(m.group("line"))
    assoc = int(m.group("assoc"))
    return {
        "capacity": nsets * line * assoc,
        "line": line,
        "assoc": assoc,
        "mshr": int(m.group("mshr")),
        "merge": int(m.group("merge")),
    }


def core_clock_hz(knobs: dict[str, str], default_mhz: float = 1410.0) -> float:
    """Core clock in Hz from ``-gpgpu_clock_domains <core>:<icnt>:<l2>:<dram>``."""
    raw = (knobs or {}).get("-gpgpu_clock_domains", "")
    try:
        return float(str(raw).split(":")[0]) * 1e6
    except (ValueError, IndexError):
        return default_mhz * 1e6


def _sub_csv(value: str, updates: dict[int, int]) -> str:
    """Replace positional fields in a comma-separated XML param value."""
    parts = [p.strip() for p in value.split(",")]
    for idx, new in updates.items():
        if idx < len(parts):
            parts[idx] = str(new)
    return ",".join(parts)


def render_power_xml(src_xml: str, dst_xml: str, knobs: dict[str, str]) -> dict:
    """Write *dst_xml* from *src_xml* with geometry taken from *knobs*.

    Walks the XML linewise tracking which ``<component>`` it is inside, because
    ``buffer_sizes`` (the MSHR count) appears once per cache and is only
    identifiable by the ``*_config`` param immediately preceding it.

    **Associativity is deliberately not propagated.** McPAT's CACTI backend
    builds a real SRAM array and rejects geometries GPGPU-Sim is happy to
    simulate: A100's L1 is ``S:4:128:64``, i.e. 4 sets of 64 ways, and handing
    that to CACTI aborts the run with ``ERROR: no valid data array
    organizations found`` before a single cycle executes. Bisected field by
    field -- capacity alone passes, capacity+line passes, adding associativity
    fails -- so capacity and line width are propagated and associativity is
    left at the validated XML's value.

    The consequence is worth stating rather than burying: an
    associativity-only sweep is invisible to the power model, so such a point
    would show a cycles change at unchanged energy. Every dimension actually
    swept here varies capacity, so the bias does not arise in practice -- but a
    future dimension that moves only associativity must not be trusted on EDP.

    Returns what was actually changed, so a caller can assert the XML moved
    when it expected it to -- the same readback discipline the knob guards use.
    """
    dl1 = parse_cache_spec(knobs.get("-gpgpu_cache:dl1", ""))
    dl2 = parse_cache_spec(knobs.get("-gpgpu_cache:dl2", ""))
    # The three read-mostly caches McPAT prices but the shipped XML sizes
    # independently of the simulator config: at SM80_A100 the XML declares a
    # 48 KB texture cache, a 16 KB constant cache and a 16 KB instruction cache
    # against a config that asks for 128/64/128 KB. That mismatch is harmless
    # for a fixed machine and fatal for a *reallocation*, where the whole point
    # is to shrink one of them and be charged correctly for it.
    tex = parse_cache_spec(knobs.get("-gpgpu_tex_cache:l1", ""))
    con = parse_cache_spec(knobs.get("-gpgpu_const_cache:l1", ""))
    il1 = parse_cache_spec(knobs.get("-gpgpu_cache:il1", ""))
    applied: dict[str, str] = {}

    try:
        shmem = int(float(knobs.get("-gpgpu_shmem_size", 0)))
    except (TypeError, ValueError):
        shmem = 0
    try:
        regs = int(float(knobs.get("-gpgpu_shader_registers", 0)))
    except (TypeError, ValueError):
        regs = 0

    def _units(knob: str, per_unit: int) -> int:
        """Functional-unit count -> McPAT lane count.

        McPAT counts *lanes* (``FPU_per_core 32``) where GPGPU-Sim counts
        *pipelines* (``-gpgpu_num_sp_units 4``). The shipped pair fixes the
        ratio at 8 lanes per pipeline, and that ratio is all this needs to be:
        the area model only has to move in proportion when a probe removes a
        pipeline. Returns 0 when the knob is absent so the XML value stands.
        """
        try:
            n = int(float(knobs.get(knob, 0)))
        except (TypeError, ValueError):
            return 0
        return n * per_unit

    out: list[str] = []
    current = ""  # which cache's buffer_sizes we are about to see
    with open(src_xml) as fh:
        for line in fh:
            m = re.search(r'name="(\w+)"\s+value="([^"]*)"', line)
            if not m:
                out.append(line)
                continue
            name, value = m.group(1), m.group(2)
            new = None

            if name == "tcache_config" and tex:
                new = _sub_csv(value, {0: tex["capacity"], 1: tex["line"]})
                current = "tcache"
            elif name == "ccache_config" and con:
                new = _sub_csv(value, {0: con["capacity"], 1: con["line"]})
                current = "ccache"
            elif name == "icache_config" and il1:
                new = _sub_csv(value, {0: il1["capacity"], 1: il1["line"]})
                current = "icache"
            elif name == "FPU_per_core" and _units("-gpgpu_num_sp_units", 8):
                new = str(_units("-gpgpu_num_sp_units", 8))
            elif name == "ALU_per_core" and _units("-gpgpu_num_int_units", 8):
                new = str(_units("-gpgpu_num_int_units", 8))
            elif name == "MUL_per_core" and _units("-gpgpu_num_sfu_units", 1):
                # McPAT's "Complex ALUs (Mul/Div)"; GPGPU-Sim routes div and
                # transcendentals to the SFU, and the shipped values agree at 4.
                new = str(_units("-gpgpu_num_sfu_units", 1))
            elif name == "dcache_config" and dl1:
                new = _sub_csv(value, {0: dl1["capacity"], 1: dl1["line"]})
                current = "dcache"
            elif name == "L2_config" and dl2:
                new = _sub_csv(value, {0: dl2["capacity"], 1: dl2["line"]})
                current = "L2"
            elif name == "sharedmemory_config" and shmem:
                new = _sub_csv(value, {0: shmem})
                current = "sharedmemory"
            elif name == "archi_Regs_IRF_size" and regs:
                new = str(regs)
            elif name == "buffer_sizes":
                if current == "dcache" and dl1:
                    new = _sub_csv(value, {0: dl1["mshr"]})
                elif current == "L2" and dl2:
                    new = _sub_csv(value, {0: dl2["mshr"]})
                # `buffer_sizes` appears once per cache, so it must be keyed by
                # the component it belongs to or the readback silently reports
                # only the last one written.
                key = f"{current}.buffer_sizes" if current else "buffer_sizes"
                if new is not None and new != value:
                    applied[key] = new
                    line = line.replace(f'value="{value}"', f'value="{new}"')
                out.append(line)
                current = ""
                continue
            elif name.endswith("_config"):
                current = ""  # some other cache; do not attribute its buffers

            if new is not None and new != value:
                applied[name] = new
                line = line.replace(f'value="{value}"', f'value="{new}"')
            out.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(dst_xml)), exist_ok=True)
    with open(dst_xml, "w") as fh:
        fh.writelines(out)
    return applied


# ======================================================================
# Power report
# ======================================================================

POWER_REPORT = "accelwattch_power_report.log"

_AVG_POWER_RE = re.compile(r"kernel_avg_power\s*=\s*([-\d.eE+]+)")
_COMP_RE = re.compile(r"gpu_avg_(\w+?)P,?\s*=\s*([-\d.eE+]+)\s*$", re.M)
_ACT_RE = re.compile(r"gpu_avg_(\w+?),\s*=\s*([-\d.eE+]+)\s*$", re.M)


def parse_power_report(path_or_dir: str) -> dict:
    """Read ``accelwattch_power_report.log`` into components and activities.

    Returns ``{}`` when the file is absent or empty rather than raising: a run
    with power disabled is a normal thing for a caller to hold.
    """
    path = path_or_dir
    if os.path.isdir(path):
        path = os.path.join(path, POWER_REPORT)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    text = open(path, errors="ignore").read()

    m = _AVG_POWER_RE.search(text)
    if not m:
        return {}
    total = float(m.group(1))

    components = {k: float(v) for k, v in _COMP_RE.findall(text)}
    activities = {
        k: float(v)
        for k, v in _ACT_RE.findall(text)
        if not k.endswith("P") and k not in components
    }
    return {
        "avg_power_w": total,
        "components": components,
        "activities": activities,
        "component_sum": sum(components.values()),
        "residual": total - sum(components.values()),
        "path": path,
    }


# Each entry says how to decide whether a zero is a finding or a broken model.
#
#   "always"   -- must be nonzero in any real kernel; zero means inert
#   "workload" -- zero is legitimate when this kernel does not use the unit
#   (name, counter) -- zero is legitimate only if the named *performance*
#                      counter is also zero; cross-model agreement is the test
POWER_EXPECTATIONS: dict[str, object] = {
    "CONST": "always",
    "STATIC": "always",
    "IDLE_CORE": "always",
    "RF": "always",
    "SCHED": "always",
    "PIPE": "always",
    "IB": "always",
    "INT": "always",
    "IC": "always",  # 146M instructions were fetched; zero is not physical
    "NOC": ("icnt", "l2_total_accesses"),
    "L2C": ("l2", "l2_total_accesses"),
    "MC": ("memctrl", "l2_total_accesses"),
    "DRAM": ("dram", "dram_traffic"),
    "DC": ("l1d", "l1_global_reads"),
    "SHRD": ("shmem", "shmem_insns"),
    "TENSOR": ("tensor", "tensor_insts"),
    "FPU": "workload",
    "FP_MUL": "workload",
    "INT_MUL": "workload",
    "CC": "workload",
    "TC": "workload",
    "TEX": "workload",
    "DPU": "workload",
    "DP_MUL": "workload",
    "DP_DIV": "workload",
    "INT_DIV": "workload",
    "INT_MUL24": "workload",
    "INT_MUL32": "workload",
    "FP_DIV": "workload",
    "FP_SQRT": "workload",
    "FP_LG": "workload",
    "FP_SIN": "workload",
    "FP_EX": "workload",
}


def _activity_counters(stats: dict, counters: dict | None = None) -> dict[str, float]:
    """Independent activity evidence, taken from the *performance* model.

    The whole point is that these come from a different accounting path than
    the power model's own counters, so agreement between them is real evidence
    and not a tautology.
    """
    stats = stats or {}
    final = (counters or {}).get("final", {}) if counters else {}

    def g(*names: str) -> float:
        for n in names:
            v = stats.get(n, final.get(n))
            if v:
                return float(v)
        return 0.0

    return {
        "dram_traffic": g("dram_reads") + g("dram_writes"),
        "l2_total_accesses": g("l2_total_accesses", "l2_bw_gbps"),
        "l1_global_reads": g("l1_global_reads", "l1_bank_conflicts"),
        "shmem_insns": g("shmem_insns", "shmem_bank_conflicts"),
        "tensor_insts": g("tensor_insts"),
        "insts": g("insts"),
    }


@dataclass
class PowerVerdict:
    """What the power numbers may and may not be used for."""

    avg_power_w: float
    components: dict[str, float] = field(default_factory=dict)
    consistent: bool = True
    residual: float = 0.0
    inert: list[str] = field(default_factory=list)
    legitimate_zero: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Components summing to the total is necessary, not sufficient.

        Inert components are still reported, because a missing term makes the
        total an *underestimate* -- which biases EDP in a knowable direction
        rather than an unknown one.
        """
        return self.consistent


def classify_components(
    power: dict, stats: dict, counters: dict | None = None, tol: float = 1e-3
) -> PowerVerdict:
    """Split zero-valued power components into findings and modeling gaps."""
    if not power:
        return PowerVerdict(0.0, consistent=False, notes=["no power report"])

    comps = power.get("components", {})
    act = _activity_counters(stats, counters)
    residual = power.get("residual", 0.0)
    verdict = PowerVerdict(
        avg_power_w=power.get("avg_power_w", 0.0),
        components=comps,
        consistent=abs(residual) <= tol * max(1.0, power.get("avg_power_w", 1.0)),
        residual=residual,
    )
    if not verdict.consistent:
        verdict.notes.append(
            f"components sum to {power.get('component_sum', 0):.5f} but total is "
            f"{power.get('avg_power_w', 0):.5f} (residual {residual:+.5f})"
        )

    for name, value in comps.items():
        if value != 0:
            continue
        rule = POWER_EXPECTATIONS.get(name)
        if rule == "always":
            verdict.inert.append(name)
        elif rule == "workload":
            verdict.legitimate_zero.append(name)
        elif isinstance(rule, tuple):
            _, counter = rule
            evidence = act.get(counter, 0.0)
            if evidence > 0:
                verdict.inert.append(name)
                verdict.notes.append(
                    f"{name} = 0 W but {counter} = {evidence:g}: the power model "
                    "is not charging for activity the performance model recorded"
                )
            else:
                verdict.legitimate_zero.append(name)
                verdict.notes.append(
                    f"{name} = 0 W agrees with {counter} = 0 (not inert)"
                )
        else:
            verdict.unverified.append(name)
    return verdict


# ======================================================================
# Objective
# ======================================================================

@dataclass
class EnergyScore:
    """Energy, delay, and their product for one simulated design point."""

    cycles: float
    avg_power_w: float
    clock_hz: float
    seconds: float
    joules: float
    edp: float
    xml_synced: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def edp(
    cycles: float,
    avg_power_w: float,
    knobs: dict[str, str] | None = None,
    *,
    clock_hz: float | None = None,
    xml_synced: bool = True,
    baseline_knobs: dict[str, str] | None = None,
) -> EnergyScore:
    """``EDP = P * t^2`` where ``t = cycles / core_clock``.

    ``xml_synced=False`` is not a hint, it is a correctness flag: if any knob in
    :data:`XML_DERIVED_KNOBS` differs from the baseline and the XML was not
    re-rendered, the returned EDP is biased low by exactly the access energy of
    the structure that was grown. The warning travels with the score rather
    than being left for a reader to remember.
    """
    f = clock_hz if clock_hz else core_clock_hz(knobs or {})
    t = cycles / f if f else float("nan")
    e = avg_power_w * t
    score = EnergyScore(
        cycles=cycles,
        avg_power_w=avg_power_w,
        clock_hz=f,
        seconds=t,
        joules=e,
        edp=e * t,
        xml_synced=xml_synced,
    )
    if not xml_synced and knobs and baseline_knobs:
        drifted = [
            k for k in XML_DERIVED_KNOBS
            if str(knobs.get(k, "")) != str(baseline_knobs.get(k, ""))
        ]
        if drifted:
            score.warnings.append(
                "EDP biased LOW: " + ", ".join(drifted) + " changed but the "
                "AccelWattch XML was not re-rendered, so the power model still "
                "charges baseline access energy for a resized structure"
            )
    return score


def perf_per_watt(flops: float, score: EnergyScore) -> float:
    """FLOP/s per watt. Identical ordering to ``1/joules`` for fixed work."""
    return flops / score.joules if score.joules else float("nan")


# ======================================================================
# Runner
# ======================================================================

def run_with_power(
    node,
    kernelslist: str,
    arch: str = "SM80_A100",
    *,
    outdir: str,
    knob_overrides: dict[str, str] | None = None,
    xml_arch: str = DEFAULT_XML_ARCH,
    sync_xml: bool = True,
    **run_kwargs,
):
    """``run_accelsim`` with the power model on and the XML kept in step.

    The XML is written into *outdir* before the simulation, because
    ``-accelwattch_xml_file`` resolves relative to the simulator's working
    directory and ``run_accelsim`` runs with ``cwd=outdir``.

    Returns ``(AccelSimRunResult, PowerVerdict, EnergyScore)``.
    """
    from chia.simulators.accelsim import AccelSimNode, read_config_knobs

    # Caller overrides win over POWER_KNOBS. A probe like ``ideal_l1`` is not a
    # buildable machine, so its watts are fiction and it should be able to pass
    # -power_simulation_enabled 0 and skip the ~3x McPAT cost. Updating in the
    # other order silently forced power on for every caller.
    knobs = {**POWER_KNOBS, **(knob_overrides or {})}

    base_cfg, _ = node.config_pair(arch)
    # Effective knobs = base config with this point's overrides applied, which
    # is what the XML must mirror -- not the overrides alone, since most
    # geometry comes from the base config untouched.
    effective = read_config_knobs(base_cfg)
    effective.update(knobs)

    os.makedirs(outdir, exist_ok=True)
    src_xml = os.path.join(
        node.accelsim_root,
        "gpu-simulator/gpgpu-sim/configs/tested-cfgs",
        xml_arch,
        "accelwattch_sass_sim.xml",
    )
    dst_xml = os.path.join(outdir, POWER_KNOBS["-accelwattch_xml_file"])
    if sync_xml:
        applied = render_power_xml(src_xml, dst_xml, effective)
    else:
        # Still place the XML -- the simulator aborts without one. Unrendered
        # is the *point* of sync_xml=False: baseline circuit geometry driving a
        # resized machine, which is the bias `edp(xml_synced=False)` warns about.
        shutil.copyfile(src_xml, dst_xml)
        applied = {}

    result = AccelSimNode.run_accelsim(
        node,
        kernelslist=kernelslist,
        arch=arch,
        outdir=outdir,
        knob_overrides=knobs,
        **run_kwargs,
    )
    power = parse_power_report(outdir)
    verdict = classify_components(power, result.stats, result.counters)
    score = edp(
        result.stats.get("cycles", float("nan")),
        power.get("avg_power_w", float("nan")),
        result.knobs,
        xml_synced=sync_xml,
    )
    if applied:
        verdict.notes.append(
            "XML re-rendered: " + ", ".join(f"{k}={v}" for k, v in applied.items())
        )
    return result, verdict, score


__all__ = [
    "DEFAULT_XML_ARCH",
    "POWER_KNOBS",
    "XML_DERIVED_KNOBS",
    "POWER_EXPECTATIONS",
    "POWER_REPORT",
    "EnergyScore",
    "PowerVerdict",
    "classify_components",
    "core_clock_hz",
    "edp",
    "parse_cache_spec",
    "parse_power_report",
    "perf_per_watt",
    "render_power_xml",
    "run_with_power",
]
