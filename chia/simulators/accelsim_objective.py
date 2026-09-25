"""chia.simulators.accelsim_objective — cost functions for the co-design loop.

**This is the pinned decision, and it is the one that decides whether the loop
produces a result or an artifact.** Everything else in this package is
plumbing; the objective is the experiment.

The failure mode is concrete, not hypothetical. Measured on this host, same
trace, same simulator:

===================  ============  =================
config               cycles        cycles x SM count
===================  ============  =================
108 SMs (A100)             27,991          3,023,028
27 SMs                     88,502      **2,389,554**
===================  ============  =================

Minimizing cycles says 108 SMs. Minimizing area-delay says 27 SMs is 21%
*better*. Identical data, opposite conclusions. An unconstrained
"fewer cycles" objective has a trivial degenerate optimum -- make every
structure bigger -- and an agent will find it immediately and report a large,
meaningless win.

Two further constraints on the choice, both established by measurement rather
than argument:

* **IPC is invalid once software changes.** Redundant arithmetic raises IPC
  while slowing the kernel. It is a usable diagnostic for a fixed kernel and a
  broken objective for steps 4-6. Use cycles.
* **perf/watt is not available on A100 here.** AccelWattch ships with this tree
  but only SM6_TITANX / SM7_GV100 / SM7_QV100 / SM7_TITANV carry a validated
  ``accelwattch_*.xml``; ``SM80_A100/`` contains ``gpgpusim.config`` and nothing
  else. An energy objective therefore means either moving the whole study to
  SM7_QV100 (validated power, older uarch) or porting a Volta XML to A100
  (a modeling exercise with no validation, i.e. a fabricated number). See
  :func:`energy_delay`.

``recommended()`` states a recommendation and why; the loop takes whichever
objective it is handed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- A100 reference point, used only to normalize the area proxy ------------
A100_REFERENCE = {
    "-gpgpu_n_clusters": 108,
    "-gpgpu_n_cores_per_cluster": 1,
    "-gpgpu_n_mem": 40,
    "-gpgpu_shader_registers": 65536,
    "-gpgpu_shmem_size": 167936,
}

_CACHE_GEOM_RE = re.compile(r"^[A-Z]:(\d+):(\d+):(\d+)")


def _cache_bytes(spec: str) -> int:
    """Bytes in a GPGPU-Sim cache spec ``<type>:<nsets>:<linesize>:<assoc>,...``."""
    m = _CACHE_GEOM_RE.match((spec or "").strip())
    if not m:
        return 0
    nsets, line, assoc = (int(g) for g in m.groups())
    return nsets * line * assoc


def area_proxy(knobs: dict[str, str], *, weights: dict[str, float] | None = None) -> float:
    """A first-order area estimate, normalized so stock A100 == 1.0.

    **This is a proxy, not a model.** It is a weighted sum of the structures a
    config knob can actually resize -- SM count, per-SM register file, shared
    memory, L1, L2, memory controllers -- with weights chosen so they sum to 1.0
    on stock A100. It captures the thing that matters (you cannot make every
    structure bigger for free) and nothing else: no wire area, no process
    scaling, no floorplanning. Report it as "area proxy", never as mm^2.

    If the submission needs a defensible area number, the honest alternatives
    are a fixed-budget objective (:func:`fixed_budget`), which needs no area
    model at all, or CACTI for the SRAM structures.
    """
    w = {
        "sm": 0.55,      # SM logic + register file, the bulk of a GPU die
        "shmem": 0.10,   # shared memory / L1 SRAM, sized per SM
        "l1": 0.05,
        "l2": 0.15,
        "mc": 0.15,      # memory controllers + PHY
        **(weights or {}),
    }

    def num(k: str, default: float) -> float:
        try:
            return float(str(knobs.get(k, default)).split(":")[0])
        except (TypeError, ValueError):
            return float(default)

    n_sm = num("-gpgpu_n_clusters", 108) * num("-gpgpu_n_cores_per_cluster", 1)
    n_mem = num("-gpgpu_n_mem", 40)
    regs = num("-gpgpu_shader_registers", 65536)
    shmem = num("-gpgpu_shmem_size", 167936)
    l1 = _cache_bytes(knobs.get("-gpgpu_cache:dl1", "")) or 32768
    l2_bank = _cache_bytes(knobs.get("-gpgpu_cache:dl2", "")) or 262144
    n_l2_banks = n_mem * num("-gpgpu_n_sub_partition_per_mchannel", 4)

    ref_sm = A100_REFERENCE["-gpgpu_n_clusters"]
    sm_scale = (n_sm / ref_sm) * (regs / A100_REFERENCE["-gpgpu_shader_registers"])
    return (
        w["sm"] * sm_scale
        + w["shmem"] * (n_sm / ref_sm) * (shmem / A100_REFERENCE["-gpgpu_shmem_size"])
        + w["l1"] * (n_sm / ref_sm) * (l1 / 32768.0)
        + w["l2"] * (n_l2_banks * l2_bank) / (160 * 262144.0)
        + w["mc"] * (n_mem / 40.0)
    )


# ======================================================================
# Objectives. All are "lower is better".
# ======================================================================

@dataclass
class Score:
    value: float | None
    ok: bool = True
    reason: str = ""
    parts: dict = field(default_factory=dict)


def raw_cycles(result) -> Score:
    """Minimize cycles. **Degenerate on its own** -- included so a baseline arm
    can demonstrate the degeneracy rather than assert it."""
    if not result.ok:
        return Score(None, False, "run failed")
    return Score(result.stats["cycles"], parts={"cycles": result.stats["cycles"]})


def area_delay(result, *, weights: dict[str, float] | None = None) -> Score:
    """``cycles x area_proxy``. Penalizes buying speed with silicon.

    The natural default if the study is about microarchitecture. Its weakness is
    that it inherits every weakness of :func:`area_proxy`, so a result that
    hinges on a 5% area-delay difference is not a result.
    """
    if not result.ok:
        return Score(None, False, "run failed")
    a = area_proxy(result.knobs, weights=weights)
    c = result.stats["cycles"]
    return Score(c * a, parts={"cycles": c, "area_proxy": a})


def fixed_budget(
    result,
    *,
    budget: float = 1.0,
    weights: dict[str, float] | None = None,
    tolerance: float = 0.02,
) -> Score:
    """Minimize cycles **subject to** the area proxy staying within *budget*.

    The most defensible framing for a hackathon result, and the one to prefer
    unless there is a reason not to: it turns the search from "how much bigger
    can we make the chip" into "given an A100's worth of silicon, is this the
    right way to spend it". Reallocation results survive a weak area model in a
    way that absolute area-delay comparisons do not, because both sides of the
    comparison carry the same modeling error.

    A config over budget scores infinite rather than being silently penalized --
    the agent should see a rejection, not a bad number.
    """
    if not result.ok:
        return Score(None, False, "run failed")
    a = area_proxy(result.knobs, weights=weights)
    c = result.stats["cycles"]
    if a > budget * (1.0 + tolerance):
        return Score(
            float("inf"), False,
            f"area proxy {a:.3f} exceeds budget {budget:.3f}",
            parts={"cycles": c, "area_proxy": a},
        )
    return Score(c, parts={"cycles": c, "area_proxy": a})


def energy_delay(result) -> Score:
    """``energy x delay``. Requires AccelWattch, which A100 does not have here.

    Returns a failed score with the reason rather than a fabricated number.
    To make this work: run on ``SM7_QV100`` (which ships
    ``accelwattch_sass_sim.xml``) with ``-power_simulation_enabled 1``.
    """
    power = result.stats.get("avg_power_w") or result.counters.get("final", {}).get("avg_power_w")
    if not power:
        return Score(
            None, False,
            "no power data: -power_simulation_enabled requires an accelwattch "
            "XML, and SM80_A100 ships none. Use SM7_QV100 for an energy study.",
        )
    c = result.stats["cycles"]
    return Score(power * c * c, parts={"cycles": c, "avg_power_w": power})


def kernel_quality(result, *, reference_cycles: float) -> Score:
    """Score a *kernel* rewrite, holding hardware fixed at the baseline config.

    Steps 4-6 change software while steps 1-3 change hardware, and the two
    searches will happily launder each other's gains: a kernel evaluated on
    hardware that was tuned for it looks better than it is. Every candidate
    kernel must therefore also be scored here, on stock hardware, against the
    original kernel's cycles on that same stock hardware. The difference between
    this score and the tuned-hardware score *is* the co-design benefit, and it
    is the only number in the study that justifies the word "co-design".
    """
    if not result.ok:
        return Score(None, False, "run failed")
    c = result.stats["cycles"]
    return Score(c / reference_cycles, parts={"cycles": c, "speedup": reference_cycles / c})


OBJECTIVES = {
    "raw_cycles": raw_cycles,
    "area_delay": area_delay,
    "fixed_budget": fixed_budget,
    "energy_delay": energy_delay,
}

DEFAULT_OBJECTIVE = "fixed_budget"


def recommended() -> str:
    return (
        "PINNED -- not yet decided by the user.\n"
        "Ranked suggestions:\n"
        "  1. fixed_budget(budget=1.0)  -- minimize cycles at constant area proxy.\n"
        "     Best defensibility per unit of work: reallocation under a fixed\n"
        "     budget is robust to a weak area model, because the modeling error\n"
        "     appears on both sides of every comparison. Needs no power model.\n"
        "  2. area_delay                -- cycles x area proxy. Use if the story\n"
        "     is 'a smaller chip would have been better', which the 27-SM\n"
        "     measurement already supports. Inherits the area proxy's weakness.\n"
        "  3. energy_delay on SM7_QV100 -- the only validated-power option in\n"
        "     this tree. Strongest physical grounding, but it means abandoning\n"
        "     A100 for Volta and re-establishing every baseline.\n"
        "  4. raw_cycles                -- degenerate; keep it only as a control\n"
        "     arm that demonstrates the degeneracy.\n"
        "Whichever is chosen, score candidate kernels with kernel_quality() on\n"
        "STOCK hardware as well, or kernel gains and hardware gains cannot be\n"
        "told apart."
    )
