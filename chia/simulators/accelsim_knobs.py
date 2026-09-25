"""The simulator's knob surface, derived from its own option registrar.

The co-design loop used to carry a hand-written menu of things it could
change.  That menu was wrong in the way hand-written menus are always wrong:
it was short, nobody could say *how* short, and an agent told "coalescing
policy is not expressible" believed it and trimmed its diagnosis to fit.
The simulator registers every option it accepts in one place; this module
reads that place, so coverage is computed rather than remembered.

Two things it deliberately does not do.  It does not claim an option is safe
to sweep -- several abort, lie, or silently do nothing (see the recorded knob
traps), and only a measurement settles that.  And it does not expand every
composite string: ``-gpgpu_cache:dl1`` is fifteen positional fields behind one
option name, so the *field* surface is larger than the option count.  The
fields that are known to matter are named in :data:`COMPOSITE_FIELDS` so the
undercount is at least visible.
"""

from __future__ import annotations

import json
import os
import re

__all__ = ["KnobSpec", "scan_registrar", "load_catalog", "save_catalog",
           "classify", "coverage", "format_menu", "CATALOG_PATH",
           "COMPOSITE_FIELDS", "OPERATIONAL_RX", "FIXTURE"]

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "accelsim_knob_catalog.json")

#: ``option_parser_register(opp, "-name", OPT_TYPE, &var, "help", "default")``
_REG = re.compile(
    r'option_parser_register\s*\(\s*opp\s*,\s*"(-[^"]+)"\s*,\s*(\w+)\s*,'
    r'\s*&?([^,]+),\s*"((?:[^"\\]|\\.)*)"\s*,\s*"?((?:[^",\\]|\\.)*)"?\s*\)',
    re.S)

#: Options that change what is *reported* rather than what is *built*.
#: Sweeping any of these changes the output file, not the chip.
OPERATIONAL_RX = re.compile(
    r"visualizer|checkpoint|resume|_file$|xml|power_sim|accelwattch|aggregate_power"
    r"|debug|verbose|log$|cflog|memlatstat|runtime_stat|steady|liveness|dump"
    r"|print|profil|flush|save|watch|deadlock|ignore|random|max_insn|max_cta"
    r"|max_cycle|max_completed_cta|ptx|cuda_|opencl|SST|sst|trace_|kernel_launch"
    r"|occupancy_sm_number|compute_capability|experimental_lib|heap_size|stack_size"
    r"|runtime_sync_depth|runtime_pending_launch|trywait|keep$",
    re.I)

#: Test fixtures left in the parser's own unit test, not real options.
FIXTURE = {"ABC", "DEF", "-coption", "-fdata", "-idata", "-sdata", "-ulldata",
           "-someflag", "-otherflag", "-keep", "-o", "-D", "-I"}

#: Positional fields hidden inside one option name.  Counting option names
#: undercounts the surface; this is how much by, for the ones that matter.
COMPOSITE_FIELDS: dict[str, tuple[str, ...]] = {
    "-gpgpu_cache:dl1": (
        "cache_type", "n_set", "line_size", "assoc", "replacement_policy",
        "write_policy", "alloc_policy", "write_alloc_policy",
        "set_index_function", "mshr_type", "mshr_entries", "mshr_max_merge",
        "miss_queue_size", "result_fifo_entries", "data_port_width"),
    "-gpgpu_cache:dl2": (
        "cache_type", "n_set", "line_size", "assoc", "replacement_policy",
        "write_policy", "alloc_policy", "write_alloc_policy",
        "set_index_function", "mshr_type", "mshr_entries", "mshr_max_merge",
        "miss_queue_size", "result_fifo_entries", "data_port_width"),
    "-gpgpu_pipeline_widths": (
        "ID_OC_SP", "ID_OC_DP", "ID_OC_INT", "ID_OC_SFU", "ID_OC_MEM",
        "OC_EX_SP", "OC_EX_DP", "OC_EX_INT", "OC_EX_SFU", "OC_EX_MEM",
        "EX_WB", "ID_OC_TENSOR", "OC_EX_TENSOR"),
}


class KnobSpec(dict):
    """One registered option: name, C type, default, help text, source file."""

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def fields(self) -> tuple[str, ...]:
        return COMPOSITE_FIELDS.get(self["name"], ())


def classify(name: str, desc: str = "") -> str:
    """``fixture`` | ``operational`` | ``architectural``.

    Architectural is the residue: an option that survives both filters is one
    that plausibly changes the simulated machine.  It is a claim about intent,
    not about whether sweeping it works.
    """
    if name in FIXTURE or not name.startswith("-"):
        return "fixture"
    if OPERATIONAL_RX.search(name) or OPERATIONAL_RX.search(desc or ""):
        return "operational"
    return "architectural"


def scan_registrar(src_root: str) -> dict[str, KnobSpec]:
    """Walk the simulator source and read every option it registers."""
    out: dict[str, KnobSpec] = {}
    for root, _, files in os.walk(src_root):
        for fn in files:
            if not fn.endswith((".cc", ".cpp", ".c", ".h", ".cu")):
                continue
            path = os.path.join(root, fn)
            try:
                src = open(path, errors="ignore").read()
            except OSError:
                continue
            for m in _REG.finditer(src):
                name, ctype, _var, desc, dflt = m.groups()
                desc = re.sub(r"\s+", " ", desc).strip()
                out.setdefault(name, KnobSpec(
                    name=name, ctype=ctype, desc=desc,
                    default=dflt.strip(),
                    source=os.path.relpath(path, src_root),
                    kind=classify(name, desc),
                    fields=list(COMPOSITE_FIELDS.get(name, ()))))
    return out


def save_catalog(cat: dict, path: str = CATALOG_PATH) -> str:
    with open(path, "w") as fh:
        json.dump({k: dict(v) for k, v in sorted(cat.items())}, fh, indent=1)
    return path


def load_catalog(path: str = CATALOG_PATH) -> dict[str, KnobSpec]:
    with open(path) as fh:
        return {k: KnobSpec(**v) for k, v in json.load(fh).items()}


def coverage(covered: dict[str, str], cat: dict | None = None) -> dict:
    """Split the architectural surface into what the loop can and cannot touch.

    Args:
        covered: option name -> the knob family that reaches it. An option
            reached only as one positional field of a composite string still
            counts as covered *for that field*; the uncovered fields are
            reported separately because that is where the undercount lives.
    """
    cat = cat or load_catalog()
    arch = {k: v for k, v in cat.items() if v["kind"] == "architectural"}
    have = {k: v for k, v in arch.items() if k in covered}
    miss = {k: v for k, v in arch.items() if k not in covered}
    comp = {k: [f for f in v.get("fields") or ()
                if f not in (covered.get(k) or "")]
            for k, v in have.items() if v.get("fields")}
    return {"total": len(cat), "architectural": len(arch),
            "operational": sum(1 for v in cat.values()
                               if v["kind"] == "operational"),
            "fixture": sum(1 for v in cat.values() if v["kind"] == "fixture"),
            "covered": sorted(have), "uncovered": sorted(miss),
            "uncovered_fields": {k: v for k, v in comp.items() if v},
            "catalog": cat}


def format_menu(cov: dict, *, groups: dict[str, str] | None = None,
                show_uncovered: int = 0) -> str:
    """The agent-facing menu, generated from the catalog.

    Replaces the hand-written text that previously told an agent three things
    were available and everything else was impossible.  When
    *show_uncovered* is non-zero the menu also names options the simulator
    accepts but the loop does not yet drive, marked as such -- an agent that
    wants one of those is telling us what to build next, which is worth more
    than an agent that trims its diagnosis to fit.
    """
    lines = ["The simulator registers "
             f"{cov['architectural']} architectural options "
             f"({cov['total']} total; {cov['operational']} are reporting or "
             f"debug options and {cov['fixture']} are parser fixtures).",
             ""]
    if groups:
        lines.append("Driven by this loop, grouped by knob family:")
        for fam, what in sorted(groups.items()):
            lines.append(f"  {fam:<16} {what}")
        lines.append("")
    if cov.get("uncovered_fields"):
        lines.append("Composite options the loop drives only partly "
                     "(one option name, many positional fields):")
        for opt, flds in sorted(cov["uncovered_fields"].items()):
            lines.append(f"  {opt}: untouched fields = {', '.join(flds)}")
        lines.append("")
    if show_uncovered:
        un = cov["uncovered"][:show_uncovered]
        lines.append(f"Accepted by the simulator but NOT yet driven by this "
                     f"loop ({len(cov['uncovered'])} total"
                     + (f", first {len(un)} shown" if
                        len(cov["uncovered"]) > len(un) else "") + "):")
        cat = cov["catalog"]
        for k in un:
            d = (cat[k]["desc"] or "")[:64]
            lines.append(f"  {k:<44} {d}")
        lines.append("")
        lines.append("If your diagnosis points at one of these, say so and "
                     "name it. It is not currently wired up, but it is real, "
                     "and naming it is a result.")
    return "\n".join(lines)
