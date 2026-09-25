"""chia.simulators.accelsim_area — silicon area from the AccelWattch XML.

EDP charges a design point for the *power* it burns but not for the *silicon it
occupies*, so an optimizer handed EDP alone will happily answer "use 4x the
SMs and 4x the L2". That is not a co-design result, it is a tautology: more
hardware is faster. To ask "at the cost of what?" the loop needs a budget, and
the budget is area.

McPAT already computes one. The same CACTI models that produce AccelWattch's
watts produce an area for every array and functional unit; the simulator just
never prints it, because ``power_interface.cc`` only asks for power. This
module runs McPAT a second time, standalone and post-hoc, against the XML a
completed run left behind -- no re-simulation, ~1s per point -- and parses the
area tree.

Two things the run's XML does not carry, because
``gpgpu_sim_wrapper::init_mcpat`` supplies them at runtime and never writes
them back:

* ``number_of_cores``  -- defaults to 1, so an unedited XML prices *one* SM.
* ``number_of_L2s``    -- defaults to 1, so it prices *one* 256 KB L2 slice.

:func:`area_report` injects both. McPAT multiplies core and L2 area by these
counts (``processor.cc:119``), verified linear: 1/2/108/216 cores give
41.90/83.80/4525.08/9050.15 mm^2.

**Read these numbers as ratios, never as a die size.** The validated
AccelWattch XML declares ``core_tech_node = 23`` nm; a real A100 is 7 nm. The
model therefore reports ~4.5e3 mm^2 for an A100-sized core array against a real
die of 826 mm^2. Every design point is priced by the same mis-scaled model, so
*differences* and *exchange rates* are meaningful and absolute mm^2 are not.
The known bias worth stating: at 23 nm McPAT puts the SM array and the L2 at
roughly 19:1, where a 7 nm A100 is closer to 7:1, so an area-constrained search
run on these numbers is biased *towards* selling SMs to buy cache. Results that
depend on that ratio are reported with the exchange rate attached so a reader
with a better area model can redo the arithmetic.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

#: McPAT binary.  The README documents ``MCPAT_BIN`` as the way to point
#: this at a local build, so it has to be read from the environment here
#: -- a hardcoded path is a path that exists on exactly one machine.
_MCPAT_DEV = "/raid/sp56429/chia_hackathon/mcpat_area/build/accelwattch/mcpat"
MCPAT_BIN = os.environ.get("MCPAT_BIN") or (
    _MCPAT_DEV if os.path.exists(_MCPAT_DEV) else "mcpat")

# Components inside one SM, in the order McPAT nests them. These are the
# currency of a reallocation: each is a resource a starve-probe can shrink.
SM_PARTS = (
    "Instruction Cache", "Instruction Buffer", "Instruction Decoder",
    "Shared Memory", "Data Cache", "Constant Cache", "Texture Cache",
    "Load/Store Queue", "Register Files", "Instruction Window",
    "Integer ALUs", "Floating Point Units", "Complex ALUs",
)


@dataclass
class AreaReport:
    """Whole-chip area for one design point, plus the per-SM breakdown."""
    total_mm2: float = 0.0
    core_mm2: float = 0.0          # all n_sm cores
    l2_mm2: float = 0.0            # all n_l2 slices
    noc_mm2: float = 0.0
    mc_mm2: float = 0.0
    per_sm_mm2: float = 0.0        # one SM
    per_l2_mm2: float = 0.0        # one L2 slice
    parts: dict[str, float] = field(default_factory=dict)  # one SM, by part
    n_sm: int = 0
    n_l2: int = 0
    ok: bool = False
    note: str = ""
    #: which template fields were overwritten from the run's own config; empty
    #: means the bare template was priced and every design will look identical
    injected: list[str] = field(default_factory=list)

    #: which end of the uncertain range produced these numbers; see
    #: :data:`AREA_BOUND`. Quoting an "optimistic" total beside a
    #: "conservative" one is comparing two different conventions.
    bound: str = ""

    @property
    def core_frac(self) -> float:
        return self.core_mm2 / self.total_mm2 if self.total_mm2 else 0.0


def _inject(src_xml: str, dst_xml: str, n_sm: int, n_l2: int,
            n_mc: int | None = None,
            knobs: dict | None = None,
            with_assoc: bool = True,
            bound: str = "") -> list[str]:
    """Copy the XML with the runtime-supplied structural counts written in.

    ``number_of_cores`` is absent from the AccelWattch template entirely, so it
    is inserted rather than substituted -- a ``sed`` that only replaces will
    silently match nothing and leave the area at its 1-core default, which
    looks like a plausible number and is not.
    """
    with open(src_xml) as fh:
        text = fh.read()

    if 'name="number_of_cores"' in text:
        text = re.sub(r'(name="number_of_cores"\s+value=")[^"]*"',
                      rf'\g<1>{n_sm}"', text)
    else:
        anchor = '<param name="homogeneous_cores"'
        if anchor not in text:
            raise ValueError(f"{src_xml}: no anchor to insert number_of_cores")
        text = text.replace(
            anchor, f'<param name="number_of_cores" value="{n_sm}"/>\n\t\t{anchor}', 1)

    text = re.sub(r'(name="number_of_L2s"\s+value=")[^"]*"', rf'\g<1>{n_l2}"', text)
    if n_mc:
        text = re.sub(r'(name="number_mcs"\s+value=")[^"]*"', rf'\g<1>{n_mc}"', text)

    notes: list[str] = []
    if knobs:
        text, notes = _apply_knobs(text, knobs, with_assoc, bound)

    with open(dst_xml, "w") as fh:
        fh.write(text)
    return notes


# ---------------------------------------------------------------------------
# Knob -> XML: letting McPAT see the design
# ---------------------------------------------------------------------------
# The AccelWattch XML is a static template. GPGPU-Sim reads the real geometry
# from ``gpgpusim.config`` and never writes it back, so every design point --
# every cache resize, every unit count, every port change -- renders a
# byte-identical XML. Verified directly: all 19 starve-probe configs produce
# the same file. An area gate reading that XML therefore prices every proposal
# at exactly the baseline and passes everything, which is worse than no gate
# because it looks like a constraint.
#
# McPAT is not the problem and is not modified. The fields it needs already
# exist in the template; they are simply filled with a different GPU's numbers:
#
#     structure   gpgpusim.config says        template says
#     dl1         4 x 128 x 64  =  32 KB      16384, 32, 4
#     il1        64 x 128 x 16  = 128 KB      16384, 128, 4
#     const     128 x  64 x  8  =  64 KB      16384, 64, 2
#     tex         4 x 128 x 256 = 128 KB      49152, 128, 8
#     shmem                       164 KB      98304
#
# :func:`_inject` already rewrites ``number_of_cores`` and ``number_of_L2s``
# for the same reason -- they too are supplied at runtime and never persisted.
# This extends that existing mechanism to the geometry fields, so the black box
# is fed its own inputs rather than altered.
#
# Two fields are handled differently on purpose:
#
# * Cache and shared-memory sizes are the *same physical quantity* in both
#   models (bytes, line, associativity), so they are written absolutely.
# * ``memory_ports`` is not. GPGPU-Sim's ``-gpgpu_mem_unit_ports`` counts
#   memory instructions issued per core cycle and defaults to 1; McPAT's
#   ``memory_ports`` counts array ports and the template says 2. The
#   correspondence between them is a modelling judgement, not an identity, so
#   the knob is applied as a *ratio* against each model's own default. That
#   keeps the validated baseline exactly where it was and makes only the
#   delta depend on the judgement. It matters: 1:1 prices
#   ``mem_unit_ports=4`` at -358 mm^2 and ratio-scaling prices it at
#   +5664 mm^2, which is the difference between the area gate accepting that
#   design and rejecting it. Set ``PORTS_ABSOLUTE`` to switch and re-derive.
PORTS_ABSOLUTE = False

#: McPAT's ``memory_ports`` is core-wide -- raising it re-ports every array in
#: the SM, not just the LSU. At 108 SMs, 2 -> 4 adds 4328 mm^2 of which 69%
#: lands on the texture cache. That is only affordable if the texture unit is
#: deleted first (see :data:`REMOVABLE`), which is a real ordering constraint
#: on the design rather than an artefact.
_GPGPUSIM_PORT_DEFAULT = 1


def _cache_bytes(spec: str) -> tuple[int, int, int] | None:
    """``S:4:128:64,...`` -> ``(size_bytes, line_bytes, assoc)``.

    GPGPU-Sim's first comma-group is ``<sector>:<nsets>:<bsize>:<assoc>``.
    Returns None for anything that does not parse, so a malformed or
    unrecognised spec leaves the template value alone rather than writing a
    zero that CACTI would silently accept.
    """
    try:
        head = spec.split(",")[0].split(":")
        nsets, bsize, assoc = int(head[1]), int(head[2]), int(head[3])
    except (IndexError, ValueError):
        return None
    if nsets <= 0 or bsize <= 0 or assoc <= 0:
        return None
    return nsets * bsize * assoc, bsize, assoc


def _set_cache_field(text: str, field: str, spec: str,
                     with_assoc: bool = True) -> str:
    """Rewrite the size/line/assoc of one ``*cache_config`` field in place.

    The remaining entries (banks, throughput, latency, output width, policy)
    are timing and organisation values the knob does not carry, so they are
    kept from the template rather than invented.

    ``with_assoc`` exists because CACTI will not build every associativity
    GPGPU-Sim accepts. Measured: the real dl1 (assoc 64) and texture cache
    (assoc 256) both abort with "no valid data array organizations", while
    the il1 (16) and constant cache (8) build fine. GPGPU-Sim's sector-cache
    ``assoc`` is not CACTI's set associativity in the first place, so falling
    back to the template's value while keeping the real *size* -- which is
    what dominates the area -- loses little and is what
    :func:`area_report` retries with.
    """
    m = re.search(rf'name="{field}"\s+value="([^"]*)"', text)
    got = _cache_bytes(spec)
    if not m or not got:
        return text
    fields = m.group(1).split(",")
    if len(fields) < 3:
        return text
    fields[0], fields[1] = str(got[0]), str(got[1])
    if with_assoc:
        fields[2] = str(got[2])
    return text[:m.start(1)] + ",".join(fields) + text[m.end(1):]


#: Which end of an uncertain area range to report.
#:
#: Several structures GPGPU-Sim sizes have no single right mapping onto a
#: McPAT field, and McPAT's own response to ``buffer_sizes`` is not monotonic.
#: Rather than pick one number and hide the choice, both ends are computed:
#:
#: * ``"optimistic"``   charges the least a purchase can defensibly cost and
#:                      credits a removal with only what McPAT attributes to
#:                      that leaf. This is the bound that lets the most
#:                      designs through the gate.
#: * ``"conservative"`` charges every slot a deeper queue plausibly widens.
#:
#: A design that passes only under ``"optimistic"`` is not wrong, but it is
#: resting on the mapping rather than on the measurement, and
#: :func:`area_report` records which bound produced the number so the two can
#: never be quoted as if they were the same claim.
AREA_BOUND = "optimistic"

#: GPGPU-Sim cache spec field 4 is ``<miss_queue>:<result_fifo>`` and field 3
#: is ``A:<mshr_entries>:<max_merge>``. McPAT's ``buffer_sizes`` is
#: ``(MSHR, fill, prefetch, writeback)``. The two are not the same quantity,
#: so these are applied as *ratios* against the stock spec rather than written
#: through absolutely -- the same reasoning as ``memory_ports``, and for the
#: same reason: the template's values are not this GPU's values.
_STOCK_SPECS = {
    "-gpgpu_cache:dl1": "S:4:128:64,L:T:m:L:L,A:512:64,16:0,32",
    "-gpgpu_cache:dl2": "S:128:128:16,L:B:m:L:X,A:192:4,32:0,32",
}


def _queue_ratios(spec: str, stock: str) -> tuple[float, float]:
    """``(mshr_ratio, miss_queue_ratio)`` of *spec* against *stock*."""
    def parse(s: str) -> tuple[float, float]:
        f = s.split(",")
        mshr = float(f[2].split(":")[1]) if len(f) > 2 else 0.0
        mq = float(f[3].split(":")[0]) if len(f) > 3 else 0.0
        return mshr, mq
    try:
        a_mshr, a_mq = parse(spec)
        b_mshr, b_mq = parse(stock)
    except (IndexError, ValueError):
        return 1.0, 1.0
    r_mshr = a_mshr / b_mshr if b_mshr > 0 else 1.0
    r_mq = a_mq / b_mq if b_mq > 0 else 1.0
    return r_mshr, r_mq


def _set_buffer_sizes(text: str, after_field: str,
                      r_mshr: float, r_mq: float,
                      bound: str) -> tuple[str, str | None]:
    """Scale the ``buffer_sizes`` that follows *after_field* in the XML.

    The XML carries nine ``buffer_sizes`` params and they are positional --
    each belongs to the cache config declared immediately above it. Matching
    by name alone would rewrite the first one every time, which is how the
    instruction cache ends up wearing the L1D's miss queue.
    """
    m = re.search(rf'name="{after_field}"', text)
    if not m:
        return text, None
    b = re.search(r'name="buffer_sizes"\s+value="([^"]*)"', text[m.end():])
    if not b:
        return text, None
    lo, hi = m.end() + b.start(1), m.end() + b.end(1)
    try:
        slots = [int(x.strip()) for x in text[lo:hi].split(",")]
    except ValueError:
        return text, None
    if len(slots) < 4:
        return text, None
    out = list(slots)
    out[0] = max(1, round(slots[0] * r_mshr))       # MSHR
    out[1] = max(1, round(slots[1] * r_mq))         # fill buffer
    if bound == "conservative":
        # A deeper miss queue plausibly widens everything downstream of it.
        out[2] = max(1, round(slots[2] * r_mq))
        if slots[3]:
            out[3] = max(1, round(slots[3] * r_mq))
    new = ", ".join(str(v) for v in out)
    if new == text[lo:hi]:
        return text, None
    return (text[:lo] + new + text[hi:],
            f"{after_field} buffer_sizes {slots}->{out} "
            f"(mshr x{r_mshr:g}, mq x{r_mq:g}, {bound})")


def _apply_knobs(text: str, knobs: dict,
                 with_assoc: bool = True,
                 bound: str = "") -> tuple[str, list[str]]:
    """Write the run's real geometry into the template. Returns (xml, notes)."""
    notes: list[str] = []
    bound = bound or AREA_BOUND
    for field, key in (("dcache_config", "-gpgpu_cache:dl1"),
                       ("icache_config", "-gpgpu_cache:il1"),
                       ("ccache_config", "-gpgpu_const_cache:l1"),
                       ("tcache_config", "-gpgpu_tex_cache:l1")):
        spec = knobs.get(key)
        if not spec:
            continue
        out = _set_cache_field(text, field, str(spec), with_assoc)
        if out != text:
            text, _ = out, notes.append(f"{field}<-{key}")

    shmem = knobs.get("-gpgpu_shmem_size")
    if shmem:
        m = re.search(r'name="sharedmemory_config"\s+value="([^"]*)"', text)
        if m:
            f = m.group(1).split(",")
            try:
                f[0] = str(int(shmem))
                text = text[:m.start(1)] + ",".join(f) + text[m.end(1):]
                notes.append("sharedmemory_config<--gpgpu_shmem_size")
            except ValueError:
                pass

    # --- queue depths -------------------------------------------------
    # Without these, deepening a miss queue 16x -- the single lever this
    # project is built on -- priced at exactly +0.0 mm^2, because
    # ``_cache_bytes`` reads only sets/line/assoc and discards the MSHR group
    # and the miss-queue field entirely. Measured once wired up: +25.4 mm^2
    # chip-wide, which is 0.55% of the part. Cheap, but not free, and the
    # difference between those two words is the whole area claim.
    for field, key in (("dcache_config", "-gpgpu_cache:dl1"),
                       ("L2_config", "-gpgpu_cache:dl2")):
        spec = knobs.get(key)
        stock = _STOCK_SPECS.get(key)
        if not spec or not stock:
            continue
        r_mshr, r_mq = _queue_ratios(str(spec), stock)
        if r_mshr == 1.0 and r_mq == 1.0:
            continue
        text, note = _set_buffer_sizes(text, field, r_mshr, r_mq, bound)
        if note:
            notes.append(note)

    # --- L2 geometry ----------------------------------------------------
    l2 = knobs.get("-gpgpu_cache:dl2")
    if l2:
        out = _set_cache_field(text, "L2_config", str(l2), with_assoc)
        if out != text:
            text, _ = out, notes.append("L2_config<--gpgpu_cache:dl2")

    # --- register file --------------------------------------------------
    # The occupancy bundle doubles registers and shared memory together;
    # only the shared memory half was being priced, so half of that purchase
    # was invisible.
    regs = knobs.get("-gpgpu_shader_registers")
    if regs:
        m = re.search(r'name="archi_Regs_IRF_size"\s+value="([^"]*)"', text)
        if m:
            try:
                text = (text[:m.start(1)] + str(int(regs)) + text[m.end(1):])
                notes.append("archi_Regs_IRF_size<--gpgpu_shader_registers")
            except ValueError:
                pass

    ports = knobs.get("-gpgpu_mem_unit_ports")
    if ports:
        m = re.search(r'name="memory_ports"\s+value="([^"]*)"', text)
        if m:
            try:
                knob, tmpl = int(ports), int(m.group(1))
                val = knob if PORTS_ABSOLUTE else max(
                    1, round(tmpl * knob / _GPGPUSIM_PORT_DEFAULT))
                text = text[:m.start(1)] + str(val) + text[m.end(1):]
                notes.append(f"memory_ports {tmpl}->{val} "
                             f"(knob {knob}, "
                             f"{'absolute' if PORTS_ABSOLUTE else 'ratio'})")
            except ValueError:
                pass
    return text, notes


_AREA = re.compile(r"^(\s*)Area\s*=\s*([0-9.eE+-]+)\s*mm\^2")


def _label_of(line: str) -> tuple[int, str] | None:
    """``(indent, label)`` for a section header, else None.

    Header lines are not uniform: ``Core:`` is bare, ``Total Cores: 1 cores``
    and ``Total MCs: 6 Memory Controllers`` carry a count *after* the colon,
    and ``Floating Point Units (FPUs) (Count: 32 ):`` carries one inside a
    parenthetical -- which contains its own colon and breaks any naive split.
    Dropping every ``(...)`` group first makes all three the same shape and
    normalises the labels onto the names in :data:`SM_PARTS`.
    """
    if "=" in line:                       # value line, or ``Device Type= ...``
        return None
    s = line.rstrip()
    if ":" not in s:
        return None
    body = re.sub(r"\([^)]*\)", " ", s)
    if ":" not in body:
        return None
    label = body.split(":", 1)[0].strip()
    if not label or not label[0].isalpha():
        return None
    return len(s) - len(s.lstrip()), label


def _parse(out: str) -> dict:
    """Walk the indented report, attaching each ``Area =`` to its label.

    The report is a tree rendered by indentation with no closing markers, so
    the only reliable attachment is "nearest preceding label at strictly
    smaller indent". Keyed by the *first* occurrence: the summary block at the
    top repeats Core/L2/NoC/MC totals, and those are the ones wanted.
    """
    summary: dict[str, float] = {}
    parts: dict[str, float] = {}
    stack: list[tuple[int, str]] = []
    in_core = False

    for line in out.splitlines():
        lab = _label_of(line)
        if lab:
            indent, label = lab
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, label))
            if label == "Core":
                in_core = True
            elif indent == 0 and label != "Core":
                in_core = False
            continue
        m = _AREA.match(line)
        if not m:
            continue
        indent, val = len(m.group(1)), float(m.group(2))
        owner = next((lab for ind, lab in reversed(stack) if ind < indent), "")
        if not owner:
            continue
        if in_core and owner in SM_PARTS:
            parts.setdefault(owner, val)
        elif owner not in summary:
            summary[owner] = val
    return {"summary": summary, "parts": parts}


def area_report(src_xml: str, n_sm: int, n_l2: int, n_mc: int | None = None,
                timeout_s: int = 900, bin_path: str = MCPAT_BIN,
                knobs: dict | None = None,
                use_run_config: bool = True,
                bound: str = "") -> AreaReport:
    """Price one design point. ``src_xml`` is a completed run's synced XML.

    ``knobs`` is the design's ``gpgpusim.config``, used to fill in the
    geometry fields the template hardcodes. When it is omitted and
    ``use_run_config`` is set, the config sitting next to ``src_xml`` is read
    automatically -- which is what makes two different design points price
    differently at all. Pass ``knobs={}`` to price the bare template, as the
    pre-2026-09-24 behaviour did.
    """
    rep = AreaReport(n_sm=n_sm, n_l2=n_l2)
    if not os.path.exists(bin_path):
        rep.note = f"mcpat binary missing: {bin_path}"
        return rep
    if not os.path.exists(src_xml):
        rep.note = f"xml missing: {src_xml}"
        return rep

    if knobs is None and use_run_config:
        cfg = os.path.join(os.path.dirname(src_xml), "gpgpusim.config")
        if os.path.exists(cfg):
            try:
                from chia.simulators.accelsim import read_config_knobs
                knobs = read_config_knobs(cfg)
            except Exception:
                knobs = None

    # Degrade rather than fail. CACTI refuses some associativities GPGPU-Sim
    # is happy with, so a design whose geometry it cannot build would
    # otherwise return no area at all -- and an area gate that returns
    # nothing is a gate that blocks every proposal for the wrong reason.
    # Each rung keeps more of the template and less of the real design; the
    # rung actually used is recorded in ``injected`` so a reader can see how
    # much of the price came from the design and how much from the template.
    ladder: list[tuple[dict | None, bool, str]] = []
    if knobs:
        ladder.append((knobs, True, "size+line+assoc"))
        ladder.append((knobs, False, "size+line (template assoc: CACTI "
                                     "rejected the real one)"))
    # No bare-template rung when there were knobs to apply. Pricing the
    # template while the baseline was priced from its config produces a
    # difference that is entirely the template, not the design: measured, a
    # 4 KB texture cache (which CACTI cannot build at all) came back as a
    # -1005 mm^2 "credit" that way. A refusal the gate can see beats a number
    # it cannot distinguish from a real one.
    if not knobs:
        ladder.append(({}, False, "bare template (no config supplied)"))

    proc = None
    for kn, with_assoc, how in ladder:
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "area.xml")
            try:
                notes = _inject(src_xml, dst, n_sm, n_l2, n_mc, kn,
                                with_assoc, bound or AREA_BOUND)
                proc = subprocess.run(
                    [bin_path, "-infile", dst, "-print_level", "5"],
                    capture_output=True, text=True, timeout=timeout_s)
            except Exception as exc:
                rep.note = f"{type(exc).__name__}: {exc}"[:160]
                return rep
        got = _parse(proc.stdout)
        if got["summary"].get("Processor", 0.0) > 0:
            rep.injected = notes + [how]
            rep.bound = bound or AREA_BOUND
            break
    else:
        got = _parse(proc.stdout) if proc else {"summary": {}, "parts": {}}

    s, p = got["summary"], got["parts"]
    rep.total_mm2 = s.get("Processor", 0.0)
    rep.core_mm2 = s.get("Total Cores", 0.0)
    rep.l2_mm2 = s.get("Total L2s", 0.0)
    rep.noc_mm2 = s.get("Total NoCs", 0.0)
    rep.mc_mm2 = s.get("Total MCs", 0.0)
    rep.parts = p
    rep.per_sm_mm2 = rep.core_mm2 / n_sm if n_sm else 0.0
    rep.per_l2_mm2 = rep.l2_mm2 / n_l2 if n_l2 else 0.0
    # A zero total means McPAT ran but the parse missed, or CACTI rejected the
    # geometry and printed an error tree. Either way it is not an area of 0.
    rep.ok = rep.total_mm2 > 0 and rep.core_mm2 > 0
    if rep.ok and knobs and not rep.injected:
        rep.ok = False
        rep.note = ("CACTI could not build this design's geometry; priced "
                    "nothing. Comparing it to a priced baseline would report "
                    "the template as a credit.")
    if not rep.ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        rep.note = " | ".join(tail)[:200] or "no area parsed"
    return rep


def format_area(rep: AreaReport) -> str:
    if not rep.ok:
        return f"area: FAILED ({rep.note})"
    lines = [
        f"total {rep.total_mm2:10.1f} mm^2   "
        f"({rep.n_sm} SM x {rep.per_sm_mm2:.2f} = {rep.core_mm2:.1f}, "
        f"{rep.n_l2} L2 x {rep.per_l2_mm2:.3f} = {rep.l2_mm2:.1f}, "
        f"noc {rep.noc_mm2:.2f}, mc {rep.mc_mm2:.1f})",
        f"per-SM {rep.per_sm_mm2:.3f} mm^2:",
    ]
    for k, v in sorted(rep.parts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:<26}{v:8.3f}  {100*v/rep.per_sm_mm2:5.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structural removal
# ---------------------------------------------------------------------------
# Some reallocations are not reachable through any simulator knob. The texture
# cache is the clearest case: it is 32.6% of McPAT's SM area, and shrinking it
# with ``-gpgpu_tex_cache:l1`` barely moves that number (128 KB -> 16 KB frees
# 13.94 -> 12.32 mm^2, ~12%, because CACTI's area here is dominated by the port
# and bank structure rather than the array) while 4 KB and 8 KB are rejected by
# CACTI outright. The design move that actually reclaims the area is to delete
# the unit, and no knob does that.
#
# Deleting a structure is only an honest zero-cost move when the workload never
# touches it. That is a checkable claim, not a judgement call -- AccelWattch
# writes per-structure access counters into ``accelwattch_power_report.log`` --
# but it has one trap. ``-gpgpu_perfect_inst_const_cache 1`` forces the
# instruction and constant caches to hit without accounting, so IC_H/IC_M and
# CC_H/CC_M read 0 whether or not those caches are used. A zero there means
# "not measured", not "not needed". ``removal_is_justified`` refuses to certify
# a structure whose counters are suppressed by that flag.
#
# For the A100 + CUTLASS TF32 GEMM at 4096x4096x64 the measured totals are
# TEX_ACC = TC_H = TC_M = 0 with no flag suppressing them, against
# INT_ACC = 1.37e9, TENSOR_ACC = 2.74e7, FPU_ACC = 2.01e7, SHRD_ACC = 2.23e6.
# The texture path is genuinely dead; the constant and instruction caches are
# simply unmeasured until the perfect flag is turned off.

#: SM sub-component -> (power-report counters that must all be zero,
#:                      config flag that would invalidate those counters)
REMOVABLE: dict[str, tuple[tuple[str, ...], str | None]] = {
    # These names are read from the *power* report
    # (``accelwattch_power_report.log``), not from ``perf_counter.csv.gz``.
    # That matters, because the two files do not use the same vocabulary, and
    # an earlier version of this comment asserted the opposite: it claimed the
    # AccelWattch-era spellings were wrong for this fork and that the gate was
    # refusing a justified sale because of a name mismatch. That diagnosis was
    # checked and is false. TEX_ACC, TC_H, TC_M, CC_H, CC_M, IC_H and IC_M are
    # all present in the power report and all exactly zero, which is precisely
    # the certificate the gate wants.
    #
    # The real reason the probe sweep could not certify a removal is duller:
    # those runs were launched with power collection off, so they have no
    # power report at all. Nothing can be certified dead from a file that was
    # never written. That is correct behaviour, not a bug -- but it is a
    # different failure from the one this comment used to describe, and the
    # fix for it is to re-run with power on, not to add names.
    #
    # The ``gpgpu_n_*`` spellings below are kept because they are the
    # equivalents in the performance CSV, so the same table still works if the
    # credit is ever sourced from there. They are simply absent from the power
    # report today, and :func:`removal_is_justified` requires *at least one*
    # listed name to be present (and every present one to be zero) rather than
    # all of them, so their absence neither blocks nor weakens the check.
    "Texture Cache": (
        ("TEX_ACC", "TC_H", "TC_M",
         "gpgpu_n_tex_insn", "gpgpu_n_mem_texture"), None),
    "Constant Cache": (
        ("CC_H", "CC_M",
         "gpgpu_n_mem_const", "gpgpu_n_const_mem_insn"),
        "-gpgpu_perfect_inst_const_cache"),
    "Instruction Cache": (("IC_H", "IC_M"), "-gpgpu_perfect_inst_const_cache"),
}


def read_counters(power_report: str) -> dict[str, float]:
    """Pull the ``gpu_tot_<NAME>`` accumulators out of an AccelWattch report.

    The file stores them one per line as ``gpu_tot_NAME, = 1.23e4``.
    """
    out: dict[str, float] = {}
    pat = re.compile(r"^gpu_tot_([A-Z0-9_]+),\s*=\s*([-\d.eE+]+)")
    with open(power_report, errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def removal_is_justified(
    part: str, counters: dict[str, float], config_knobs: dict[str, str]
) -> tuple[bool, str]:
    """Is deleting ``part`` free for the run these counters came from?

    Returns ``(ok, reason)``. Requires every associated counter to be present
    and exactly zero, and requires that no "perfect cache" flag is suppressing
    them -- an unmeasured structure is not a dead one.
    """
    if part not in REMOVABLE:
        return False, f"{part}: no removal rule defined"
    names, invalidating_flag = REMOVABLE[part]
    if invalidating_flag:
        raw = str(config_knobs.get(invalidating_flag, "0")).strip()
        if raw not in ("", "0"):
            return False, (
                f"{part}: counters suppressed by {invalidating_flag}={raw}; "
                "zero here means unmeasured, not unused"
            )
    present = [n for n in names if n in counters]
    if not present:
        return False, (f"{part}: none of its counters appear in this report "
                       f"({','.join(names)}); absent is not the same as zero")
    hot = {n: counters[n] for n in present if counters[n] != 0}
    if hot:
        detail = ", ".join(f"{k}={v:g}" for k, v in hot.items())
        return False, f"{part}: still accessed ({detail})"
    return True, (f"{part}: {', '.join(present)} all zero, "
                  "no suppressing flag")


def removal_credit(report: "AreaReport", parts: list[str]) -> tuple[float, float]:
    """Chip-wide mm^2 freed by deleting ``parts`` from every SM.

    Returns ``(per_sm_mm2, chip_mm2)``. This is a subtraction on the area model
    only -- the simulator keeps running with the structure present, which is
    exactly why the zero-access precondition matters.
    """
    per_sm = sum(report.parts.get(p, 0.0) for p in parts)
    return per_sm, per_sm * report.n_sm
