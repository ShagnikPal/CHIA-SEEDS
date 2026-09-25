"""An LLM architect on the hardware axis, with the deterministic chain as control.

``accelsim_codesign.codesign_loop`` decides everything by threshold: the
attribution names a limiter, ``LIMITER_KNOB`` maps it onto a knob, a cross
product is swept, EDP ranks it and a band vetoes it.  It produced this
project's headline design (``t128x128x16_s3 + queue16``, +8.72%..+13.54%
sign-stable over SM 104-112) with no model anywhere in it.

This module puts a model where those thresholds are, and *nothing else
changes*.  The measurements handed to the agent are produced by the same three
functions the loop calls, in the same order, from the same baseline run:
``fan`` over ``evaluate_point`` for the baseline, ``CODESIGN_PROBES`` +
``probe_report`` for headroom, ``attribute_failures`` for the limiter.  The
verification afterwards is the same three-arm ``sign_stable_band``.  What
differs is the middle: instead of ``propose_hw_points`` and ``propose_tiles``
returning a fixed menu, an agent reads the attribution and spends a
**simulation budget** on whatever points it wants, then commits to one design
and says why.

``accelsim_codesign.py`` is deliberately not imported-and-patched, only
imported.  It has to keep producing the recorded result unchanged, because
that is what makes it a control arm rather than a previous draft.

Three design decisions worth stating, because each of them could have been
made the flattering way:

**The agent gets its own ``out_root``.**  ``evaluate_point`` caches per
output directory, so pointing the agent at the deterministic chain's tree
would hand it every point that chain already measured, for free, and the
comparison would be between an agent with 40 cached simulations and a chain
with 40.  It gets an empty tree instead: 16 simulations, all of them paid for.

**Cache hits inside the agent's own tree are still free.**  That is not
generosity, it is the correct accounting -- re-requesting a point it already
measured is a no-op, and charging for it would only teach the model to keep
its own notes rather than to explore.

**``unapplied`` is surfaced verbatim.**  GPGPU-Sim accepts several knobs it
then ignores; eleven of this project's probes measured exactly 0.00% for that
reason.  A model told only "no change" would reasonably conclude the structure
does not matter.  It is told "the simulator never applied this", which is a
different fact and the one that is true.
"""
from __future__ import annotations

import json
import os
import time

from chia.base.tools.ChiaTool import ChiaTool
from chia.simulators.accelsim_codesign import (
    ADAPTIVE_OFF,
    LIMITER_KNOB,
    Band,
    LoopConfig,
    attribute_failures,
    evaluate_point,
    fan,
    format_attribution,
    occupancy_knobs,
    pct_gain,
    queue_knobs,
    sign_stable_band,
    sm_knobs,
    trace_dir_name,
    trace_name,
    writeback_knobs,
)
from chia.simulators.accelsim_dse import (
    BASELINE_TILE,
    LEGAL_TILES,
    resident_ctas,
)

#: Knob grammar the agent writes in.  Each entry is ``(builder, legal
#: multipliers or None for "any positive int", one-line description)``.  The
#: family name is the key, and two knobs from the same family may not appear
#: in one point -- "queue4+queue16" is not a composition, it is queue16 with a
#: misleading label, and reading it as a synergy would be an artefact.
#: An unbounded multiplier turns a design knob back into an idealization
#: probe.  ``queue99999`` is a 1.6M-entry FIFO: it would "win", it would price
#: absurdly in the area ledger, and proposing it skips the entire second half
#: of the method -- find what the machine is starved of by making it
#: effectively infinite, then buy a *realisable* amount of it by giving up
#: something else.  The deterministic chain proposes 4x and 16x; 64x is
#: already generous for a structure that has to be built.
MAX_KNOB_MULT = 64

KNOB_GRAMMAR = {
    "queue": (queue_knobs, None,
              "miss-queue depth in L1D and L2, xN. NOT the MSHRs -- the MSHR "
              "groups are a different field and are left alone."),
    "occ": (occupancy_knobs, (2, 4),
            "the four co-limiting residency resources (registers, shared "
            "memory, unified L1D size, legal shmem partitions) scaled "
            "together. Only 2 and 4 exist: the shared-memory figure is the "
            "hardware's next legal partition size, not a multiple."),
    "wb": (writeback_knobs, None,
           "EX_WB pipeline width, i.e. writeback bandwidth. The other 12 "
           "pipeline stages stay stock."),
    "mshr": (LIMITER_KNOB["mshr"][1], None,
             "MSHR entries in L1D and L2, xN."),
    "l1_capacity": (LIMITER_KNOB["line_alloc"][1], None,
                    "unified L1D size in KiB, xN from the stock 192."),
}


def parse_knobs(spec: str) -> tuple[dict, str]:
    """``"queue4+occ2"`` -> the knob dict, or ``({}, reason)`` if it is not legal.

    Refusing with a reason rather than a rejection is the point: the model
    only ever sees what this function returns, so "no such family" and "that
    multiplier was never measured" have to be distinguishable from "the
    simulator ran it and nothing happened".
    """
    spec = (spec or "").strip()
    if spec in ("", "stock", "none"):
        return dict(ADAPTIVE_OFF), ""
    knobs, seen = dict(ADAPTIVE_OFF), set()
    for term in spec.split("+"):
        term = term.strip()
        fam = next((f for f in sorted(KNOB_GRAMMAR, key=len, reverse=True)
                    if term.startswith(f)), None)
        if fam is None:
            return {}, (f"'{term}': no such knob family. Legal families: "
                        f"{', '.join(sorted(KNOB_GRAMMAR))}, or 'stock'.")
        if fam in seen:
            return {}, (f"'{spec}' names the {fam} family twice. Two "
                        f"multipliers of one family do not compose -- the "
                        f"point would just be the larger one.")
        seen.add(fam)
        rest = term[len(fam):]
        try:
            mult = int(rest)
        except ValueError:
            return {}, (f"'{term}': expected an integer multiplier after "
                        f"'{fam}', got '{rest}'.")
        builder, legal, _doc = KNOB_GRAMMAR[fam]
        if mult < 1:
            return {}, f"'{term}': multiplier must be >= 1."
        if mult > MAX_KNOB_MULT:
            return {}, (f"'{term}': multiplier is capped at {MAX_KNOB_MULT}. "
                        f"Above that this is an idealization probe, not a "
                        f"design -- the structure still has to be built and "
                        f"paid for in area. The probe sweep you were shown "
                        f"already tells you what an infinite one is worth; "
                        f"the question here is how much of it to buy.")
        if legal is not None and mult not in legal:
            return {}, (f"'{term}': only {legal} exist for '{fam}'; those are "
                        f"the bundles that were actually measured.")
        try:
            knobs.update(builder(mult))
        except Exception as exc:                       # builder's own veto
            return {}, f"'{term}': {exc}"
    return knobs, ""


def traced_tiles(trace_root: str, m: int, n: int, k: int) -> dict:
    """Tiles with a trace on disk for *this* problem shape, name -> kernelslist.

    Shape matters: the directory name encodes it, so a trace for a different
    GEMM would replay here unnoticed and report a cycle count for a problem
    nobody asked about.
    """
    out = {}
    for t in (BASELINE_TILE,) + tuple(LEGAL_TILES):
        tn = trace_name(t)
        p = os.path.join(trace_root, trace_dir_name(tn, m, n, k),
                         "traces", "kernelslist.g")
        if os.path.exists(p):
            out[tn] = p
    base = os.path.join(trace_root, "base", "traces", "kernelslist.g")
    if os.path.exists(base):
        out.setdefault("base", base)
        out["base"] = base
    return out


class CoDesignTool(ChiaTool):
    """The architect agent's entire world: three free reads and two actions.

    Everything operational -- where Accel-Sim lives, which traces exist, how
    big the problem is, where results are written -- is bound at construction
    and never appears as a model-facing argument.  The agent's vocabulary is
    "tell me what was measured", "tell me what I am allowed to change", "run
    these points", "this is my design and here is why".
    """

    def __init__(self, name: str, *, accelsim_root: str, cuda_path: str,
                 trace_root: str, out_root: str, m: int, n: int, k: int,
                 n_sm: int = 108, budget: int = 16,
                 baseline: dict | None = None, probe_report: dict | None = None,
                 attribution=None, task_options=None):
        super().__init__(name, task_options=task_options)
        self._a, self._c = accelsim_root, cuda_path
        self._trace_root, self._out_root = trace_root, out_root
        self._m, self._n, self._k, self._n_sm = m, n, k, n_sm
        self._budget = int(budget)
        self._spent = 0
        self._baseline = baseline or {}
        self._probe = probe_report or {}
        self._attrib = attribution
        self._traces = traced_tiles(trace_root, m, n, k)
        self._ledger = os.path.join(out_root, "agent_ledger")
        os.makedirs(self._ledger, exist_ok=True)
        self._design: dict | None = None
        self._calls = 0
        self.mcp.add_tool(self.get_measurements, name=f"{name}_get_measurements")
        self.mcp.add_tool(self.knob_vocabulary, name=f"{name}_knob_vocabulary")
        self.mcp.add_tool(self.tile_vocabulary, name=f"{name}_tile_vocabulary")
        self.mcp.add_tool(self.simulate, name=f"{name}_simulate")
        self.mcp.add_tool(self.propose_design, name=f"{name}_propose_design")
        super().__post_init__()

    # -- persistence ------------------------------------------------------
    # Tool methods run inside the tool's Ray actor, so anything stored on
    # ``self`` exists only there; the driver holds a different object and
    # would report an empty run however much the agent did.  (That is not
    # hypothetical -- it is exactly how the kernel forge's first run came
    # back "submissions: 0" after grading five kernels.)  Disk is the copy
    # that survives the actor boundary.
    def _write(self, stem: str, obj) -> None:
        try:
            with open(os.path.join(self._ledger, stem), "w") as fh:
                json.dump(obj, fh, indent=1, default=str)
        except OSError:
            pass

    # -- free reads -------------------------------------------------------
    def get_measurements(self) -> str:
        """What the simulator already measured on the stock chip. Free to call.

        Three things, all produced before you were asked anything: the
        baseline run, an idealization sweep saying which structures are even
        on the critical path, and an attribution of the baseline's own
        reservation failures by reason. The attribution is the one that names
        a cause; the sweep only bounds what is worth chasing.
        """
        b = self._baseline
        lines = [
            f"PROBLEM   {self._m}x{self._n}x{self._k} GEMM, {self._n_sm} SMs, "
            f"stock A100 (SM80), adaptive L1 off.",
            # Printing zeros for a baseline that was never run would read as a
            # measurement of zero.  Absence has to look like absence.
            (f"BASELINE  tile {trace_name(BASELINE_TILE)}: "
             f"{float(b['cycles']):,.0f} cycles, "
             f"{float(b.get('power_w') or 0):.1f} W, "
             f"{float(b.get('joules') or 0) * 1e3:.2f} mJ, "
             f"occupancy {b.get('occupancy')}, "
             f"{b.get('cta_per_core')} CTA/SM."
             if b.get("cycles") else
             "BASELINE  NOT MEASURED -- no baseline was handed to this tool. "
             "Simulate a stock point yourself before comparing anything."),
            "",
            "IDEALIZATION SWEEP -- each probe makes one structure effectively",
            "infinite and re-runs. The gain is an upper bound on what any",
            "realisable version of that structure could ever be worth. The",
            "starvation probes go the other way, shrinking one structure to",
            "price what giving it up would cost; both halves are below.",
        ]
        if not (self._probe.get("ranked") or []):
            lines.append("  (no probe sweep was handed to this tool)")
        for h in (self._probe.get("ranked") or []):
            nm = getattr(h, "name", None) or (
                h.get("name") if isinstance(h, dict) else str(h))
            pc = getattr(h, "headroom_pct", None)
            if pc is None and isinstance(h, dict):
                pc = h.get("headroom_pct")
            reads = getattr(h, "reads_as", "") or (
                h.get("reads_as", "") if isinstance(h, dict) else "")
            try:
                lines.append(f"  {str(nm):<22}{float(pc):>+8.2f}%   {reads}")
            except (TypeError, ValueError):
                lines.append(f"  {str(nm):<22}{'n/a':>9}   {reads}")
        flat = list(self._probe.get("no_headroom") or ())
        if flat:
            lines += ["",
                      f"APPLIED BUT FLAT: {', '.join(map(str, flat))}",
                      "  The knob was applied and the cycle count did not",
                      "  move. That is a real measurement: the structure is",
                      "  not on the critical path."]
        inert = list(self._probe.get("inert") or ())
        un = self._probe.get("unapplied") or {}
        if inert or un:
            lines += ["", f"INERT (never applied): "
                          f"{', '.join(map(str, inert)) or 'none'}"]
            for k, v in (un.items() if isinstance(un, dict) else ()):
                lines.append(f"  {k}: simulator ignored "
                             f"{', '.join(map(str, v))}")
            lines += ["  A 0.00% from an inert probe is not a measurement and",
                      "  must not be used to rule the structure out."]
        # Both directions, when the sweep collected them. Without this the
        # model sees only "does more of X help?" and will happily propose
        # funding its top lever by trimming that same structure elsewhere --
        # L2 buys +9.19% and sells for -13.69%, so that trade loses.
        pg = self._probe.get("pairs") or {}
        if pg.get("rows"):
            # Local import: accelsim_probes is imported lazily elsewhere in
            # this module to keep the import graph acyclic.
            from chia.simulators.accelsim_probes import format_pair_gradient
            lines += ["", format_pair_gradient(pg),
                      "",
                      "Read the ratio column before proposing any trade. A",
                      "structure with ratio > 1.00 costs more to give up than",
                      "it pays to acquire; it is a target, never a source of",
                      "area. Spend from FREE TO SURRENDER first."]
        if self._attrib is not None:
            lines += ["", "FAILURE ATTRIBUTION (the baseline's own counters):",
                      format_attribution(self._attrib)]
        lines += [
            "",
            "Note on the SM axis: this machine's cycle count swings +-5..8%",
            "per 2-SM step, deterministically. Any claim you make will be",
            "re-measured across SM 104-112 afterwards and thrown out if the",
            f"sign is not stable. Your {self._budget} simulations all run at "
            f"SM {self._n_sm}; do not chase differences below ~1%.",
        ]
        return "\n".join(lines)

    def knob_vocabulary(self) -> str:
        """The hardware changes you are allowed to ask for. Free to call.

        A point's knob string is either ``stock`` or one or more family+
        multiplier terms joined by ``+``, e.g. ``queue16``, ``occ2``,
        ``queue4+occ2``. Two terms from the same family are refused.
        """
        out = ["Knob string grammar:  stock | family<N>[+family<N>...]", ""]
        for fam, (_b, legal, doc) in sorted(KNOB_GRAMMAR.items()):
            m = f"multipliers {legal}" if legal else "any integer multiplier"
            out.append(f"  {fam:<12} ({m})")
            out.append(f"       {doc}")
        out += [
            "",
            "Examples: stock, queue16, occ2, wb4, mshr4, l1_capacity2,",
            "          queue4+occ2, queue16+wb4",
            "",
            "Every point also carries -gpgpu_adaptive_cache_config=0, because",
            "adaptive L1 resizing otherwise re-partitions underneath the knob",
            "and the change you asked for is not the change that ran.",
            "",
            "Each simulated point reports 'unapplied': knobs GPGPU-Sim",
            "accepted on the command line and then ignored. If a knob is in",
            "that list, its result tells you nothing about the structure.",
        ]
        return "\n".join(out)

    def tile_vocabulary(self) -> str:
        """The software side: CUTLASS tiles with a trace for this shape. Free.

        These are not all the tiles CUTLASS can emit -- they are the ones
        already traced on real hardware for this exact problem shape. A tile
        without a trace cannot be simulated here.
        """
        rows = [f"{'tile':<22}{'threads':>8}{'shmem B':>9}{'CTA/SM':>8}"
                f"{'blocks':>8}   note"]
        seen: set = set()
        for t in (BASELINE_TILE,) + tuple(LEGAL_TILES):
            tn = trace_name(t)
            if tn not in self._traces or tn in seen:
                continue
            seen.add(tn)
            ok, why = t.feasible()
            note = "BASELINE -- the chip you are trying to beat" \
                if t.name == BASELINE_TILE.name else ""
            if not ok:
                note = f"infeasible: {why}"
            rows.append(f"{tn:<22}{t.threads:>8}{t.smem_bytes:>9}"
                        f"{resident_ctas(t):>8}{t.blocks(self._n):>8}   {note}")
        untraced = sorted({trace_name(t) for t in LEGAL_TILES}
                          - set(self._traces))
        rows += [
            "",
            "CTA/SM is residency: how many threadblocks fit on one SM at once.",
            f"blocks is how many this {self._n}x{self._n} problem launches in "
            f"total; there are {self._n_sm} SMs.",
        ]
        if untraced:
            rows += ["", "Legal CUTLASS tiles with no trace for this shape, so "
                         "not simulatable here:", "  " + ", ".join(untraced)]
        return "\n".join(rows)

    # -- budgeted actions -------------------------------------------------
    async def simulate(self, points: list[str]) -> str:
        """Simulate a batch of design points. Costs budget; batch, do not drip.

        Each point is the string ``"<tile>/<knobs>"``, e.g.
        ``"t128x128x16_s3/queue16"`` or ``"base/stock"``. Pass the whole batch
        in one call: they run in parallel on the cluster, so twelve points in
        one call take about as long as one, and twelve separate calls take
        twelve times as long.

        Budget is charged per point that actually ran. A point you already
        simulated returns from cache and is free. An illegal point is refused
        with a reason and is free.

        Returns, per point: cycles, watts, EDP, occupancy, CTA/SM, and
        ``unapplied`` -- any knob the simulator ignored.
        """
        import asyncio
        return await asyncio.to_thread(self._simulate_sync, points)

    def _simulate_sync(self, points: list[str]) -> str:
        """Body of :meth:`simulate`, run off the event loop.

        FastMCP calls a *synchronous* tool function directly on uvicorn's
        event loop -- ``FuncMetadata.call_fn_with_arg_validation`` ends in a
        bare ``return fn(**kwargs)`` when ``fn_is_async`` is false. This body
        blocks for as long as GPGPU-Sim takes, which is tens of minutes, so
        leaving it synchronous wedged the entire MCP server rather than just
        the one call: during the agent's first 10-point batch even
        ``knob_vocabulary``, which had answered instantly moments earlier,
        timed out. The agent diagnosed that correctly as a wedged server and
        spent its remaining hour retrying instead of measuring, finishing
        with 0 of 16 simulations and no proposal. Running the body in a
        worker thread keeps the loop free to serve the free endpoints while
        a batch is in flight.
        """
        import ray
        self._calls += 1
        if self._design is not None:
            return ("You already called propose_design; the run is closed. "
                    "No further simulations are accepted.")
        if isinstance(points, str):
            points = [points]
        specs, refs, labels, refused = [], [], [], []
        for p in points:
            tile, _, kspec = str(p).partition("/")
            tile = tile.strip() or "base"
            if tile not in self._traces:
                refused.append(f"{p}: no trace for tile '{tile}'. Call "
                               f"tile_vocabulary for the ones that exist.")
                continue
            knobs, why = parse_knobs(kspec)
            if why:
                refused.append(f"{p}: {why}")
                continue
            name = (kspec.strip() or "stock")
            if self._spent + len(specs) >= self._budget:
                refused.append(f"{p}: budget exhausted ({self._budget} "
                               f"simulations). Propose a design with what "
                               f"you have.")
                continue
            outdir = os.path.join(self._out_root, tile, "agent",
                                  name.replace("/", "_"))
            specs.append((tile, name))
            labels.append(f"{tile}/{name}")
            refs.append(evaluate_point.chia_remote(
                self._a, self._c, tile, f"{tile}/{name}", self._traces[tile],
                outdir, {**knobs, **sm_knobs(self._n_sm)},
                True, None, False, 10800))

        res = fan(refs, labels, f"agent batch {self._calls}",
                  log=lambda *_a, **_k: None) if refs else []
        by = {str(r.get("point")): r for r in res}
        ran = 0
        rows = [f"{'point':<34}{'cycles':>11}{'W':>7}{'EDP':>12}"
                f"{'occ%':>7}{'CTA/SM':>8}  flags"]
        recs = []
        for tile, name in specs:
            r = by.get(f"{tile}/{name}", {})
            recs.append(r)
            if not r.get("ok"):
                rows.append(f"{tile + '/' + name:<34}  FAILED: "
                            f"{str(r.get('note', ''))[:70]}")
                continue
            cached = bool(r.get("cached"))
            if not cached:
                ran += 1
            un = r.get("unapplied") or []
            flags = []
            if cached:
                flags.append("cached, free")
            if un:
                flags.append(f"UNAPPLIED: {','.join(map(str, un))[:70]}")
            rows.append(
                f"{tile + '/' + name:<34}{float(r['cycles']):>11,.0f}"
                f"{float(r.get('power_w') or 0):>7.1f}"
                f"{float(r.get('edp') or 0):>12.4g}"
                f"{float(r.get('occupancy') or 0):>7.2f}"
                f"{str(r.get('cta_per_core')):>8}  {'; '.join(flags)}")
        self._spent += ran
        self._write(f"simulate_{self._calls:02d}.json",
                    {"points": labels, "records": recs, "refused": refused,
                     "charged": ran, "spent": self._spent})
        base = self._baseline.get("cycles")
        if base:
            rows.append("")
            rows.append(f"(baseline for reference: {float(base):,.0f} cycles)")
        if refused:
            rows += ["", "REFUSED (no budget charged):"] + \
                    [f"  {x}" for x in refused]
        rows += ["", f"budget: {ran} charged this call, {self._spent} spent, "
                     f"{self._budget - self._spent} left."]
        return "\n".join(rows)

    def propose_design(self, tile: str, knobs: str, rationale: str) -> str:
        """Commit to one design. Call this once, at the end.

        ``tile`` and ``knobs`` are the same strings ``simulate`` takes.
        ``rationale`` must say which measurement licensed the choice -- the
        design is recorded with it, and a design without the chain that
        produced it is not reproducible.

        What happens next is not yours to decide: the design is re-measured
        against the stock baseline across SM 104-112, split into the tile's
        contribution and the knob's, and any claim whose sign flips inside
        that band is recorded as not quotable.
        """
        if self._design is not None:
            # First commitment stands.  A design that can be revised after
            # the fact is not a commitment, and "which one did it propose"
            # has to have one answer.
            d = self._design
            return (f"already committed to {d['tile']} + {d['knobs']}; that "
                    f"stands. Nothing was changed.")
        tile = (tile or "").strip()
        if tile not in self._traces:
            return (f"no trace for tile '{tile}'; nothing recorded. Call "
                    f"tile_vocabulary.")
        kn, why = parse_knobs(knobs)
        if why:
            return f"{why}  Nothing recorded; fix the knob string and retry."
        if not (rationale or "").strip():
            return "rationale is required; nothing recorded."
        self._design = {"tile": tile, "knobs": (knobs or "stock").strip(),
                        "knob_dict": kn, "rationale": rationale,
                        "sims_spent": self._spent, "sim_calls": self._calls}
        self._write("design.json", self._design)
        return (f"recorded: {tile} + {self._design['knobs']}, after "
                f"{self._spent} of {self._budget} simulations. The band will "
                f"now decide whether it is quotable.")

    # -- driver side ------------------------------------------------------
    def read_ledger(self) -> dict:
        """What the agent actually did, read back from disk by the driver."""
        out = {"design": None, "batches": []}
        d = os.path.join(self._ledger, "design.json")
        if os.path.exists(d):
            out["design"] = json.load(open(d))
        for f in sorted(os.listdir(self._ledger)):
            if f.startswith("simulate_"):
                out["batches"].append(json.load(
                    open(os.path.join(self._ledger, f))))
        out["spent"] = sum(b.get("charged", 0) for b in out["batches"])
        return out


# ---------------------------------------------------------------------------
# The two halves the agent sits between.  Both are the deterministic chain's
# own code, called the same way: measurement before, verification after.  The
# agent only replaces what happens in between.
# ---------------------------------------------------------------------------

def measure(cfg: LoopConfig, out_root: str, *, log=print) -> dict:
    """Steps 1-3 of ``codesign_loop``, verbatim: baseline, probes, attribution.

    These are measurements, not decisions, so they are not the agent's job and
    must not differ between the arms -- if the two arms were handed different
    numbers the comparison would be about the numbers.  The probe sweep in
    particular is 14 simulations that neither arm chose, so it is not charged
    to either budget.
    """
    from chia.simulators.accelsim_dse import CODESIGN_PROBES
    from chia.simulators.accelsim_probes import probe_report, format_probe_report

    A, C = cfg.accelsim_root, cfg.cuda_path
    traces = traced_tiles(cfg.trace_root, cfg.m, cfg.n, cfg.k)
    base_t = trace_name(BASELINE_TILE)
    if base_t not in traces:
        raise SystemExit(f"no baseline trace for {base_t} at {cfg.trace_root}")
    stock = {**ADAPTIVE_OFF, **sm_knobs(cfg.n_sm)}

    def ev(point, knobs, *, power=True, sub="pts", trace_knobs=None):
        return evaluate_point.chia_remote(
            A, C, base_t, f"{base_t}/{point}", traces[base_t],
            os.path.join(out_root, base_t, sub, point), knobs,
            power, dict(trace_knobs or {}), False, 10800)

    log("\n=== 1. baseline")
    b = fan([ev("baseline", stock)], ["baseline"], "baseline", log=log)[0]
    if not b.get("ok"):
        raise SystemExit(f"baseline failed: {b.get('note')}")
    log(f"  {float(b['cycles']):,.0f} cycles  {b['power_w']:.1f} W  "
        f"{b.get('cta_per_core')} CTA/SM")

    log("\n=== 2. idealization probes")
    prefs = [ev(f"probe_{p.name}", {**stock, **p.knobs}, power=False,
                sub="probe", trace_knobs=p.trace_knobs)
             for p in CODESIGN_PROBES]
    pres = fan(prefs, [p.name for p in CODESIGN_PROBES], "probes", log=log)

    class _R:                    # probe_report wants AccelSimRunResult-alikes
        def __init__(self, r):
            self.stats = {"cycles": r.get("cycles")}
            self.knobs = r.get("effective") or {}
            self.trace_knobs = r.get("effective") or {}
            self.ok = bool(r.get("ok"))
            self.stderr_tail = r.get("note", "")
            self.stdout_tail = r.get("stdout_tail", "")
            self.counters = {}
            self.stall_mix = r.get("stall_mix") or {}

    shim = {"baseline": _R(b)}
    for r in pres:
        shim[str(r["point"]).split("/", 1)[-1][len("probe_"):]] = _R(r)
    rep = probe_report(shim, baseline="baseline", probes=CODESIGN_PROBES)
    log(format_probe_report(rep))

    log("\n=== 3. attribute reservation failures")
    attrib = attribute_failures(b["outdir"])
    log(format_attribution(attrib))
    return {"baseline": b, "probe_report": rep, "attribution": attrib,
            "traces": traces, "base_tile": base_t}


SYSTEM = """You are a computer architect. You have a cycle-accurate GPU
simulator behind a small set of tools and nothing else -- no shell, no files,
no web. Every number you will ever see comes back from those tools.

Work from the measurements, not from what is usually true of GPUs. When you
pick a knob, be able to name the counter that licensed it. When a result
surprises you, check whether the simulator actually applied the knob before
you build a theory on top of it.

Your simulation budget is small and the cluster runs points in parallel, so
send batches, not single points."""


def brief(budget: int, cfg: LoopConfig) -> str:
    """The task. Deliberately states the objective and withholds the answer.

    It does not name a knob, a tile, or the limiter -- those are in
    ``get_measurements``, which the agent has to read.  What it does state is
    the thing the deterministic chain also knows and would be unfair to hide:
    that a single SM count decides nothing here.
    """
    return f"""Design a better {cfg.m}x{cfg.n}x{cfg.k} GEMM design point for an
A100-class GPU: one CUTLASS tile shape, plus one set of microarchitectural
knob changes. Both halves are yours to choose, and either may be left at stock.

Start by calling get_measurements, knob_vocabulary and tile_vocabulary. Those
are free and they are the whole of what is known before you run anything.

You then have {budget} simulations. Spend them, then call propose_design once
with the tile, the knobs, and a rationale that names the measurement each part
of the choice rests on.

Two things about this machine that will cost you if you ignore them:

  - The SM-count axis is deterministically chaotic: the same design measured
    at 104, 106, 108, 110 and 112 SMs swings by 5-8% per step. Your points all
    run at {cfg.n_sm} SMs. A 0.5% difference between two points is noise from
    an axis you are not varying, not a result. After you propose, your design
    is re-measured across that whole band and any gain whose sign flips inside
    it is recorded as not quotable -- so a fragile design scores zero, not a
    little less than a robust one.

  - Making a structure bigger is not automatically a win, and the interesting
    question is never "does more cache help". Area is finite. A knob is worth
    proposing when a counter says that specific structure is where the machine
    is stalling.
"""


def agent_codesign(cfg: LoopConfig, *, budget: int = 16, out_root: str = "",
                   model: str = "claude-sonnet-4-6", timeout_seconds: int = 3600,
                   log=print) -> dict:
    """Measure, hand the measurements to an agent, then verify what it says.

    The return value has the same shape as ``codesign_loop``'s journal -- a
    list of steps, each carrying the measurement that licensed it -- so one
    reporter prints both arms and the comparison is not mediated by prose.
    """
    from chia.simulators.accelsim_codesign import area_ledger, format_ledger
    from chia.simulators.accelsim_sandbox import sandboxed_llm

    out_root = out_root or cfg.out_root
    os.makedirs(out_root, exist_ok=True)
    journal: dict = {"config": cfg.__dict__.copy(), "arm": "agent",
                     "budget": budget, "steps": []}

    def step(name, **kw):
        journal["steps"].append({"step": name, **kw})
        return kw

    t0 = time.time()
    m = measure(cfg, out_root, log=log)
    b, traces, base_t = m["baseline"], m["traces"], m["base_tile"]
    ref_cycles = float(b["cycles"])
    step("baseline", cycles=ref_cycles, record=b)
    step("probes", report=m["probe_report"])
    # Flattened exactly as ``codesign_loop`` flattens it, for two reasons:
    # one reporter has to read both arms, and ``Attribution.counts`` is keyed
    # by (unit, rw, reason) tuples, which json refuses outright -- a journal
    # written with ``__dict__`` would raise at the end of a 40-minute run.
    at = m["attribution"]
    step("attribution", limiter=at.limiter, share=at.limiter_share,
         write_share=at.write_share, total=at.total, note=at.note,
         dominant_unit=at.dominant_unit)

    log(f"\n=== 4. agent proposes (budget {budget} simulations)")
    tool = CoDesignTool(
        "codesign", accelsim_root=cfg.accelsim_root, cuda_path=cfg.cuda_path,
        trace_root=cfg.trace_root, out_root=out_root,
        m=cfg.m, n=cfg.n, k=cfg.k, n_sm=cfg.n_sm, budget=budget,
        baseline=b, probe_report=m["probe_report"], attribution=m["attribution"])
    sessions = os.path.join(out_root, "claude_sessions")
    os.makedirs(sessions, exist_ok=True)
    llm = sandboxed_llm([tool], model=model, system_message=SYSTEM,
                        timeout_seconds=timeout_seconds,
                        projects_cwd=sessions, log_dir=sessions)
    log(f"  tool up at {tool.hostname}:{tool.port}")
    res = llm.prompt(brief(budget, cfg), tools=[tool])
    log(f"  agent finished: success={res.success} rc={res.returncode}")

    led = tool.read_ledger()
    step("agent", spent=led.get("spent"), batches=led.get("batches"),
         design=led.get("design"), report=res.result, success=res.success)
    design = led.get("design")
    if not design:
        log("  agent proposed no design; nothing to band.")
        journal["design"] = None
        journal["wall_s"] = time.time() - t0
        return journal
    wk, wspec = design["tile"], design["knobs"]
    log(f"  design: {wk} + {wspec}  ({led.get('spent')} sims)")
    log(f"  rationale: {str(design.get('rationale'))[:400]}")

    # -- verification: the same three arms the deterministic chain uses ----
    log(f"\n=== 5. sign-stability band for {wk}/{wspec} over SM {cfg.band_sm}")
    wknobs, why = parse_knobs(wspec)
    if why:
        raise SystemExit(f"recorded design has an illegal knob string: {why}")
    stock = dict(ADAPTIVE_OFF)
    same_tile = (wk == base_t)
    brefs, blabels = [], []
    for nsm in cfg.band_sm:
        arms = [("base", base_t, stock)]
        if not same_tile:
            arms.append(("ref", wk, stock))
        arms.append(("win", wk, wknobs))
        for tag, kern, kn in arms:
            pt = f"band_{tag}_sm{nsm}"
            brefs.append(evaluate_point.chia_remote(
                cfg.accelsim_root, cfg.cuda_path, kern, f"{kern}/{pt}",
                traces[kern], os.path.join(out_root, kern, "band", pt),
                {**kn, **sm_knobs(nsm)}, True, None, False, 10800))
            blabels.append(f"{tag}_sm{nsm}")
    bres = fan(brefs, blabels, "band", log=log)
    bg = {str(r["point"]).split("/", 1)[-1]: r["cycles"]
          for r in bres if r.get("ok")}
    if same_tile:
        for nsm in cfg.band_sm:
            if f"band_base_sm{nsm}" in bg:
                bg[f"band_ref_sm{nsm}"] = bg[f"band_base_sm{nsm}"]

    def _band(num, den) -> Band:
        return sign_stable_band({
            nsm: pct_gain(bg[f"band_{num}_sm{nsm}"], bg[f"band_{den}_sm{nsm}"])
            for nsm in cfg.band_sm
            if f"band_{num}_sm{nsm}" in bg and f"band_{den}_sm{nsm}" in bg})

    band, band_tile, band_knob = _band("win", "base"), _band("ref", "base"), \
        _band("win", "ref")
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

    # -- area, on the same terms as the other arm -------------------------
    design_dir = os.path.join(out_root, wk, "agent", wspec.replace("/", "_"))
    ledger = None
    if os.path.isdir(design_dir):
        try:
            lg = area_ledger(os.path.join(out_root, base_t, "pts", "baseline"),
                             design_dir, n_sm=cfg.n_sm, n_l2=160)
            log("\n=== 6. area ledger")
            log(format_ledger(lg))
            ledger = lg.__dict__
            step("area", ledger=ledger)
        except Exception as exc:
            log(f"  area ledger unavailable: {type(exc).__name__}: {exc}")

    journal["design"] = {
        "arm": "agent", "tile": wk, "knobs": wspec,
        "rationale": design.get("rationale"),
        "sims_spent": led.get("spent"), "budget": budget,
        "verdict": band.verdict, "lo": band.lo, "hi": band.hi,
        "median": band.median, "stable": band.stable,
        # A design the band cannot resolve is not a failed run.  It is a
        # measured statement that the gain is smaller than the SM axis, and
        # recording it as such is the only way the comparison stays honest
        # when the agent loses.
        "quotable": bool(band.stable and band.verdict == "WIN"),
        "area": ledger,
    }
    journal["wall_s"] = time.time() - t0
    return journal
