"""Hand the raw performance counters to an agent and see what it diagnoses.

Step 4 of the co-design loop asks "what is the bottleneck?".  Today that
question is answered by :func:`accelsim_codesign.attribute_failures`, which
sums six reservation-failure reasons over two cache units and two access
directions.  It is deterministic, it is fast, and it is right about the
thing it looks at -- but it reads 3,216 of the 15,449 columns the simulator
actually writes, and it can only ever return one of three answers because
``LIMITER_KNOB`` has three entries.

This module is the other end of that trade: the whole counter file, served
to an agent as a queryable surface, with no hypothesis baked in.  It exists
to answer a scoping question -- *can an agent reach the hardcoded answer,
and does it see anything the hardcoded answer structurally cannot?*

The first run said yes to both: it found the same structure, fixed the
denominator (93.9% "of failures" is not a share of anything that predicts a
speedup; 28.4% of SM-cycles is), and named a zero-SRAM policy change the
menu did not list.  That run also exposed two faults of its own, which is
why this module no longer looks the way it did.  Its numbers had to be
checked by hand, so ``submit_diagnosis`` now re-reads every counter quoted
and rejects a diagnosis whose arithmetic does not survive.  And it was
open-loop, diagnosing without ever testing, so a small simulation budget is
available through :meth:`CounterTool.test_hypothesis`.

The counter names carry their own structure: ``L2_bank_007_GLOBAL_ACC_W_MISS``
is unit ``L2_bank``, instance 7, counter ``GLOBAL_ACC_W_MISS``.  Collapsing
the instance index gives 158 *families*, which is small enough to show an
agent in full and wide enough that the answer is not already implied by the
question.  Per-instance values stay available through :meth:`counter_spread`,
because "the mean is fine and one bank is on fire" is a real bottleneck that
no aggregate can express.
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import os
import re
import statistics
from dataclasses import dataclass, field

from chia.base.tools.ChiaTool import ChiaTool

__all__ = ["CounterSnapshot", "load_counters", "CounterTool",
           "COUNTER_BRIEF", "COUNTER_SYSTEM", "agent_diagnose"]

#: Instance indices are zero-padded numbers embedded in the column name.
#: Collapsing them is what turns 15,449 columns into 158 families.
_INSTANCE = re.compile(r"_\d+")


@dataclass
class CounterSnapshot:
    """The final cumulative row of ``perf_counter.csv.gz``, indexed two ways."""

    path: str
    cycles: float = 0.0
    raw: dict[str, float] = field(default_factory=dict)
    #: family -> [instance values], in column order
    families: dict[str, list[float]] = field(default_factory=dict)

    def total(self, family: str) -> float:
        return float(sum(self.families.get(family, ())))

    def match(self, pattern: str) -> list[str]:
        """Families matching *pattern* (case-insensitive regex, "" = all)."""
        if not pattern:
            return list(self.families)
        rx = re.compile(pattern, re.I)
        return [f for f in self.families if rx.search(f)]


def load_counters(path: str) -> CounterSnapshot:
    """Read the *last* row -- the counters are cumulative, so it is the total.

    ``perf_counter.csv.gz`` is written once per kernel-launch sample; every
    row restates the running total, so the final row is the whole run and
    the intermediate rows are only interesting for a time series nobody has
    asked for yet.
    """
    snap = CounterSnapshot(path=path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        last = None
        for row in rdr:
            if row:
                last = row
    if last is None:
        return snap
    for name, cell in zip(header, last):
        try:
            val = float(cell)
        except (TypeError, ValueError):
            continue
        snap.raw[name] = val
        snap.families.setdefault(_INSTANCE.sub("_N", name), []).append(val)
    # The simulator's cycle count is not a counter column; it is the first
    # field of every row, whatever that column happens to be named.
    try:
        snap.cycles = float(last[0])
    except (TypeError, ValueError, IndexError):
        snap.cycles = 0.0
    return snap


# How many simulations may be in flight at once.  The Ray cluster declares
# an ``accelsim`` resource; this is the ceiling the agent is told about and
# the width each dispatch wave is chunked to.
MAX_PARALLEL = 100


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}".rstrip("0").rstrip(".")


class CounterTool(ChiaTool):
    """Reads over one counter file, an optional simulation budget, one answer.

    The reads are free: every one is a lookup against a dict built once at
    construction, so the agent can be as exhaustive as it likes.  Only
    :meth:`test_hypothesis` costs anything, and it is offered at all only
    when a budget and a trace were supplied -- diagnosis-only remains a
    valid configuration, and is what the scoping run used.  ``budget=0``
    means unlimited -- the agent decides how many points are worth running,
    and up to ``MAX_PARALLEL`` of them run at once.
    """

    def __init__(self, name: str, *, counter_csv: str, out_dir: str,
                 cycles: float | None = None, knob_menu: str = "",
                 accelsim_root: str = "", cuda_path: str = "",
                 kernelslist: str = "", budget: int = 0,
                 knob_families: dict | None = None,
                 baseline_dir: str = "", n_sm: int = 108, n_l2: int = 32,
                 task_options=None):
        super().__init__(name, task_options=task_options)
        self._snap = load_counters(counter_csv)
        if cycles:
            self._snap.cycles = float(cycles)
        self._knob_menu = knob_menu
        self._out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._calls: list[dict] = []
        self._diagnosis: dict | None = None
        # -- simulation, if a budget was granted --------------------------
        self._a, self._c, self._kl = accelsim_root, cuda_path, kernelslist
        self._budget = int(budget)
        self._spent = 0
        self._families = dict(knob_families or {})
        self._runs: list[dict] = []
        # -- area ----------------------------------------------------------
        # The gate needs two numbers the agent cannot supply itself: what the
        # baseline costs, and what each design it tests costs. Both come from
        # McPAT run post-hoc against the XML each run leaves behind -- no
        # re-simulation, ~1s a point -- and McPAT is used unmodified. What
        # changed on 2026-09-24 is that ``area_report`` now fills the
        # template's geometry fields from each run's own ``gpgpusim.config``;
        # before that every design priced identically and a gate would have
        # passed everything.
        self._base_dir = baseline_dir or os.path.dirname(counter_csv)
        self._n_sm, self._n_l2 = int(n_sm), int(n_l2)
        self._area_cache: dict[str, object] = {}
        self.mcp.add_tool(self.counter_overview, name=f"{name}_counter_overview")
        self.mcp.add_tool(self.counter_search, name=f"{name}_counter_search")
        self.mcp.add_tool(self.counter_spread, name=f"{name}_counter_spread")
        self.mcp.add_tool(self.knob_menu, name=f"{name}_knob_menu")
        # A budget of 0 means *unlimited*, not disabled: simulation is
        # offered whenever a trace exists to run.  Diagnosis-only remains a
        # valid configuration -- it is what you get with no trace at all.
        if accelsim_root and kernelslist:
            self.mcp.add_tool(self.test_hypotheses,
                              name=f"{name}_test_hypotheses")
            self.mcp.add_tool(self.test_hypothesis,
                              name=f"{name}_test_hypothesis")
        self.mcp.add_tool(self.submit_diagnosis, name=f"{name}_submit_diagnosis")
        super().__post_init__()

    # -- persistence ------------------------------------------------------
    # Tool bodies execute inside this tool's Ray actor.  Anything kept on
    # ``self`` is invisible to the driver, which holds a different object
    # and would report an empty session however much the agent did.
    def _write(self, stem: str, obj) -> None:
        try:
            with open(os.path.join(self._out, stem), "w") as fh:
                json.dump(obj, fh, indent=1, default=str)
        except OSError:
            pass

    def _log(self, tool: str, **kw) -> None:
        self._calls.append({"tool": tool, **kw})
        self._write("counter_calls.json", self._calls)

    # -- reads ------------------------------------------------------------
    async def counter_overview(self, top: int = 60) -> str:
        """Every counter family in the run, largest first. Start here.

        A *family* is one counter summed over all its hardware instances:
        ``L1D_N_GLOBAL_ACC_W_MISS_QUEUE_FULL`` is the write-side miss-queue-full
        count over all 108 L1 caches.  The ``xN`` column is how many instances
        were summed, so a family with ``x160`` is per-L2-bank and one with
        ``x1`` is a whole-GPU total.

        Args:
            top: how many families to show, by magnitude. Pass a large number
                to see all of them; there are fewer than 200.
        """
        return await asyncio.to_thread(self._overview, int(top))

    def _overview(self, top: int) -> str:
        s = self._snap
        rows = sorted(((s.total(f), f, len(v)) for f, v in s.families.items()),
                      key=lambda r: -abs(r[0]))
        self._log("counter_overview", top=top, families=len(rows))
        out = [f"run: {os.path.basename(os.path.dirname(s.path))}",
               f"total cycles: {_fmt(s.cycles)}",
               f"counter columns: {len(s.raw):,}   families: {len(s.families)}",
               "", f"{'total':>18}  {'inst':>5}  family"]
        for val, fam, n in rows[:max(1, top)]:
            out.append(f"{_fmt(val):>18}  {('x%d' % n):>5}  {fam}")
        if len(rows) > top:
            out.append(f"... {len(rows) - top} more families; "
                       f"raise `top` or use counter_search.")
        return "\n".join(out)

    async def counter_search(self, pattern: str, top: int = 40) -> str:
        """Counter families whose name matches a regular expression.

        Case-insensitive, matched anywhere in the family name. Useful when
        you have a hypothesis: `stall`, `queue_full`, `bkconflict`,
        `GLOBAL_ACC_W`, `port_busy`, `dram`, `icnt`.

        Args:
            pattern: Python regular expression, e.g. `L2.*RESERVATION`.
            top: maximum rows to return.
        """
        return await asyncio.to_thread(self._search, str(pattern), int(top))

    def _search(self, pattern: str, top: int) -> str:
        s = self._snap
        try:
            names = s.match(pattern)
        except re.error as exc:
            return f"bad regular expression {pattern!r}: {exc}"
        self._log("counter_search", pattern=pattern, hits=len(names))
        if not names:
            return (f"no family matches {pattern!r}. "
                    f"counter_overview(top=200) lists every name.")
        rows = sorted(((s.total(f), f, len(s.families[f])) for f in names),
                      key=lambda r: -abs(r[0]))
        out = [f"{len(names)} families match {pattern!r}",
               f"{'total':>18}  {'inst':>5}  family"]
        for val, fam, n in rows[:max(1, top)]:
            out.append(f"{_fmt(val):>18}  {('x%d' % n):>5}  {fam}")
        if len(rows) > top:
            out.append(f"... {len(rows) - top} more matches")
        return "\n".join(out)

    async def counter_spread(self, family: str) -> str:
        """How one family is distributed across its hardware instances.

        An aggregate hides imbalance: 108 L1 caches summing to a large number
        could be 108 busy caches or four saturated ones. This returns min,
        median, max and the busiest instances, so "hot bank" and "uniformly
        loaded" are distinguishable.

        Args:
            family: exact family name from counter_overview / counter_search,
                with the instance index written as `_N`.
        """
        return await asyncio.to_thread(self._spread, str(family))

    def _spread(self, family: str) -> str:
        s = self._snap
        vals = s.families.get(family)
        if vals is None:
            near = [f for f in s.families if family.lower() in f.lower()][:8]
            self._log("counter_spread", family=family, found=False)
            return (f"no family named {family!r}."
                    + (f" did you mean: {', '.join(near)}" if near else ""))
        self._log("counter_spread", family=family, instances=len(vals))
        tot = float(sum(vals))
        out = [f"{family}  ({len(vals)} instances, total {_fmt(tot)})"]
        if len(vals) == 1:
            out.append(f"  single value: {_fmt(vals[0])}")
            return "\n".join(out)
        srt = sorted(vals)
        out += [f"  min {_fmt(srt[0])}   median {_fmt(statistics.median(srt))}"
                f"   max {_fmt(srt[-1])}   mean {_fmt(tot / len(vals))}",
                f"  max/median ratio: "
                f"{(srt[-1] / statistics.median(srt)) if statistics.median(srt) else float('inf'):.2f}"]
        idx = sorted(range(len(vals)), key=lambda i: -vals[i])[:6]
        out.append("  busiest instances: "
                   + ", ".join(f"#{i}={_fmt(vals[i])}" for i in idx))
        return "\n".join(out)

    async def knob_menu(self) -> str:
        """The hardware changes this co-design loop is currently able to make.

        Read this *after* you have formed a diagnosis, not before. If your
        diagnosis points at something this menu cannot buy, say so in
        submit_diagnosis -- that gap is a result, not a mistake.
        """
        return await asyncio.to_thread(lambda: self._knob_menu or "(not supplied)")

    async def test_hypothesis(self, knob_family: str, mult: str = "",
                              why: str = "") -> str:
        """Run one simulation of a proposed change and see what it does.

        Prefer `test_hypotheses` -- it runs several at once for the same
        wall-clock cost as this one. Use this only when a single point is
        genuinely all you want next.

        Args:
            knob_family: a family name from `knob_menu`.
            mult: the family's argument, if it takes one (e.g. `4`). Leave
                empty for the family's own default.
            why: one line on what you expect to happen. Recorded next to the
                result, so a wrong prediction is on the record too.
        """
        return await asyncio.to_thread(
            self._test_batch,
            [{"knob_family": str(knob_family), "mult": str(mult),
              "why": str(why)}])

    async def test_hypotheses(self, points: list) -> str:
        """Run SEVERAL simulations at once. They execute in parallel.

        This is the difference between a diagnosis and a guess, and the
        parallelism is the difference between testing one idea and testing
        the whole shape of an idea. Up to 100 simulations run concurrently,
        so a batch of twenty costs about the same wall-clock time as a batch
        of one. There is no reason to serialise.

        Spend them on what your evidence points at. Good batches look like:
        a sweep of one family over several multipliers, to find where it
        saturates; a set of rival mechanisms, to find which one is real; a
        change plus its own control, to prove the change is what moved the
        counter. A batch of unrelated guesses is still a guess.

        Each result reports cycles against the baseline *and* which counter
        families moved most -- so a change that helps for the wrong reason,
        or that relocates a stall rather than removing it, is visible rather
        than inferred.

        Args:
            points: a list of `{"knob_family": ..., "mult": ..., "why": ...}`.
                `mult` is the family's argument (e.g. `"4"`), empty for its
                default. `why` is one line on what you expect -- recorded
                next to the result, so a wrong prediction is on the record.
        """
        return await asyncio.to_thread(self._test_batch, points)

    # -- simulation -------------------------------------------------------
    def _remaining(self) -> float:
        """Simulations left. ``inf`` when no budget was imposed."""
        if self._budget <= 0:
            return float("inf")
        return max(0, self._budget - self._spent)

    def _budget_line(self) -> str:
        if self._budget <= 0:
            return f"budget: unlimited ({self._spent} simulations run so far)"
        return f"budget: {self._spent}/{self._budget} spent"

    def _build(self, family: str, mult: str):
        """Resolve one point to knobs, or to the reason it is not legal."""
        if family not in self._families:
            return None, (f"{family!r} is not a knob family. Available: "
                          f"{', '.join(sorted(self._families))}")
        builder = self._families[family][1]
        try:
            if not mult:
                knobs = builder()
            else:
                arg = int(mult) if mult.lstrip("-").isdigit() else mult
                knobs = builder(arg)
        except Exception as exc:                      # the builder's own veto
            return None, f"{family}({mult or 'default'}) refused: {exc}"
        # Every dl1 edit is discarded unless adaptive cache config is off --
        # the knob would silently do nothing and the agent would conclude,
        # reasonably and wrongly, that the structure does not matter.
        from chia.simulators.accelsim_codesign import ADAPTIVE_OFF
        return {**ADAPTIVE_OFF, **knobs}, ""

    def _normalise(self, points) -> list[dict]:
        """Accept a list of dicts, of strings, or a single dict."""
        if isinstance(points, dict):
            points = [points]
        if isinstance(points, str):                   # a JSON string, or one name
            try:
                points = json.loads(points)
            except Exception:
                points = [points]
            if isinstance(points, dict):
                points = [points]
        out = []
        for p in list(points or []):
            if isinstance(p, str):
                out.append({"knob_family": p, "mult": "", "why": ""})
            elif isinstance(p, dict):
                out.append({
                    "knob_family": str(p.get("knob_family")
                                       or p.get("family") or ""),
                    "mult": str(p.get("mult") if p.get("mult") is not None
                                else ""),
                    "why": str(p.get("why") or "")})
        return out

    def _diff(self, outdir: str) -> list:
        """What moved, not just how fast."""
        newcsv = os.path.join(outdir, "perf_counter.csv.gz")
        if not os.path.exists(newcsv):
            return []
        try:
            after = load_counters(newcsv)
        except Exception:
            return []
        moved = []
        for fam in self._snap.families:
            b, a = self._snap.total(fam), after.total(fam)
            if b or a:
                d = a - b
                if abs(d) > max(1.0, abs(b) * 0.02):
                    moved.append((abs(d), fam, b, a))
        moved.sort(reverse=True)
        return moved

    def _price(self, outdir: str, bound: str = ""):
        """McPAT area for one completed run, or None if it cannot be priced.

        Unpriceable is a real outcome, not a zero: CACTI refuses some
        geometries GPGPU-Sim runs happily (a 4 KB texture cache, for one), and
        a design whose area is unknown must not be compared against a baseline
        whose area is known.
        """
        key = f"{outdir}|{bound or ''}"
        if key in self._area_cache:
            return self._area_cache[key]
        rep = None
        try:
            from chia.simulators.accelsim_area import area_report
            from chia.simulators.accelsim import read_config_knobs
            xml = self._area_xml(outdir)
            if xml:
                # Pass the design explicitly. ``area_report`` otherwise looks
                # for a config beside the XML, and the XML may now be the
                # stock template from the accel-sim tree -- whose neighbour is
                # the *tested-cfg's* config, not this run's. Measured: that
                # silently priced a GV100 for every design and returned one
                # identical number, which is precisely the blindness the knob
                # injection exists to remove.
                knobs = read_config_knobs(
                    os.path.join(outdir, "gpgpusim.config"))
                got = area_report(xml, self._n_sm, self._n_l2,
                                  knobs=knobs or None, bound=bound)
                rep = got if got.ok else None
        except Exception:
            rep = None
        self._area_cache[key] = rep
        return rep

    def _area_xml(self, outdir: str) -> str:
        """The XML to price ``outdir`` from, or "" if there is none.

        ``accelwattch_sass_sim.xml`` is not written by the simulator -- it is
        copied verbatim out of the tested-cfg directory, and is byte-identical
        for every GPU model shipped with accel-sim (one md5 covers TITANX,
        GV100, QV100 and RTX2060 alike). It therefore carries no information
        about the run at all; the design lives entirely in ``gpgpusim.config``,
        which :func:`area_report` reads separately.

        That matters because a run launched without ``-power_simulation_enabled``
        has no copy of it, and requiring one made every such run report
        "unpriced" -- a refusal that looks like a CACTI failure but is only a
        missing constant. So fall back to any sibling run's copy, then to the
        stock template itself.
        """
        own = os.path.join(outdir, "accelwattch_sass_sim.xml")
        if os.path.exists(own):
            return own
        for cand in (os.path.join(self._base_dir,
                                  "accelwattch_sass_sim.xml"),
                     getattr(self, "_xml_template", "") or ""):
            if cand and os.path.exists(cand):
                return cand
        sibs = os.path.dirname(os.path.abspath(outdir))
        try:
            for name in sorted(os.listdir(sibs)):
                cand = os.path.join(sibs, name, "accelwattch_sass_sim.xml")
                if os.path.exists(cand):
                    return cand
        except OSError:
            pass
        root = getattr(self, "_a", "") or ""
        if root:
            from chia.simulators.accelsim_power import DEFAULT_XML_ARCH
            cand = os.path.join(
                root, "gpu-simulator/gpgpu-sim/configs/tested-cfgs",
                DEFAULT_XML_ARCH, "accelwattch_sass_sim.xml")
            if os.path.exists(cand):
                return cand
        return ""

    def _credit(self, outdir: str,
                bound: str = "") -> tuple[float, list[str], list[str]]:
        """mm^2 freed by deleting structures this run never touched.

        The only area big enough to fund anything comes from deleting a whole
        unit, and that is honest exactly when the workload never uses it --
        a claim the power report can settle. ``removal_is_justified`` also
        refuses a structure whose counters are suppressed by
        ``-gpgpu_perfect_inst_const_cache``, where zero means unmeasured
        rather than unused.
        """
        rep = self._price(outdir, bound)
        if rep is None:
            return 0.0, [], ["design could not be priced"]
        try:
            from chia.simulators.accelsim_area import (
                read_counters, removal_is_justified, removal_credit, REMOVABLE)
            from chia.simulators.accelsim import read_config_knobs
        except Exception as exc:
            return 0.0, [], [str(exc)[:80]]
        log = os.path.join(outdir, "accelwattch_power_report.log")
        cfg = os.path.join(outdir, "gpgpusim.config")
        if not os.path.exists(log):
            return 0.0, [], ["no power report: cannot certify anything dead"]
        counters = read_counters(log)
        knobs = read_config_knobs(cfg) if os.path.exists(cfg) else {}
        dead, refused = [], []
        for part in REMOVABLE:
            ok, why = removal_is_justified(part, counters, knobs)
            (dead if ok else refused).append(part if ok else why)
        if not dead:
            return 0.0, [], refused
        _per_sm, chip = removal_credit(rep, dead)
        return chip, dead, refused

    def _test_batch(self, points) -> str:
        pts = self._normalise(points)
        if not pts:
            return ("no points given. Pass a list of "
                    "{knob_family, mult, why}.")
        # -- validate everything before spending anything -----------------
        plan, rejected = [], []
        seen = set()
        for p in pts:
            fam, mult = p["knob_family"], p["mult"]
            knobs, err = self._build(fam, mult)
            if err:
                rejected.append(err)
                continue
            label = f"{fam}{mult or ''}".replace("/", "_")
            if label in seen:
                rejected.append(f"{label}: duplicate in this batch, dropped")
                continue
            seen.add(label)
            plan.append({"family": fam, "mult": mult, "why": p["why"],
                         "label": label, "knobs": knobs,
                         "outdir": os.path.join(self._out, "sims", label)})
        # A dispatch with no trace configured cannot be scheduled -- it
        # requests the ``accelsim`` resource and waits for a node that will
        # never appear, which is indistinguishable from a model that stopped
        # responding. Refuse it here, where the reason is still legible.
        if not (self._a and self._kl):
            return ("simulation is not configured for this session (no "
                    "accel-sim root or kernelslist). Diagnose from the "
                    "counters alone and submit.")
        room = self._remaining()
        if room <= 0:
            return (f"budget exhausted ({self._budget} simulations). "
                    f"Submit your diagnosis with what you have.")
        if len(plan) > room:
            dropped = [q["label"] for q in plan[int(room):]]
            plan = plan[:int(room)]
            rejected.append(f"budget allows {int(room)} more; not run: "
                            f"{', '.join(dropped)}")
        if not plan:
            return "nothing to run.\n" + "\n".join(rejected)

        from chia.simulators.accelsim_codesign import evaluate_point
        import ray

        # -- dispatch, in waves no wider than the cluster ------------------
        results: dict[str, object] = {}
        for i in range(0, len(plan), MAX_PARALLEL):
            wave = plan[i:i + MAX_PARALLEL]
            refs = {}
            for q in wave:
                try:
                    refs[q["label"]] = evaluate_point.chia_remote(
                        self._a, self._c, "base", f"base/{q['label']}",
                        self._kl, q["outdir"], q["knobs"], True, None,
                        True, 10800)
                except Exception as exc:
                    results[q["label"]] = exc
            for label, ref in refs.items():
                try:
                    results[label] = ray.get(ref)
                except Exception as exc:
                    results[label] = exc

        # -- score ---------------------------------------------------------
        base = self._snap.cycles or 0.0
        base_area = self._price(self._base_dir)
        rows, warnings, details = [], [], []
        for q in plan:
            self._spent += 1
            r = results.get(q["label"])
            if isinstance(r, Exception) or r is None:
                rec = {"family": q["family"], "mult": q["mult"],
                       "why": q["why"], "error": str(r)}
                self._runs.append(rec)
                rows.append((q["label"], None, None, f"FAILED: {r}", None))
                continue
            cyc = float(r.get("cycles") or 0)
            gain = (base - cyc) / base * 100.0 if (base and cyc) else 0.0
            unapplied = list(r.get("unapplied") or [])
            moved = self._diff(q["outdir"])
            # Price it now, while the run's XML is on disk and the point is
            # still in front of the agent.  Showing cycles without area is
            # what let every previous design be chosen as though silicon
            # were free.
            ar = self._price(q["outdir"])
            d_area = (ar.total_mm2 - base_area.total_mm2
                      if (ar and base_area) else None)
            rec = {"family": q["family"], "mult": q["mult"], "why": q["why"],
                   "cycles": cyc, "gain_pct": gain, "unapplied": unapplied,
                   "knobs": q["knobs"], "outdir": q["outdir"],
                   "area_mm2": (ar.total_mm2 if ar else None),
                   "d_area_mm2": d_area,
                   "moved": [{"family": f, "before": b, "after": a}
                             for _, f, b, a in moved[:12]]}
            self._runs.append(rec)
            note = ""
            if unapplied:
                note = "SEE WARNING"
                warnings.append(
                    f"  {q['label']}: the simulator IGNORED {unapplied} -- "
                    f"this result is NOT the change you asked for.")
            rows.append((q["label"], cyc, gain, note, d_area))
            if moved:
                det = [f"{q['label']}  counters that moved most "
                       f"(before -> after):"]
                for _, f, b, a in moved[:12]:
                    pct = ((a - b) / b * 100.0) if b else float("inf")
                    det.append(f"  {f:<52} {b:>14,.0f} -> {a:>14,.0f}  "
                               + (f"{pct:+.1f}%" if b else "new"))
                details.append("\n".join(det))
        self._write("counter_runs.json", self._runs)

        # -- report ---------------------------------------------------------
        head = f"{len(plan)} simulation(s), run in parallel. " \
               f"Baseline {base:,.0f} cycles"
        head += (f", {base_area.total_mm2:,.0f} mm^2."
                 if base_area else ", area unpriced.")
        out = [head, ""]
        width = max(len(r[0]) for r in rows)
        for label, cyc, gain, note, d_area in sorted(
                rows, key=lambda r: (r[2] is None, -(r[2] or 0))):
            if cyc is None:
                out.append(f"  {label:<{width}}  {note}")
                continue
            # Area is reported beside every result, not only at submission.
            # A knob whose gain is worth having and whose price is not is a
            # thing the agent should learn while it can still change course.
            area = (f"{d_area:+9.1f} mm^2" if d_area is not None
                    else "   unpriced")
            out.append(f"  {label:<{width}}  {cyc:>12,.0f} cy  "
                       f"{gain:+7.2f}%  {area}  {note}")
        out.append("")
        out.append(self._budget_line())
        if warnings:
            out.append("")
            out.append("WARNINGS:")
            out.extend(warnings)
        if rejected:
            out.append("")
            out.append("not run:")
            out.extend("  " + r for r in rejected)
        if details:
            out.append("")
            out.append("\n\n".join(details))
        return "\n".join(out)

    async def submit_diagnosis(self, limiter_family: str, mechanism: str,
                               evidence: str, share_estimate: str,
                               counters_used: list,
                               proposed_change: str, rationale: str,
                               secondary: str = "",
                               wanted_but_undriven: str = "",
                               proposed_family: str = "",
                               proposed_mult: str = "",
                               funded_by: str = "") -> str:
        """Record what is limiting this kernel and what you would change.

        Two registers, deliberately. ``limiter_family`` is a label the rest
        of the co-design loop can act on mechanically; ``mechanism`` is the
        sentence that says what is actually happening, which is the part a
        fixed table can never hold. Give both -- a label with no mechanism
        is what the old hardcoded reader already produced.

        Args:
            limiter_family: one of the families from `knob_menu`, or the
                exact string `none_of_these` if the binding constraint is
                something the loop cannot currently change. Do not pick the
                closest available family to be helpful; `none_of_these` plus
                `wanted_but_undriven` is more useful than a bad fit.
            mechanism: one or two sentences saying what is physically
                happening, not just which structure is full.
            evidence: the counter families and numbers that support it.
            share_estimate: how dominant it is, with the denominator named.
            counters_used: the counters your numbers came from, as a list of
                {"family": "<exact family name>", "value": <number>}. These
                are re-read from the file and checked against what you wrote
                before the diagnosis is accepted, so quote them exactly.
            proposed_change: the hardware change you would make.
            rationale: why that change addresses this limiter.
            secondary: further limiters, couplings, or anything unresolved.
            wanted_but_undriven: if the change you actually want is an option
                the loop does not drive, name it here (an exact option name
                from `knob_menu`'s undriven list if you can). This is a
                result, not a failure -- it tells us what to wire up next.
            proposed_family: the knob family the co-design loop should carry
                forward into its cross-product against every tile, and
                `proposed_mult` its argument. This is the machine-readable
                half of `proposed_change`: give it and your design is built
                and run against the software axis; leave it blank and your
                diagnosis is recorded but nothing is built from it. It need
                not equal `limiter_family` -- the structure that is full and
                the structure worth buying are often not the same one.
            proposed_mult: the argument for `proposed_family` (e.g. `"4"`),
                empty for that family's own default.
            funded_by: what you are giving up to pay for it. Your design is
                priced in mm^2 against the baseline, and a design that costs
                area must say where the area comes from or it is rejected.
                Name the structures you are deleting, comma-separated, from
                those this workload never touches -- run
                `test_hypotheses` on your design first and the reply tells
                you both its price and what is certified dead. Write
                `none` only if your design does not cost area; that is
                checked, not taken on trust.
        """
        return await asyncio.to_thread(
            self._submit, limiter_family, mechanism, evidence, share_estimate,
            counters_used, proposed_change, rationale, secondary,
            wanted_but_undriven, proposed_family, proposed_mult, funded_by)

    def verify_counters(self, claimed) -> dict:
        """Re-read every counter the agent quoted and check its arithmetic.

        The scoping run produced about thirty numbers in free text. They were
        all correct, and nothing in the system knew that: a confident wrong
        diagnosis is textually identical to a confident right one. This makes
        the quoted values falsifiable -- not the reasoning, but at least the
        inputs to it.
        """
        rows, bad = [], 0
        for item in (claimed or []):
            if not isinstance(item, dict):
                rows.append({"family": str(item), "status": "unparseable"})
                bad += 1
                continue
            fam = str(item.get("family", "")).strip()
            try:
                said = float(item.get("value"))
            except (TypeError, ValueError):
                rows.append({"family": fam, "status": "non-numeric value"})
                bad += 1
                continue
            if fam not in self._snap.families:
                rows.append({"family": fam, "claimed": said,
                             "status": "no such family"})
                bad += 1
                continue
            actual = self._snap.total(fam)
            # Exact for counts; a relative tolerance only so a value quoted
            # in millions is not failed on a rounding digit.
            ok = (actual == said) or (
                actual and abs(actual - said) / abs(actual) < 1e-6)
            rows.append({"family": fam, "claimed": said, "actual": actual,
                         "status": "ok" if ok else "MISMATCH"})
            bad += 0 if ok else 1
        return {"checked": len(rows), "bad": bad, "rows": rows,
                "verdict": "verified" if (rows and not bad)
                           else ("unverified" if not rows else "FAILED")}

    def _area_verdict(self, pfamily: str, pmult: str,
                      funded_by: str) -> dict:
        """Price the proposed design and decide whether it is affordable.

        The rule is the one the brief states: a design that costs area must
        name what pays for it, and the payment has to be certified dead
        rather than asserted. Three outcomes, and the middle one matters
        most --

        * ``free``      the design costs nothing or gives area back.
        * ``unpriced``  McPAT could not build the geometry. Recorded, but
                        not quotable: an unknown price is not a price of
                        zero, and the alternative (pricing the bare
                        template) measured a 4 KB texture cache as a
                        -1005 mm^2 credit.
        * ``over``      it costs more than the named sellers free. Rejected,
                        with the ledger, so the agent can pick a bigger
                        seller or a cheaper rung of the same knob.

        Requiring the design to have been simulated first is not bureaucracy:
        the price comes from the XML that run leaves behind, so an untested
        design has no price to check.
        """
        want = f"{pfamily}{pmult or ''}"
        run = next((r for r in self._runs
                    if f"{r.get('family')}{r.get('mult') or ''}" == want), None)
        if run is None:
            return {"verdict": "untested", "detail": want}
        buy = run.get("d_area_mm2")
        if buy is None:
            # ``_test_batch`` fills this in, but a record can reach here
            # without it -- reloaded from a journal, or produced before
            # pricing existed. Both run directories are still on disk, so
            # price them now rather than reporting "unpriced", which is
            # reserved for a geometry CACTI genuinely cannot build.
            # Asymmetric on purpose, and both ends are reported. A purchase
            # is charged at the cheapest defensible mapping; a sale is
            # credited only with what survives the completest one. The
            # opposite pairing would let a design pass on the convention
            # rather than on the measurement.
            here = self._price(run["outdir"], "optimistic")
            base = self._price(self._base_dir, "optimistic")
            if here is not None and base is not None:
                buy = here.total_mm2 - base.total_mm2
            hc = self._price(run["outdir"], "conservative")
            bc = self._price(self._base_dir, "conservative")
            if hc is not None and bc is not None:
                run["d_area_conservative"] = hc.total_mm2 - bc.total_mm2
        if buy is None:
            return {"verdict": "unpriced", "buy_mm2": None}
        if buy <= 0:
            return {"verdict": "free", "buy_mm2": buy, "credit_mm2": 0.0,
                    "net_mm2": buy, "sold": [], "dead": []}
        credit, dead, refused = self._credit(run["outdir"],
                                                    "conservative")
        named = [w.strip() for w in str(funded_by or "").split(",")
                 if w.strip() and w.strip().lower() != "none"]
        # Only credit what the agent actually named AND the counters certify.
        # Crediting everything dead would let a design pay with structures
        # its author never considered, which is a ledger that balances
        # itself and therefore checks nothing.
        sold = [d for d in dead
                if any(n.lower() in d.lower() or d.lower() in n.lower()
                       for n in named)]
        got = 0.0
        if sold:
            rep = self._price(run["outdir"])
            try:
                from chia.simulators.accelsim_area import removal_credit
                _per, got = removal_credit(rep, sold)
            except Exception:
                got = 0.0
        return {"verdict": "ok" if buy - got <= 0 else "over",
                "buy_mm2": buy, "credit_mm2": got, "net_mm2": buy - got,
                "sold": sold, "dead": dead, "refused": refused[:4],
                "named": named}

    def _submit(self, family, mechanism, evidence, share, counters_used,
                change, rationale, secondary, undriven,
                pfamily="", pmult="", funded_by="") -> str:
        ver = self.verify_counters(counters_used)
        # Resolve the machine-readable proposal now, so the loop downstream
        # gets knobs it can run rather than a family name it has to trust.
        pknobs, perr = ({}, "")
        if pfamily:
            pknobs, perr = self._build(str(pfamily), str(pmult))
            if perr:
                return ("REJECTED -- `proposed_family` is not runnable: "
                        + perr + "\n(Fix it, or leave it blank and submit "
                        "the diagnosis on its own.)")
        # -- the area gate -------------------------------------------------
        area = {"verdict": "not_applicable"}
        if pfamily:
            area = self._area_verdict(str(pfamily), str(pmult), funded_by)
            if area["verdict"] == "untested":
                return ("REJECTED -- you proposed "
                        f"{area['detail']} but never ran it, so it has no "
                        "price. Call `test_hypotheses` on your own design "
                        "first; the reply gives you its area and what this "
                        "workload leaves dead, then resubmit with "
                        "`funded_by`.")
            if area["verdict"] == "over":
                dead = ", ".join(area["dead"]) or "nothing"
                return ("REJECTED on area. Your design costs "
                        f"{area['buy_mm2']:+,.1f} mm^2; what you named "
                        f"({', '.join(area['named']) or 'nothing'}) frees "
                        f"{area['credit_mm2']:,.1f}; net "
                        f"{area['net_mm2']:+,.1f} mm^2, and the budget is "
                        "constant area.\n"
                        f"Certified dead in your own run, so sellable: {dead}."
                        "\nEither name more of it in `funded_by`, or test a "
                        "cheaper multiplier of the same family and propose "
                        "that. Area scales faster than the gain does.")
        self._diagnosis = {
            "area": area,
            "funded_by": funded_by,
            "quotable": area["verdict"] in ("ok", "free", "not_applicable"),
            "limiter_family": family, "mechanism": mechanism,
            "evidence": evidence, "share_estimate": share,
            "counters_used": counters_used, "verification": ver,
            "proposed_change": change, "rationale": rationale,
            "secondary": secondary, "wanted_but_undriven": undriven,
            "proposed_family": pfamily, "proposed_mult": pmult,
            "proposed_knobs": pknobs or {},
            "reads": len(self._calls), "sims_spent": self._spent,
            "runs": list(self._runs),
        }
        self._write("diagnosis.json", self._diagnosis)
        self._log("submit_diagnosis", limiter_family=family,
                  verification=ver["verdict"])
        if ver["verdict"] == "FAILED":
            bad = [r for r in ver["rows"] if r.get("status") != "ok"]
            self._diagnosis = None
            self._write("diagnosis.json", {})
            return ("REJECTED -- the counters you quoted do not match the "
                    "file. Nothing was recorded; fix these and resubmit:\n"
                    + "\n".join(
                        f"  {r.get('family')}: you said "
                        f"{r.get('claimed')}, file says {r.get('actual')} "
                        f"({r.get('status')})" for r in bad[:8]))
        if ver["verdict"] == "unverified":
            return ("REJECTED -- `counters_used` was empty. Quote the "
                    "counters your numbers came from so they can be checked, "
                    "then resubmit.")
        carried = (f"design carried forward: {pfamily}({pmult or 'default'})"
                   if pfamily else
                   "no design carried forward (proposed_family was blank)")
        if area["verdict"] in ("ok", "free"):
            carried += (f" at {area['net_mm2']:+,.1f} mm^2 net"
                        + (f", funded by {', '.join(area['sold'])}"
                           if area.get("sold") else ""))
        elif area["verdict"] == "unpriced":
            carried += (" -- but CACTI could not build its geometry, so it "
                        "is recorded unpriced and NOT quotable as an "
                        "iso-area result")
        return (f"recorded: limiter_family={family!r}, "
                f"{ver['checked']} counters verified against the file, "
                f"after {len(self._calls)} reads and {self._spent} "
                f"simulations. {carried}. You may stop now.")

    # -- driver side ------------------------------------------------------
    def read_diagnosis(self) -> dict:
        """Read the diagnosis back through disk, not through the actor."""
        p = os.path.join(self._out, "diagnosis.json")
        try:
            with open(p) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def read_calls(self) -> list:
        p = os.path.join(self._out, "counter_calls.json")
        try:
            with open(p) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return []


def default_knob_menu(covered: dict | None = None, *,
                      families: dict | None = None,
                      show_uncovered: int = 30) -> str:
    """Build the agent-facing menu from the simulator's own option registrar.

    The previous menu was three families and a sentence asserting that
    everything else -- coalescing, write policy, scheduling -- was
    inexpressible.  Two of those were untrue, and an agent believed the
    assertion and trimmed a correct diagnosis to fit it.  Deriving the menu
    means the loop can still be narrow, but it can no longer be wrong about
    how narrow it is.
    """
    from chia.simulators.accelsim_knobs import coverage, format_menu
    from chia.simulators.accelsim_codesign import EXTENDED_LIMITER_KNOB
    fams = families if families is not None else EXTENDED_LIMITER_KNOB
    if covered is None:
        covered = {}
        try:
            from chia.simulators.accelsim_dse import CODESIGN_PROBES
            for pr in CODESIGN_PROBES:
                for src in (getattr(pr, "knobs", None) or {},
                            getattr(pr, "trace_knobs", None) or {}):
                    for k in src:
                        covered.setdefault(k, pr.name)
        except Exception:
            pass
        for fam, (_lbl, builder) in fams.items():
            try:
                for k in builder():
                    covered.setdefault(k, fam)
            except Exception:
                continue
    cov = coverage(covered)
    groups = {}
    for fam, (_lbl, builder) in sorted(fams.items()):
        try:
            groups[fam] = ", ".join(sorted(builder()))
        except Exception:
            groups[fam] = "(defaults unavailable)"
    return format_menu(cov, groups=groups, show_uncovered=show_uncovered)


COUNTER_SYSTEM = (
    "You are a GPU microarchitect reading raw simulator performance counters "
    "from an Accel-Sim run of a single GEMM kernel. Your job is diagnosis: "
    "name the structure that is actually limiting this kernel and show the "
    "counters that prove it. You have no ability to run simulations and no "
    "access to the filesystem; the counter tools are your only instrument. "
    "Be skeptical of large raw numbers -- a counter that is large because it "
    "is incremented every cycle on every unit is not evidence of anything. "
    "Always form a ratio against a denominator you can defend."
)


def COUNTER_BRIEF(cycles: float, budget: int = 0, *, can_simulate: bool = True) -> str:
    if not can_simulate:
        sim = ""
    elif budget > 0:
        sim = """
You may run simulations (`test_hypotheses`), up to %d of them. They run in
parallel -- %d at a time -- so a batch of ten costs about the same wall-clock
time as a batch of one. Serialising them wastes the budget's real currency,
which is your own turns, not the cluster's.
""" % (budget, MAX_PARALLEL)
    else:
        sim = """
You may run as many simulations as you think the question needs
(`test_hypotheses`). There is no budget. They run in parallel -- up to %d at
a time -- so a batch of twenty costs about the same wall-clock time as a
batch of one, and there is no reason to run them one at a time.

Use that width. Sweep a family across several multipliers in one call to
find where it saturates. Test rival mechanisms side by side rather than in
sequence. Run a change together with its own control, so you can attribute
the movement instead of assuming it. A run reports cycles *and* which
counters moved, so you can see whether a change helped for the reason you
predicted or merely relocated the stall -- predict before you run, because
your `why` is recorded next to the result either way.

What this does not buy you is breadth for its own sake. A batch of twenty
unrelated guesses is still twenty guesses; a batch of four that isolates one
variable is an experiment.
""" % MAX_PARALLEL
    return f"""Diagnose the bottleneck in this kernel from its performance counters.

The run is a tiled single-precision GEMM on a simulated A100-class GPU:
108 SMs, 160 L2 banks, {_fmt(cycles)} total cycles. The kernel is already
correct and already tiled; the question is purely what the hardware is
spending those cycles waiting on.

Work in this order:

1. `counter_overview(top=200)` -- see the entire counter vocabulary at once.
2. Form a hypothesis and test it with `counter_search`. Think about what the
   *denominator* is: a stall count means nothing without the number of
   attempts, and a per-cycle counter means nothing without the cycle count.
3. Use `counter_spread` on anything that matters, to tell a saturated
   structure apart from a hot instance.
4. Only then call `knob_menu`. It lists what this loop drives *and* what
   the simulator accepts but the loop does not yet drive. If what you want
   is in the second list, name it in `wanted_but_undriven` -- do not pick
   the nearest available family to be helpful.
{sim}
5. Call `submit_diagnosis`. Every counter you quote in `counters_used` is
   re-read from the file and checked before the diagnosis is accepted, so
   quote exact family names and exact values. Give `proposed_family` and
   `proposed_mult` as well: that is the design the co-design loop will
   actually build and run against every software tile. A diagnosis without
   it is recorded but changes nothing.

   The design is priced in mm^2 and the budget is constant area, so run
   your own proposal through `test_hypotheses` before you submit it -- an
   untested design has no price and is refused. Every result you get back
   carries its area beside its cycles. If yours costs area, name what pays
   for it in `funded_by`. The only payment that counts is deleting a
   structure this workload never touches, certified from the run's own
   counters rather than asserted; the rejection message tells you which
   structures qualify. Two things worth knowing before you choose a
   multiplier: area usually scales faster than the gain does, so the
   expensive rung of a good knob can fail where a cheaper rung passes; and
   there is not much to sell, so spend it on the knob that pays most.

Two things are explicitly worth your attention. If you find that the real
limiter is something the knob menu cannot buy, say so plainly -- that is a
more useful result than a diagnosis trimmed to fit the available knobs. And
if several mechanisms are entangled such that fixing one would just move the
stall, say that too, in `secondary`.
"""


def agent_diagnose(counter_csv: str, out_dir: str, *, knob_menu: str = "",
                   cycles: float | None = None,
                   model: str = "claude-sonnet-4-6",
                   accelsim_root: str = "", cuda_path: str = "",
                   kernelslist: str = "", budget: int = 0,
                   knob_families: dict | None = None,
                   baseline_dir: str = "", n_sm: int = 108, n_l2: int = 32,
                   timeout_seconds: int = 3600, log=print) -> dict:
    """Serve one counter file to a sandboxed agent and record its diagnosis."""
    from chia.simulators.accelsim_sandbox import sandboxed_llm
    from chia.simulators.accelsim_codesign import EXTENDED_LIMITER_KNOB

    os.makedirs(out_dir, exist_ok=True)
    fams = knob_families if knob_families is not None else EXTENDED_LIMITER_KNOB
    menu = knob_menu or default_knob_menu(families=fams)
    tool = CounterTool("counters", counter_csv=counter_csv, out_dir=out_dir,
                       baseline_dir=(baseline_dir
                                     or os.path.dirname(counter_csv)),
                       n_sm=n_sm, n_l2=n_l2,
                       cycles=cycles, knob_menu=menu,
                       accelsim_root=accelsim_root, cuda_path=cuda_path,
                       kernelslist=kernelslist, budget=budget,
                       knob_families=fams)
    log(f"  counter tool up at {tool.hostname}:{tool.port} "
        f"({len(tool._snap.families)} families, {len(tool._snap.raw):,} columns, "
        f"{len(fams)} knob families, "
        f"budget {'unlimited' if budget <= 0 else budget})")
    sessions = os.path.join(out_dir, "claude_sessions")
    os.makedirs(sessions, exist_ok=True)
    llm = sandboxed_llm([tool], model=model, system_message=COUNTER_SYSTEM,
                        timeout_seconds=timeout_seconds,
                        projects_cwd=sessions, log_dir=sessions)
    can_sim = bool(accelsim_root and kernelslist)
    res = llm.prompt(COUNTER_BRIEF(tool._snap.cycles, budget,
                                   can_simulate=can_sim), tools=[tool])
    log(f"  agent finished: success={res.success} rc={res.returncode}")
    runs = []
    try:
        with open(os.path.join(out_dir, "counter_runs.json")) as fh:
            runs = json.load(fh)
    except (OSError, ValueError):
        pass
    return {"diagnosis": tool.read_diagnosis(), "calls": tool.read_calls(),
            "runs": runs, "report": res.result, "success": res.success,
            "cycles": tool._snap.cycles, "menu": menu}
