#!/usr/bin/env python3
"""Run the whole design space exploration for one kernel, end to end.

    run_dse.py                       explore the built-in CUTLASS tile
    run_dse.py mykernel.cu           put your kernel through the gate first
    run_dse.py mykernel.cu --k 256   at a different reduction depth
    run_dse.py --report out_dir      re-print a finished run's tables

Stages, in order:

    0   baseline trace     captured if it is not already there
    0b  kernel gate        compile, correctness, work, trace, simulate
    1   baseline           cycles, power, energy, occupancy
    2   probes             push each resource to a limit, rank the headroom
    3   attribution        which stall reason dominates
    4   counter reading    the raw counters behind that limiter
    5   proposals          hardware points and tiles worth trying
    6   build and trace    the proposed tiles
    7   grid               every tile against every hardware point
    8   composition        whether two wins still win together
    9   SM band            re-measured across a range of SM counts
    10  area               what the design costs and what pays for it

Everything the loop measures is printed as it happens, and again at the end as
tables.  The same numbers are written to ``datapoints.json`` and
``datapoints.csv`` in the output directory.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray

from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_codesign import (
    ADAPTIVE_OFF, BASELINE_TILE, LoopConfig, build_and_trace, codesign_loop,
    sm_knobs, trace_dir_name, trace_name,
)
from chia.simulators.accelsim_sandbox import KernelForgeTool

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT",
                          os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
CUTLASS = os.environ.get("CUTLASS_ROOT", os.path.join(PROJECT, "cutlass"))
FORGE = os.path.join(PROJECT, "testkernel", "forge")
N_SLOTS = int(os.environ.get("DSE_SLOTS", "100"))


# ---------------------------------------------------------------- printing

def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def table(headers, rows, widths=None) -> None:
    """Print a table, sizing each column to its widest cell."""
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    w = widths or [max([len(h)] + [len(r[i]) for r in rows] or [len(h)])
                   for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(headers))))
    print(flush=True)


def num(x, spec=",.0f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "-"


_REPR_FIELD = re.compile(r"[(,]\s*([A-Za-z_]\w*)=")


def field(obj, name, default=None):
    """Read a field from a live record or from one reloaded off disk.

    A live run hands the reporter real dataclasses.  A saved journal has been
    through ``json.dump(default=str)``, which turns each of them into its
    ``repr`` -- so the same reporter has to read ``Headroom(name='2x_sm',
    cycles=89059.0, ...)`` as well.  Parsing the repr is ugly but it is what
    keeps ``--report`` on a finished run identical to the live output.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    if not isinstance(obj, str):
        return getattr(obj, name, default)
    hits = [(m.group(1), m.end(), m.start()) for m in _REPR_FIELD.finditer(obj)]
    for i, (key, val_at, _) in enumerate(hits):
        if key != name:
            continue
        stop = obj.rindex(")") if i + 1 == len(hits) else hits[i + 1][2]
        try:
            return ast.literal_eval(obj[val_at:stop].strip())
        except (ValueError, SyntaxError):
            return obj[val_at:stop].strip()
    return default


# ---------------------------------------------------------------- stage 0/1

def boot() -> None:
    """``accelsim`` slots are CPU simulation; ``accelsim_gpu`` gates tracing.

    Declaring them apart is what stops every simulation queueing behind the
    GPUs, which is the whole economics of replaying one trace across a grid.
    """
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": float(N_SLOTS), "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {"PYTHONPATH": CHIA_ROOT}})


def ensure_baseline_trace(traces: str, k: int, work: str) -> bool:
    """Capture the baseline trace if it is missing.

    Every later stage replays this trace; the loop traces the tiles it
    proposes but not this one, because it is the thing they are compared
    against.  An existing trace is reused untouched.
    """
    base_t = trace_name(BASELINE_TILE)
    dname = trace_dir_name(base_t, 4096, 4096, k)
    kl = os.path.join(traces, dname, "traces", "kernelslist.g")
    if os.path.exists(kl):
        print(f"  baseline trace present: {kl}")
        return True
    print(f"  no baseline trace at {kl}; capturing it (needs a GPU)")
    r = ray.get(build_and_trace.chia_remote(
        ACCELSIM, CUDA, CUTLASS, f"{ROOT}/src/gemm_tpl.cu",
        json.dumps(BASELINE_TILE.__dict__, default=list),
        4096, 4096, k, work, traces, 3600))
    table(["tile", "ok", "stage", "kernels", "wall s", "note"],
          [[r.get("tile"), r.get("ok"), r.get("stage"), r.get("n_kernels"),
            num(r.get("wall_s"), ".0f"), str(r.get("note", ""))[:40]]])
    return bool(r.get("ok"))


def gate_kernel(src: str, traces: str, out: str, k: int) -> str | None:
    """Put a hand-written kernel through the same gate the agent's kernels face.

    Compile against the harness, check the output against a reference produced
    on a real GPU, check it did the arithmetic, trace it, simulate it.  A
    kernel that survives is published into the trace root under a tile name,
    which is the only thing the exploration stage knows how to consume -- so a
    kernel that arrives this way is indistinguishable downstream from one the
    loop synthesised itself.
    """
    m = n = 4096
    seed = int(os.environ.get("FORGE_SEED", "12345"))
    label = os.environ.get("FORGE_LABEL") or os.path.splitext(
        os.path.basename(src))[0]
    label = "".join(c if c.isalnum() or c in "_.-" else "_" for c in label)[:64]

    d = os.path.join(out, "reference")
    os.makedirs(d, exist_ok=True)
    node = AccelSimNode(ACCELSIM, CUDA)

    print(f"  reference: building {FORGE}/reference_mnk.cu")
    build = AccelSimNode.build_cuda.chia_remote_blocking(
        node, sources=[f"{FORGE}/harness_mnk.cu", f"{FORGE}/reference_mnk.cu"],
        binary=os.path.join(d, "harness"), arch_sm="sm_80")
    if not build.ok:
        print(f"  reference build FAILED\n{build.stderr_tail[:400]}")
        return None
    ver = AccelSimNode.verify_kernel.chia_remote_blocking(
        node, binary=build.binary, dump=os.path.join(d, "reference.bin"),
        args=(str(m), str(n), str(k), str(seed)), reference_dump=None, cwd=d)
    if not ver.ok or ver.n_elems != m * n:
        print(f"  reference dump FAILED: {ver.reason}")
        return None

    # One FFMA per (m, n, k) triple, 32 threads to a warp-instruction.  The
    # shape fixes this exactly, so the work gate's reference does not need a
    # trace of a kernel that would issue 33.5M warp-instructions to find out.
    ref_opcodes = {"FFMA": m * n * k // 32}
    table(["reference", "value"],
          [["output elements", f"{ver.n_elems:,}"],
           ["FFMA warp-insts required", f"{ref_opcodes['FFMA']:,}"]])

    tool = KernelForgeTool(
        "run_dse", node, harness_cu=f"{FORGE}/harness_mnk.cu",
        work_dir=os.path.join(out, "candidates"),
        reference_dump=os.path.join(d, "reference.bin"),
        reference_opcodes=ref_opcodes, mnk=(m, n, k), seed=seed,
        simulate=True, sim_sm=108, max_sims=1, max_est_sim_seconds=3600.0,
        accelsim_root=ACCELSIM, cuda_path=CUDA)

    print(f"  submitting {src} as {label!r}", flush=True)
    t0 = time.time()
    rec = tool._submit_kernel_sync(label, open(src).read())
    v = rec.get("verify") or {}
    table(["gate", "result"],
          [["reached stage", rec.get("stage")],
           ["accepted", rec.get("accepted")],
           ["output match", v.get("ok")],
           ["mismatched elements", v.get("n_mismatch")],
           ["max abs err", num(v.get("max_abs_err"), ".3e")],
           ["work ratio", num((rec.get("work_check") or {}).get("ratio"),
                              ".4f")],
           ["warp-insts", num(rec.get("total_warp_insts"))],
           ["global loads", num(rec.get("global_loads"))],
           ["shared loads", num(rec.get("shared_loads"))],
           ["cycles", num(rec.get("cycles"))],
           ["CTA/SM", rec.get("cta_per_core")],
           ["gate wall s", num(time.time() - t0, ".0f")]])
    if not rec.get("accepted"):
        print(f"  REJECTED at the {rec.get('stage')} stage: "
              f"{rec.get('error')}")
        return None

    tname = f"forge_{label}"
    dst = os.path.join(traces, trace_dir_name(tname, m, n, k))
    if os.path.exists(dst):
        print(f"  {tname} already published; left untouched")
        return tname
    src_traces = os.path.dirname(os.path.abspath(rec.get("kernelslist") or ""))
    if not os.path.exists(os.path.join(src_traces, "kernelslist.g")):
        print(f"  no kernelslist under {src_traces}; not published")
        return None
    os.makedirs(dst, exist_ok=True)
    os.symlink(src_traces, os.path.join(dst, "traces"))
    with open(os.path.join(dst, "shape.json"), "w") as fh:
        json.dump({"m": m, "n": n, "k": k}, fh)
    print(f"  published {tname} -> {src_traces}")
    return tname


# ---------------------------------------------------------------- stage 3

def report(journal: dict) -> list:
    """Re-print every measurement that moved a decision, and return the grid.

    The loop already narrates itself while it runs.  This is the same numbers
    laid out after the fact, in the order they were used, so the journey can be
    read without scrolling back through hours of dispatch logs.
    """
    steps = {s["step"]: s for s in journal.get("steps", [])}
    grid = []

    b = (steps.get("baseline") or {}).get("record") or {}
    if b:
        rule("STEP 1  baseline -- what everything else is measured against")
        table(["metric", "value"],
              [["cycles", num(b.get("cycles"))],
               ["power W", num(b.get("power_w"), ".1f")],
               ["energy mJ", num((b.get("joules") or 0) * 1e3, ".2f")],
               ["occupancy", b.get("occupancy")],
               ["CTA/SM", b.get("cta_per_core")]])

    rep = (steps.get("probes") or {}).get("report") or {}
    if rep:
        rule("STEP 2  probes -- push each resource to a limit, see what moves")
        rows = []
        for r in rep.get("ranked", []):
            rows.append([field(r, "name"), num(field(r, "cycles")),
                         num(field(r, "headroom_pct"), "+.2f"),
                         field(r, "direction", ""),
                         str(field(r, "question", ""))[:44]])
        table(["probe", "cycles", "headroom %", "dir", "question"], rows)
        for key, title in (("inert", "knob never applied (not 'no effect')"),
                           ("no_headroom", "applied, changed nothing"),
                           ("failed", "probe failed to run")):
            if rep.get(key):
                print(f"  {title}: {', '.join(rep[key])}")
        if rep.get("hw_targets"):
            print(f"  -> worth buying: {', '.join(rep['hw_targets'])}")
        print(flush=True)

    a = steps.get("attribution")
    if a:
        rule("STEP 3  attribution -- which stall reason actually dominates")
        table(["metric", "value"],
              [["limiter", a.get("limiter")],
               ["share of failures", num(a.get("share"), ".2f") + "%"],
               ["of those, writes", num(a.get("write_share"), ".2f") + "%"],
               ["total failures", num(a.get("total"))],
               ["note", str(a.get("note", ""))[:60]]])

    ag = steps.get("agent_diagnosis")
    if ag and ag.get("diagnosis"):
        rule("STEP 4  counter reading")
        d = ag["diagnosis"]
        table(["field", "value"],
              [[k, str(v)[:70]] for k, v in d.items()
               if not isinstance(v, (list, dict))])
        if ag.get("runs"):
            table(["knob family", "mult", "cycles", "vs base %", "unapplied",
                   "why the agent tried it"],
                  [[field(r, "family"), field(r, "mult"),
                    num(field(r, "cycles")),
                    num(field(r, "gain_pct"), "+.2f"),
                    ",".join(field(r, "unapplied") or ()) or "-",
                    str(field(r, "why", ""))[:46]]
                   for r in sorted(ag["runs"],
                                   key=lambda r: -(field(r, "gain_pct") or 0))])

    p = steps.get("proposals")
    if p:
        rule("STEP 5  proposals -- what the measurements license trying")
        table(["kind", "name", "because"],
              [["HW", nm, str(why)[:58]] for nm, _, why in p.get("hw", [])]
              + [["SW", nm, str(why)[:58]] for nm, why in p.get("sw", [])])

    bt = (steps.get("build_trace") or {}).get("results")
    if bt:
        rule("STEP 6  build and trace the proposed tiles")
        table(["tile", "ok", "stage", "kernels", "wall s", "note"],
              [[r.get("tile"), r.get("ok"), r.get("stage"), r.get("n_kernels"),
                num(r.get("wall_s"), ".0f"), str(r.get("note", ""))[:30]]
               for r in bt])

    cross = (steps.get("cross") or {}).get("records") or []
    if cross:
        rule("STEP 7  the grid -- every tile against every hardware point")
        ref = (steps.get("baseline") or {}).get("cycles") or 0
        for r in sorted(cross, key=lambda r: r.get("cycles") or 9e18):
            if not r.get("ok"):
                continue
            pt = str(r.get("point", "")).split("/", 1)[-1]
            gain = ((ref - r["cycles"]) / ref * 100.0) if ref else None
            grid.append({
                "tile": r.get("kernel"), "hw_point": pt,
                "cycles": r.get("cycles"), "power_w": r.get("power_w"),
                "energy_mj": (r.get("joules") or 0) * 1e3,
                "edp": r.get("edp"), "occupancy": r.get("occupancy"),
                "cta_per_sm": r.get("cta_per_core"),
                "gain_vs_baseline_pct": gain,
                "unapplied": ",".join(r.get("unapplied") or ()),
            })
        table(["tile", "hw point", "cycles", "W", "mJ", "EDP", "CTA/SM",
               "vs base %"],
              [[g["tile"], g["hw_point"], num(g["cycles"]),
                num(g["power_w"], ".1f"), num(g["energy_mj"], ".2f"),
                num(g["edp"], ".3e"), g["cta_per_sm"],
                num(g["gain_vs_baseline_pct"], "+.2f")] for g in grid])

    comp = (steps.get("composition") or {}).get("verdicts") or []
    if comp:
        rule("STEP 8  composition -- do two wins still win together")
        table(["tile", "verdict", "note"],
              [[t, str(c.get("verdict", "")).upper(),
                str(c.get("note", ""))[:52]] for t, c in comp])

    bd = steps.get("band")
    if bd:
        rule("STEP 9  SM band -- is the result a finding or a lucky SM count")
        sms = sorted(set(list((bd.get("gains") or {}).keys())
                         + list(((bd.get("tile_band") or {}).get("points")
                                 or {}).keys())), key=lambda x: int(x))
        rows = []
        for s in sms:
            rows.append([
                s,
                num(((bd.get("tile_band") or {}).get("points") or {}).get(s),
                    "+.2f"),
                num(((bd.get("knob_band") or {}).get("points") or {}).get(s),
                    "+.2f"),
                num((bd.get("gains") or {}).get(s), "+.2f")])
        table(["SM", "tile %", "knob %", "full %"], rows)
        table(["band", "verdict", "low %", "high %", "stable"],
              [["tile", (bd.get("tile_band") or {}).get("verdict"),
                num((bd.get("tile_band") or {}).get("lo"), "+.2f"),
                num((bd.get("tile_band") or {}).get("hi"), "+.2f"),
                (bd.get("tile_band") or {}).get("stable")],
               ["knob", (bd.get("knob_band") or {}).get("verdict"),
                num((bd.get("knob_band") or {}).get("lo"), "+.2f"),
                num((bd.get("knob_band") or {}).get("hi"), "+.2f"),
                (bd.get("knob_band") or {}).get("stable")],
               ["full", bd.get("verdict"), num(bd.get("lo"), "+.2f"),
                num(bd.get("hi"), "+.2f"), bd.get("stable")]])

    lg = (steps.get("area") or {}).get("ledger")
    if lg:
        rule("STEP 10  area -- what the design costs and what pays for it")
        table(["item", "mm2"],
              [["baseline", num(lg.get("baseline_mm2"), ".2f")],
               ["design", num(lg.get("design_mm2"), ".2f")],
               ["added", num(lg.get("added_mm2"), ".2f")],
               ["unpriced", num(lg.get("unpriced_mm2"), ".2f")],
               ["credit", num(lg.get("credit_mm2"), ".2f")],
               ["net", num(lg.get("net_mm2"), ".2f")]])
        if lg.get("removed"):
            print(f"  removed: {lg['removed']}")
        if lg.get("refused"):
            print(f"  refused: {lg['refused']}")
        print(flush=True)

    return grid


def final(journal: dict, grid: list, out: str,
          journal_path: str = "") -> None:
    rule("FINAL DATAPOINTS")
    d = journal.get("design")
    if not d:
        print("  no design selected; see the steps above for where it stopped")
        return
    sel = d.get("selection") or {}
    table(["field", "value"],
          [["tile", d.get("tile")],
           ["hardware point", d.get("hw_point")],
           ["cycles", num(d.get("cycles"))],
           ["energy mJ", num((d.get("joules") or 0) * 1e3, ".2f")],
           ["EDP", num(d.get("edp"), ".3e")],
           ["vs baseline %", num(d.get("vs_baseline_pct"), "+.2f")],
           ["band low %", num((d.get("band") or {}).get("lo"), "+.2f")],
           ["band high %", num((d.get("band") or {}).get("hi"), "+.2f")],
           ["band verdict", (d.get("band") or {}).get("verdict")],
           ["sign-stable", d.get("quotable")],
           ["net area mm2", num((d.get("area") or {}).get("net_mm2"), ".2f")],
           ["limiter targeted", d.get("limiter")],
           ["selection rule", sel.get("rule")],
           ["best by EDP", sel.get("edp_best")],
           ["fastest in EDP band", sel.get("fastest_in_tie")]])
    print(f"  knobs: {json.dumps(d.get('knobs') or {}, sort_keys=True)}\n")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "datapoints.json"), "w") as fh:
        json.dump({"design": d, "grid": grid}, fh, indent=1, default=str)
    if grid:
        with open(os.path.join(out, "datapoints.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grid[0].keys()))
            w.writeheader()
            w.writerows(grid)
    print(f"  datapoints -> {out}/datapoints.json")
    if grid:
        print(f"  datapoints -> {out}/datapoints.csv")
    print(f"  full journal -> "
          f"{journal_path or os.path.join(out, 'journal.json')}",
          flush=True)


# ---------------------------------------------------------------- driver

def main() -> int:
    args = [a for a in sys.argv[1:]]
    k = 64
    if "--k" in args:
        i = args.index("--k")
        k = int(args[i + 1])
        del args[i:i + 2]
    if "--report" in args:
        # Re-print a finished run's tables without re-measuring anything.
        i = args.index("--report")
        d = args[i + 1]
        jp = d if d.endswith(".json") else os.path.join(d, "journal.json")
        if not os.path.exists(jp):
            print(f"no journal at {jp}")
            return 2
        with open(jp) as fh:
            journal = json.load(fh)
        grid = report(journal)
        final(journal, grid, os.path.dirname(os.path.abspath(jp)), jp)
        return 0

    kernel = args[0] if args else None
    if kernel and not os.path.exists(kernel):
        print(f"no such kernel source: {kernel}")
        return 2

    traces = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")
    tag = (os.path.splitext(os.path.basename(kernel))[0] if kernel
           else "baseline")
    out = os.environ.get("DSE_OUT", f"{ROOT}/out_dse_{tag}_k{k}")
    work = f"{ROOT}/build_dse_{tag}_k{k}"
    os.makedirs(out, exist_ok=True)

    rule(f"DESIGN SPACE EXPLORATION   shape 4096x4096x{k}   "
         f"kernel: {kernel or 'built-in CUTLASS tile'}")
    table(["setting", "value"],
          [["Accel-Sim", ACCELSIM], ["CUDA", CUDA], ["CUTLASS", CUTLASS],
           ["traces", traces], ["output", out],
           ["parallel sim slots", N_SLOTS]])
    boot()

    rule("STEP 0  baseline trace")
    if not ensure_baseline_trace(traces, k, work):
        print("  cannot continue without a baseline to compare against")
        return 1

    forge = ()
    if kernel:
        rule("STEP 0b  kernel gate")
        tname = gate_kernel(kernel, traces, out, k)
        if not tname:
            print("  the kernel did not pass the gate; nothing to explore")
            return 1
        forge = (tname,)

    env = os.environ.get("DSE_FORGE_TILES")
    if env is not None:
        forge = tuple(t for t in (x.strip() for x in env.split(","))
                      if t and t != "-")
    # Stage 0b above gates a kernel the *user* supplies, before anything has
    # been measured.  DSE_FORGE instead lets the loop write its own, inside
    # step 4, from the bottleneck it has just attributed -- so the kernel is a
    # response to this run's analysis rather than to a brief fixed in advance.
    in_loop = os.environ.get("DSE_FORGE", "0") != "0"
    rule("STEPS 1-10  exploration")
    print(f"  tiles entering the grid from outside: {list(forge) or 'none'}")
    print(f"  kernel agent inside step 4: "
          f"{'yes, budget ' + os.environ.get('DSE_FORGE_BUDGET', '8') if in_loop else 'no'}\n")

    t0 = time.time()
    journal = codesign_loop(LoopConfig(
        accelsim_root=ACCELSIM, cuda_path=CUDA, cutlass_root=CUTLASS,
        template=f"{ROOT}/src/gemm_tpl.cu", trace_root=traces, out_root=out,
        work_dir=work, m=4096, n=4096, k=k,
        agent_step4=os.environ.get("DSE_AGENT", "1") != "0",
        agent_sim_budget=int(os.environ.get("DSE_AGENT_BUDGET", "0")),
        forge_tiles=forge,
        agent_forge=in_loop, forge_dir=FORGE,
        forge_budget=int(os.environ.get("DSE_FORGE_BUDGET", "8")),
        forge_grade_on=os.environ.get("DSE_FORGE_GRADE", "proposal"),
        area_budget_mm2=float(
            os.environ.get("DSE_AREA_BUDGET", "0"))))
    with open(os.path.join(out, "journal.json"), "w") as fh:
        json.dump(journal, fh, indent=1, default=str)

    grid = report(journal)
    final(journal, grid, out, os.path.join(out, "journal.json"))
    print(f"\n  wall clock {(time.time() - t0) / 3600.0:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
