#!/usr/bin/env python3
"""Scoping run: can an agent do step 4's job from the raw counters?

Step 4 currently answers "what is the bottleneck?" with
``attribute_failures()`` -- a fixed sum over 6 reservation-failure reasons x
2 cache units x 2 directions.  On this baseline it returns:

    limiter = miss_queue, 93.94% of failures, worst in L1D, write path 92.6%

This script hands the *same run's* counter file to a sandboxed agent with no
hypothesis attached and records what it concludes, so the two answers can be
compared on agreement (does it find the same limiter?) and on aperture (does
it see anything the fixed sum structurally cannot?).

No simulations are run.  The agent reads one file.
"""
import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

from chia.simulators.accelsim_counters import (
    agent_diagnose, CounterTool, default_knob_menu)
from chia.simulators.accelsim_codesign import (
    attribute_failures, format_attribution, EXTENDED_LIMITER_KNOB)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_RUN = os.path.join(HERE, "out_probes_both/base/sims/baseline")
OUT = os.path.join(HERE, "out_counter_agent")


ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-11.8")
KL = os.path.join(HERE, "traces4k/base/traces/kernelslist.g")

def control() -> dict:
    """The hardcoded answer, recomputed here so the comparison is not from memory."""
    a = attribute_failures(BASE_RUN)
    return {"limiter": a.limiter, "share": a.limiter_share,
            "write_share": a.write_share, "total": a.total,
            "dominant_unit": a.dominant_unit, "note": a.note,
            "text": format_attribution(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run", "show"], default="run",
                    nargs="?")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--run-dir", default=BASE_RUN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--budget", type=int, default=0,
                    help="simulations the agent may spend to test itself; "
                         "0 (the default) means unlimited -- the agent "
                         "decides how many the question needs")
    ap.add_argument("--kernelslist", default=KL)
    args = ap.parse_args()

    csv_path = os.path.join(args.run_dir, "perf_counter.csv.gz")
    if not os.path.exists(csv_path):
        print(f"no counter file at {csv_path}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    ctl = control()
    print("=== control: the hardcoded attribution ===")
    print(ctl["text"])
    with open(os.path.join(args.out, "control.json"), "w") as fh:
        json.dump(ctl, fh, indent=1)

    if args.mode == "show":
        return 0

    menu = default_knob_menu(families=EXTENDED_LIMITER_KNOB, show_uncovered=30)

    if args.mode == "smoke":


        t = CounterTool("smoke", counter_csv=csv_path,
                        out_dir=os.path.join(args.out, "smoke"),
                        knob_menu=menu, budget=args.budget,
                        accelsim_root=ACCELSIM, cuda_path=CUDA,
                        kernelslist=args.kernelslist,
                        knob_families=EXTENDED_LIMITER_KNOB)
        print("\n--- counter_overview(12)\n" + t._overview(12))
        print("\n--- counter_search('MISS_QUEUE_FULL')\n"
              + t._search("MISS_QUEUE_FULL", 10))
        print("\n--- counter_spread('L1D_N_GLOBAL_ACC_W_MISS_QUEUE_FULL')\n"
              + t._spread("L1D_N_GLOBAL_ACC_W_MISS_QUEUE_FULL"))
        print("\n--- counter_search('stall|port_busy|bkconflict')\n"
              + t._search("stall|port_busy|bkconflict", 14))
        print("\n--- knob_menu (derived from the registrar)\n" + menu)


        print("\n--- guards (no trace supplied, so no simulation is launched)")
        g = CounterTool("guards", counter_csv=csv_path,
                        out_dir=os.path.join(args.out, "smoke"),
                        knob_menu=menu, budget=1,
                        knob_families=EXTENDED_LIMITER_KNOB)
        g._spent = 1
        print("  exhausted : " + g._test_batch([{"knob_family": "write_alloc",
                                                 "mult": "N", "why": "w"}]))
        g._spent = 0
        g._budget = 2
        print("  bad family: " + g._test_batch(["nonsense"]))
        print("  bad arg   : " + g._test_batch(
            [{"knob_family": "write_alloc", "mult": "zzz"}]))
        print("  over budget: " + g._test_batch(
            [{"knob_family": "mem_ports", "mult": str(m)} for m in (2, 4, 8)]
        ).splitlines()[-1])
        bad = g._submit("miss_queue", "m", "e", "10%",
                        [{"family": "L1D_N_GLOBAL_ACC_W_MISS_QUEUE_FULL",
                          "value": 1}], "p", "r", "", "")
        print("  bad number: " + bad.splitlines()[0])
        print("\nOK: all reads answered, menu derived, budget guard live.")
        return 0


    if True:
        import ray
        ray.init(num_cpus=220, num_gpus=8,
                 resources={"accelsim": 100.0, "accelsim_gpu": 8.0},
                 ignore_reinit_error=True, log_to_driver=False)

    print("\n=== agent: open reading of the same file ===", flush=True)
    res = agent_diagnose(csv_path, args.out, knob_menu=menu,
                         model=args.model, budget=args.budget,
                         accelsim_root=ACCELSIM, cuda_path=CUDA,
                         kernelslist=args.kernelslist,
                         knob_families=EXTENDED_LIMITER_KNOB)
    res["control"] = ctl
    with open(os.path.join(args.out, "counter_agent.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)

    d = res.get("diagnosis") or {}
    print("\n=== comparison ===")
    print(f"  control limiter : {ctl['limiter']}  ({ctl['share']:.1f}% of "
          f"{ctl['total']:,} failures, worst in {ctl['dominant_unit']})")
    print(f"  agent limiter   : {d.get('limiter', '(none recorded)')}")
    print(f"  agent share     : {d.get('share_estimate', '-')}")
    print(f"  agent proposes  : {d.get('proposed_change', '-')}")
    print(f"  buyable today   : {d.get('expressible_in_knob_menu')}")
    print(f"  counter reads   : {len(res.get('calls') or [])}")
    runs = res.get("runs") or []
    print(f"  sims spent      : {len(runs)}"
          + (f"/{args.budget}" if args.budget > 0 else " (unlimited)"))
    for r in runs:
        print(f"    {r.get('family')}({r.get('mult') or '-'}): "
              f"{r.get('cycles', 0):,.0f} cy  {r.get('gain_pct', 0):+.2f}%"
              f"   why: {(r.get('why') or '')[:70]}")
    print(f"\nwritten: {os.path.join(args.out, 'counter_agent.json')}")
    return 0 if d else 1


if __name__ == "__main__":
    raise SystemExit(main())
