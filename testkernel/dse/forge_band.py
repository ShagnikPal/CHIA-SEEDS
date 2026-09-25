#!/usr/bin/env python3
"""Simulate what the forge produced -- the last link in the agentic axis.

``KernelForgeTool`` stops at "accepted": correct, and it really did the
arithmetic. Whether an accepted kernel is *better* is a different measurement
and belongs to the simulator, on the same SM band as every other claim in this
project, because a single SM count resolves nothing on an axis that swings
+-5..8% per 2-SM step.

Two arms, stock hardware throughout, so this is a pure software comparison:
``ref`` is the naive one-thread-per-output kernel the brief describes, ``forge``
is the tiled kernel that came back through the gate. Both traces are the ones
the gate itself produced -- nothing is re-traced or hand-selected here.
"""
from __future__ import annotations

import json
import os
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray

from chia.simulators.accelsim_codesign import (
    ADAPTIVE_OFF,
    evaluate_point,
    sign_stable_band,
    sm_knobs,
)

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/out_forge")
BAND = [104, 106, 108, 110, 112]

ARMS = [("ref", f"{OUT}/reference/traces/traces/kernelslist.g"),
        ("forge", f"{OUT}/selftest/tiled_32/traces/traces/kernelslist.g")]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for tag, path in ARMS:
        if not os.path.exists(path):
            sys.exit(f"missing trace for {tag}: {path}")

    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    t0 = time.time()

    jobs = []
    for tag, kli in ARMS:
        for n in BAND:
            point = f"forgeband_{tag}_sm{n}"
            jobs.append((tag, n, evaluate_point.chia_remote(
                ACCELSIM, CUDA, tag, point, kli,
                os.path.join(OUT, "band", point),
                {**ADAPTIVE_OFF, **sm_knobs(n)}, True, None, False, 10800)))

    print(f"dispatched {len(jobs)} points")
    idx = {id(r): i for i, (_, _, r) in enumerate(jobs)}
    pending = [j[2] for j in jobs]
    res: dict[int, dict] = {}
    done = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for r in ready:
            v = ray.get(r)
            i = idx[id(r)]
            res[i] = v
            done += 1
            tag, n, _ = jobs[i]
            mark = "cache" if v.get("cached") else f"{v.get('wall_s', 0):.0f}s"
            bad = "" if v.get("ok") else f"  !! {str(v.get('note', ''))[:70]}"
            print(f"[{done:>2}/{len(jobs)}] {tag:<6}sm{n:<5}"
                  f"{v.get('cycles') or '-':>10}  {mark:>7}{bad}")

    cyc = {(jobs[i][0], jobs[i][1]): v["cycles"]
           for i, v in res.items() if v.get("ok")}
    pw = {(jobs[i][0], jobs[i][1]): v.get("power_w")
          for i, v in res.items() if v.get("ok")}

    b = sign_stable_band({
        n: (cyc[("ref", n)] - cyc[("forge", n)]) / cyc[("ref", n)] * 100.0
        for n in BAND if ("ref", n) in cyc and ("forge", n) in cyc})

    print(f"\n{'sm':<6}{'naive':>12}{'forge':>12}{'gain%':>9}{'W naive':>10}{'W forge':>10}")
    for n in BAND:
        row = f"{n:<6}"
        for t in ("ref", "forge"):
            c = cyc.get((t, n))
            row += f"{c:>12.0f}" if c else f"{'-':>12}"
        v = b.points.get(n)
        row += f"{v:>+8.2f}%" if v is not None else f"{'-':>9}"
        for t in ("ref", "forge"):
            p = pw.get((t, n))
            row += f"{p:>10.1f}" if p else f"{'-':>10}"
        print(row)
    print(f"\nforge vs naive: {b.verdict} [{b.lo:+.2f}%, {b.hi:+.2f}%]  "
          f"stable={b.stable}  -- {b.note}")

    path = os.path.join(OUT, "forge_band.json")
    with open(path, "w") as fh:
        json.dump({"band_sm": BAND,
                   "cycles": {f"{t}_sm{n}": c for (t, n), c in cyc.items()},
                   "power_w": {f"{t}_sm{n}": p for (t, n), p in pw.items()},
                   "band": b.__dict__}, fh, indent=1, default=str)
    print(f"\n{time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
