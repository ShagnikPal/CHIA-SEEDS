#!/usr/bin/env /usr/bin/python3.10
"""Run the hardware grid for one kernel and print the numbers.

Kept separate from walk.py so the grid can be re-run against any trace without
re-deriving the probe verdict. The dimensions are passed in rather than read
back from probe.json, because that file stores `ranked` as reprs.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_dse import (
    build_grid,
    cells_from_records,
    format_grid,
    pareto_front,
)

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
N_SLOTS = 48


@ray.remote(num_cpus=1)
def cell(kernel: str, point: str, kernelslist: str, outdir: str, knobs: dict,
         trace_knobs: dict | None = None) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_power import run_with_power

    t0 = time.time()
    tk = dict(trace_knobs or {})
    try:
        r, v, s = run_with_power(
            AccelSimNode(ACCELSIM, CUDA), kernelslist, outdir=outdir,
            knob_overrides=knobs, trace_knob_overrides=tk, sync_xml=True,
            collect_counters=False, per_kernel=False,
        )
    except Exception as exc:
        return {"kernel": kernel, "point": point, "ok": False,
                "note": f"{type(exc).__name__}: {exc}"[:160],
                "wall_s": time.time() - t0}
    st = r.stats
    ok = bool(r.ok) and bool(st.get("cycles")) and bool(v.avg_power_w)
    return {
        "kernel": kernel, "point": point, "ok": ok,
        "cycles": st.get("cycles"), "occupancy": st.get("occupancy"),
        "issued_ctas": st.get("issued_ctas"),
        "cta_per_core": st.get("cta_per_core"),
        "power_w": v.avg_power_w, "components": v.components,
        "inert_power": v.inert, "power_consistent": v.consistent,
        "joules": s.joules, "seconds": s.seconds, "edp": s.edp,
        "effective": {**{k: r.knobs.get(k) for k in knobs},
                      **{k: r.trace_knobs.get(k) for k in tk}},
        "wall_s": time.time() - t0,
        "note": "" if ok else (r.stderr_tail or "no cycles/power")[:160],
    }


def main() -> None:
    kernel = sys.argv[1] if len(sys.argv) > 1 else "base"
    dims = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["n_sm", "clock"])
    kl = f"{ROOT}/traces/{kernel}/traces/kernelslist.g"
    assert os.path.exists(kl), f"no trace for {kernel}"

    grid = build_grid(dims)
    print(f"kernel={kernel}  dims={dims}  points={len(grid)}")
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             runtime_env={"env_vars":
                          {"PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})

    refs = [cell.remote(kernel, g.name, kl,
                        f"{ROOT}/out/{kernel}/grid/{g.name.replace('/', '_')}",
                        g.knobs, g.trace_knobs) for g in grid]
    results, pending, done = [], list(refs), 0
    t0 = time.time()
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for rf in ready:
            r = ray.get(rf)
            results.append(r)
            done += 1
            cyc = f"{r['cycles']:.0f}" if r.get("cycles") else "-"
            pw = f"{r['power_w']:.2f}" if r.get("power_w") else "-"
            bad = "" if r["ok"] else f"  FAILED {r.get('note', '')[:50]}"
            print(f"  [{done:>2}/{len(grid)}] {r['point']:<24} cycles={cyc:>9} "
                  f"P={pw:>7}W [{r['wall_s']:.0f}s]{bad}", flush=True)
    print(f"grid done in {time.time() - t0:.0f}s")

    os.makedirs(f"{ROOT}/out/{kernel}", exist_ok=True)
    json.dump({"dims": dims, "cells": results},
              open(f"{ROOT}/out/{kernel}/grid.json", "w"), indent=1)

    cells = cells_from_records(results)
    print()
    print(format_grid(cells))
    print("\nPARETO FRONT (cycles, energy):")
    for c in pareto_front(cells):
        print(f"  {c.point:<24}{c.cycles:>9.0f} cycles{c.joules * 1e3:>9.4f} mJ")


if __name__ == "__main__":
    main()
