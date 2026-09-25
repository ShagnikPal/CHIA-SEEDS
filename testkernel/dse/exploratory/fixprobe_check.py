#!/usr/bin/env /usr/bin/python3.10
"""Does the repaired ``starve_subcore_half`` actually run?

It has never completed: cutting ``-gpgpu_num_sched_per_core`` alone aborts at
``shader.cc:130`` because the ID_OC_* register-set sizes come from
``-gpgpu_pipeline_widths``, and again at ``shader.cc:145`` for every enabled
specialized unit. The repair mirrors the verified 8-sub-core recipe downward.
A probe that aborts is not a measurement, so this runs the real thing once on
the base tile at stock 108 SMs before the probe wave is allowed to quote it.
``starve_tensor`` rides along: its knob moved from the dead tensor pipeline to
the live II lever, so its old inert reading is void too.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_codesign import ADAPTIVE_OFF, evaluate_point, sm_knobs
from chia.simulators.accelsim_probes import PROBE_BY_NAME

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
OUT = f"{ROOT}/out_chia64"
TRACES = f"{ROOT}/traces4k"
KERN = "base"
NAMES = ["starve_subcore_half", "starve_tensor"]


def main() -> None:
    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})
    t0 = time.time()
    kl = f"{TRACES}/{KERN}/traces/kernelslist.g"
    jobs = []
    for name in NAMES:
        p = PROBE_BY_NAME[name]
        ref = evaluate_point.chia_remote(
            ACCELSIM, CUDA, KERN, name, kl,
            f"{OUT}/{KERN}/fixprobe/{name}",
            {**ADAPTIVE_OFF, **sm_knobs(108), **p.knobs}, True,
            dict(p.trace_knobs) or None, False, 10800)
        jobs.append((name, ref))


    base = evaluate_point.chia_remote(
        ACCELSIM, CUDA, KERN, "sm108", kl, f"{OUT}/{KERN}/band/band_base_sm108",
        {**ADAPTIVE_OFF, **sm_knobs(108)}, True, None, False, 10800)
    jobs.append(("BASELINE", base))

    res = {}
    for name, ref in jobs:
        r = ray.get(ref)
        res[name] = r
        mark = "cache" if r.get("cached") else f"{r.get('wall_s', 0):.0f}s"
        print(f"{name:<22}{str(r.get('cycles') or '-'):>10}  ok={r.get('ok')}  "
              f"{mark:>7}  {r.get('note', '')[:90]}", flush=True)
        if r.get("unapplied"):
            print(f"  !! UNAPPLIED: {r['unapplied']}", flush=True)

    b = res["BASELINE"].get("cycles")
    for name in NAMES:
        c = res[name].get("cycles")
        if b and c:
            print(f"{name:<22}{(b - c) / b * 100:+7.2f}%  vs sm108 baseline")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
