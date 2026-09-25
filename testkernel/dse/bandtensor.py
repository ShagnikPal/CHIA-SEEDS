#!/usr/bin/env python3
"""Is the tensor-rate win sign-stable, or is it one lucky SM count?

``spec_axis.py`` and ``subcore_axis.py`` both measured the HMMA rate lever at
**108 SMs only**.  That is a measurement, not a claim.  The SM axis on this
machine is chaotic -- +-5..8% per 2-SM step, enough to swamp any knob -- and
that chaos already killed one iso-area claim of mine.  So the II=4 point has
to clear the same bar every other accepted result cleared: measured against a
*band* of SM counts, and required to keep its sign at every one of them.

Band is the project's standard {104,106,108,110,112}, on both tiles.  The
stock arm of each pair already exists under ``band/`` from the sign-stability
step and comes back from the fingerprint cache, so this pays for 10 new
simulations, not 20.

The point under test is ``-trace_opcode_latency_initiation_spec_op_3 12,4``:
stock latency, II halved 8 -> 4, i.e. **2x the aggregate HMMA rate**.  That is
the only tensor point worth banding.  The 4x point needs eight sub-cores
(a doubled SM core, and confounded by warps-per-scheduler dropping 2 -> 1),
and raising ``num_units`` alone does nothing at all -- usable FUs are capped
at ``-gpgpu_num_sched_per_core``, not set by the unit count.

Reports per-SM deltas plus the swept mean, which is the number that may be
quoted.  A sign flip anywhere in the band means the result does not survive.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray

from chia.simulators.accelsim_codesign import ADAPTIVE_OFF, evaluate_point, sm_knobs

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
OUT = os.environ.get("DSE_OUT", f"{ROOT}/out_chia64")
TRACES = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")

II = "-trace_opcode_latency_initiation_spec_op_3"
BAND = [104, 106, 108, 110, 112]


KERNELS = [("base", "base"), ("t128x256x16_s3", "ref")]


ARMS = [("stock", {}), ("ii4", {II: "12,4"})]


def kl(k: str) -> str:
    return f"{TRACES}/{k}/traces/kernelslist.g"


def outdir(kern: str, tag: str, arm: str, n: int) -> str:
    if arm == "stock":
        return os.path.join(OUT, kern, "band", f"band_{tag}_sm{n}")
    return os.path.join(OUT, kern, "bandtensor", f"{arm}_sm{n}")


def main() -> None:
    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    t0 = time.time()
    jobs = []
    for kern, tag in KERNELS:
        for arm, tk in ARMS:
            for n in BAND:
                ref = evaluate_point.chia_remote(
                    ACCELSIM, CUDA, kern, f"{kern}/sm{n}", kl(kern),
                    outdir(kern, tag, arm, n),
                    {**ADAPTIVE_OFF, **sm_knobs(n)}, True, tk or None,
                    False, 10800)
                jobs.append((kern, arm, n, ref))

    print(f"dispatched {len(jobs)} points", flush=True)
    idx = {id(r): i for i, (_, _, _, r) in enumerate(jobs)}
    pending = [j[3] for j in jobs]
    results: dict[int, dict] = {}
    n_done = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for r in ready:
            res = ray.get(r)
            i = idx[id(r)]
            results[i] = res
            n_done += 1
            kern, arm, n, _ = jobs[i]
            mark = "cache" if res.get("cached") else f"{res.get('wall_s',0):.0f}s"
            bad = "" if res.get("ok") else f"  !! {res.get('note','')[:70]}"
            print(f"[{n_done:>2}/{len(jobs)}] {kern:<16}{arm:<6}sm{n:<5}"
                  f"{res.get('cycles') or '-':>10}  {mark:>7}{bad}", flush=True)

    by: dict[tuple[str, str, int], dict] = {}
    for i, (kern, arm, n, _) in enumerate(jobs):
        by[(kern, arm, n)] = results[i]

    out = []
    print()
    for kern, _ in KERNELS:
        deltas = []
        print(f"=== {kern} : II 12,8 -> 12,4  (2x aggregate HMMA rate) ===")
        print(f"{'SMs':>5}{'stock':>10}{'ii4':>10}{'delta':>9}")
        for n in BAND:
            s, v = by[(kern, "stock", n)], by[(kern, "ii4", n)]
            if not (s.get("ok") and v.get("ok")):
                print(f"{n:>5}{'FAILED':>29}")
                continue
            d = (s["cycles"] - v["cycles"]) / s["cycles"] * 100
            deltas.append(d)
            print(f"{n:>5}{s['cycles']:>10.0f}{v['cycles']:>10.0f}{d:>+8.2f}%")
            out.append({"kernel": kern, "sm": n, "stock_cycles": s["cycles"],
                        "ii4_cycles": v["cycles"], "delta_pct": d,
                        "stock_w": s.get("power_w"), "ii4_w": v.get("power_w")})
        if deltas:
            sign = "SIGN-STABLE" if all(d > 0 for d in deltas) else "SIGN FLIPS"
            print(f"  swept mean {statistics.mean(deltas):+.2f}%  "
                  f"min {min(deltas):+.2f}%  max {max(deltas):+.2f}%  -> {sign}\n")

    path = os.path.join(OUT, "bandtensor.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"{len(out)} pairs in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
