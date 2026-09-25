#!/usr/bin/env python3
"""Rank the forge's own kernels against each other, in cycles.

The gate stops at "correct, and it did the arithmetic". The agent then ranked
its five accepted kernels by *warp instructions issued*, because that is the
number ``submit_kernel`` hands back, and picked the 256x128 tile. That metric
has a blind spot the agent named itself: a 256x128 tile at N=512 is a (4,2)
grid -- 8 CTAs on a 108-SM part, 100 SMs idle -- while the 128x128 tile is
(4,4) = 16 CTAs for 3.9% more instructions. Instruction count and occupancy
pull in opposite directions here and only the simulator arbitrates.

Naive is deliberately *not* an arm. It deadlocks GPGPU-Sim at sm108 and swings
1.87x across the band; it is a usable gate and a useless baseline. Every
comparison below is candidate-vs-candidate on stock hardware, three SM counts,
which doubles as a test of whether the SM axis moves these kernels at all.
"""
from __future__ import annotations

import glob
import json
import os
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
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/out_forge512")
BAND = [int(x) for x in os.environ.get("FORGE_BAND", "104,108,112").split(",")]


def arms() -> list[tuple[str, str, dict]]:
    found = []
    for d in sorted(glob.glob(os.path.join(OUT, "candidates", "*/"))):
        sub = os.path.join(d, "submission.json")
        kli = os.path.join(d, "traces", "traces", "kernelslist.g")
        if not (os.path.exists(sub) and os.path.exists(kli)):
            continue
        rec = json.load(open(sub))
        if rec.get("accepted"):
            found.append((os.path.basename(d.rstrip("/")), kli, rec))
    return found


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    A = arms()
    if not A:
        sys.exit(f"no accepted candidates under {OUT}/candidates")
    print(f"{len(A)} accepted candidates x {len(BAND)} SM counts")

    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True, log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {"PYTHONPATH": CHIA_ROOT}})
    t0 = time.time()

    jobs = []
    for tag, kli, _ in A:
        for n in BAND:
            point = f"forgerank_{tag}_sm{n}"
            jobs.append((tag, n, evaluate_point.chia_remote(
                ACCELSIM, CUDA, tag, point, kli,
                os.path.join(OUT, "rank", point),
                {**ADAPTIVE_OFF, **sm_knobs(n)}, True, None, False, 10800)))

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
            print(f"[{done:>2}/{len(jobs)}] {tag:<38}sm{n:<5}"
                  f"{v.get('cycles') or '-':>10}  {mark:>7}{bad}")

    cyc = {(jobs[i][0], jobs[i][1]): v["cycles"] for i, v in res.items() if v.get("ok")}
    pw = {(jobs[i][0], jobs[i][1]): v.get("power_w") for i, v in res.items() if v.get("ok")}
    rec = {tag: r for tag, _, r in A}

    hdr = f"{'candidate':<38}{'CTAs':>6}{'insts':>10}{'LDG':>8}"
    for n in BAND:
        hdr += f"{'sm' + str(n):>10}"
    hdr += f"{'W@' + str(BAND[len(BAND) // 2]):>9}"
    print("\n" + hdr)
    order = sorted(A, key=lambda a: cyc.get((a[0], BAND[len(BAND) // 2]), 1 << 60))
    for tag, _, r in order:
        ctas = CTAS.get(tag, 0)
        row = (f"{tag:<38}{ctas:>6}{r['total_warp_insts']:>10}"
               f"{r['global_loads']:>8}")
        for n in BAND:
            c = cyc.get((tag, n))
            row += f"{c:>10.0f}" if c else f"{'-':>10}"
        p = pw.get((tag, BAND[len(BAND) // 2]))
        row += f"{p:>9.1f}" if p else f"{'-':>9}"
        print(row)

    mid = BAND[len(BAND) // 2]
    ranked = [(t, cyc[(t, mid)]) for t, _, _ in A if (t, mid) in cyc]
    ranked.sort(key=lambda x: x[1])
    if len(ranked) >= 2:
        best, worst = ranked[0], ranked[-1]
        print(f"\nfastest: {best[0]} @ {best[1]:.0f} cycles")
        print(f"slowest: {worst[0]} @ {worst[1]:.0f} cycles "
              f"({(worst[1] - best[1]) / best[1] * 100:+.1f}%)")
        by_inst = min(A, key=lambda a: a[2]["total_warp_insts"])[0]
        print(f"agent's pick by instruction count: {by_inst}"
              f"{'  -- same' if by_inst == best[0] else '  -- DIFFERENT from fastest'}")
    for tag, _, _ in A:
        cs = [cyc.get((tag, n)) for n in BAND]
        if all(cs):
            spread = (max(cs) - min(cs)) / min(cs) * 100
            print(f"  {tag:<38} SM-axis spread {spread:5.2f}%")

    path = os.path.join(OUT, "forge_rank.json")
    with open(path, "w") as fh:
        json.dump({"band_sm": BAND,
                   "cycles": {f"{t}_sm{n}": c for (t, n), c in cyc.items()},
                   "power_w": {f"{t}_sm{n}": p for (t, n), p in pw.items()},
                   "ctas": CTAS,
                   "static": {t: {k: rec[t][k] for k in
                                  ("total_warp_insts", "global_loads", "shared_loads")}
                              for t in rec}}, fh, indent=1, default=str)
    print(f"\n{time.time() - t0:.0f}s -> {path}")


CTAS = {"tile128x128x8_tm8tn8_vec4": 16,
        "tile256x128x16_sr_bk16": 8,
        "tile256x128x8_doublebuffer": 8,
        "tile256x128x8_ptr_strength_reduced": 8,
        "tile256x128x8_tm16tn8_vec4": 8}

if __name__ == "__main__":
    main()
