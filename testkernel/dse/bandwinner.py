#!/usr/bin/env python3
"""Band one named design over the SM axis, with the loop's own three arms.

The loop selects on minimum EDP alone. This script does not re-run that
selection: it bands whichever design is named in ``WIN_TILE`` / ``WIN_KNOBS``
below, which is how a point that is *not* the EDP winner can still be measured
-- at K=64, ``t128x128x16_s3 + queue16`` is 1.1% faster in cycles than
``t128x256x16_s3 + queue4`` while costing 0.29% more EDP, and one SM count
cannot resolve a gap that small on an axis that swings +-5..8% per 2-SM step.

Any design banded here needs its own band; a recorded figure belongs to the
design it was measured on and cannot be carried over. Three arms, as the loop
defines them
-- ``base`` (baseline tile, stock knobs: the chip you have), ``ref`` (new tile,
stock knobs: software alone), ``win`` (new tile + queue16) -- so the tile and
knob contributions stay separable. The ``base`` arm reuses the loop's own
``band_base_sm*`` outdirs and comes back from the fingerprint cache.
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
    queue_knobs,
    sign_stable_band,
    sm_knobs,
)

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
OUT = f"{ROOT}/out_chia64"
TRACES = f"{ROOT}/traces4k"

BASE_TILE = "base"
WIN_TILE = "t128x128x16_s3"
WIN_KNOBS = {**ADAPTIVE_OFF, **queue_knobs(16)}
BAND = [104, 106, 108, 110, 112]


ARMS = [("base", BASE_TILE, dict(ADAPTIVE_OFF)),
        ("ref", WIN_TILE, dict(ADAPTIVE_OFF)),
        ("win", WIN_TILE, WIN_KNOBS)]


def kl(k: str) -> str:
    return f"{TRACES}/{k}/traces/kernelslist.g"


def main() -> None:
    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    t0 = time.time()
    jobs = []
    for tag, kern, kn in ARMS:
        for n in BAND:
            point = f"band_{tag}_sm{n}"
            jobs.append((tag, n, evaluate_point.chia_remote(
                ACCELSIM, CUDA, kern, point, kl(kern),
                os.path.join(OUT, kern, "band", point),
                {**kn, **sm_knobs(n)}, True, None, False, 10800)))

    print(f"dispatched {len(jobs)} points", flush=True)
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
            bad = "" if v.get("ok") else f"  !! {v.get('note', '')[:70]}"
            print(f"[{done:>2}/{len(jobs)}] {tag:<6}sm{n:<5}"
                  f"{v.get('cycles') or '-':>10}  {mark:>7}{bad}", flush=True)

    cyc = {(jobs[i][0], jobs[i][1]): v["cycles"]
           for i, v in res.items() if v.get("ok")}
    pw = {(jobs[i][0], jobs[i][1]): v.get("power_w")
          for i, v in res.items() if v.get("ok")}

    def band(num: str, den: str):
        return sign_stable_band({
            n: (cyc[(den, n)] - cyc[(num, n)]) / cyc[(den, n)] * 100.0
            for n in BAND if (num, n) in cyc and (den, n) in cyc})

    full, tile, knob = band("win", "base"), band("ref", "base"), band("win", "ref")
    print(f"\n{'sm':<6}{'base':>10}{'ref':>10}{'win':>10}"
          f"{'tile%':>9}{'knob%':>9}{'FULL%':>9}")
    for n in BAND:
        row = f"{n:<6}" + "".join(
            f"{cyc.get((t, n), float('nan')):>10.0f}" for t in ("base", "ref", "win"))
        for b in (tile, knob, full):
            v = b.points.get(n)
            row += f"{v:>+8.2f}%" if v is not None else f"{'-':>9}"
        print(row)
    for nm, b in (("tile", tile), ("knob", knob), ("FULL", full)):
        print(f"{nm:>5}: {b.verdict:<10} [{b.lo:+.2f}%, {b.hi:+.2f}%]  "
              f"stable={b.stable}  -- {b.note}")

    path = os.path.join(OUT, "bandwinner.json")
    with open(path, "w") as fh:
        json.dump({"tile": WIN_TILE, "hw_point": "queue16",
                   "knobs": WIN_KNOBS, "band_sm": BAND,
                   "cycles": {f"{t}_sm{n}": cyc.get((t, n)) for t, n in cyc},
                   "power_w": {f"{t}_sm{n}": pw.get((t, n)) for t, n in pw},
                   "full": full.__dict__, "tile_band": tile.__dict__,
                   "knob_band": knob.__dict__}, fh, indent=1, default=str)
    print(f"\n{time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
