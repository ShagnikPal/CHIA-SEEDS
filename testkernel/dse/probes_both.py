"""Both directions of every probe, on one trace, in one wave.

Until now the two halves of the gradient lived in different places and only
one of them was ever recorded at this shape. ``CUTLASS_PROBES`` (the upward
half) is what ``codesign_loop`` and the LLM architect both read as their free
measurement; ``STARVE_PROBES`` (the downward half) was only ever driven by
run4k.py at 4096x4096x4096, a shape that was later shown to measure the wrong
thing. So every design decision in this project so far has been made with one
side of the ledger visible.

That asymmetry is not survivable for an iso-area argument. "Enlarging X buys
9.72%" does not license spending area on X; it licenses spending area on X *if
something else can be shrunk for less than 9.72%*, and nothing here had ever
been shrunk and measured at 4096x4096x64. This script runs both halves against
the same baseline, same trace, same wave, and prints them paired by structure
so the exchange rate is a subtraction rather than a guess.

Usage:  probes_both.py [tile] [outdir]
        tile defaults to `base`; any directory under traces4k/ works.
"""
import os, sys, time, json
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray
from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_probes import (
    STARVE_PROBES, format_probe_report, probe_report, run_probes,
)
from chia.simulators.accelsim_dse import CUTLASS_PROBES

ROOT = os.environ.get("DSE_ROOT", ROOT)
TILE = sys.argv[1] if len(sys.argv) > 1 else "base"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/out_probes_both/{TILE}"
KL = f"{ROOT}/traces4k/{TILE}/traces/kernelslist.g"
assert os.path.exists(KL), f"no trace for tile {TILE} at {KL}"
shape = json.load(open(f"{ROOT}/traces4k/{TILE}/shape.json"))

PROBES = tuple(CUTLASS_PROBES) + tuple(STARVE_PROBES)


BY_NAME = {p.name: p for p in PROBES}


PAIRS = [
    ("L2 capacity (sets only)", "l2_2x", "starve_l2_half"),
    ("L2 capacity, 4x span",    "l2_2x", "starve_l2_quarter"),
    ("L1D capacity",            "ideal_l1", "starve_l1_half"),
    ("L1D capacity, 4x span",   "big_l1", "starve_l1_quarter"),
    ("L1 MSHRs",                "many_mshr", "starve_mshr"),
    ("Shared memory",           "big_shmem", "starve_shmem"),
    ("Register file",           "big_regfile", "starve_regfile"),
    ("Register banks",          "more_regbanks", "starve_regbanks"),
    ("Instruction cache",       "big_icache", "starve_icache"),
    ("Constant cache",          "big_const", "starve_const"),
    ("Texture cache",           "big_tex", "starve_tex"),
    ("Sub-cores / issue width", "2x_subcore", "starve_subcore_half"),
    ("Tensor issue rate",       "fast_tensor", "starve_tensor"),
    ("Interconnect width",      "wide_icnt", "starve_icnt"),
    ("Writeback width",         "wide_lsu", "starve_lsu"),
    ("SM count",                "2x_sm", "half_sm"),
]

ray.init(num_cpus=200, num_gpus=8,
         resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
         ignore_reinit_error=True, log_to_driver=False)
node = AccelSimNode(os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework")),
                    os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8"))

print(f"tile={TILE}  shape={shape['m']}x{shape['n']}x{shape['k']}  "
      f"{len(PROBES)} probes + baseline", flush=True)
t0 = time.time()
res = run_probes(node, KL, probes=PROBES, outroot=f"{OUT}/sims")
print(f"\n{len(res)} sims in {time.time()-t0:.0f}s wall\n", flush=True)

rep = probe_report(res, probes=PROBES)
print(format_probe_report(rep))

base_c = res["baseline"].stats["cycles"]


def pct(name):
    """Signed % change vs the probe's own reference, or None."""
    r = res.get(name)
    if r is None or not r.ok or not r.stats.get("cycles"):
        return None
    p = BY_NAME.get(name)
    ref_c = base_c
    if p is not None and p.reference:
        rr = res.get(p.reference)
        if rr is not None and rr.ok and rr.stats.get("cycles"):
            ref_c = rr.stats["cycles"]
    return 100.0 * (ref_c - r.stats["cycles"]) / ref_c


def why(name):
    r = res.get(name)
    if r is None:
        return "not run"
    if not r.ok:
        hay = (r.stderr_tail or "") + (r.stdout_tail or "")
        return ("DEADLOCK" if "deadlock" in hay.lower()
                else "ABORT" if "ssertion" in hay else "FAILED")
    return ""


print(f"\n=== PAIRED GRADIENT, tile {TILE}, baseline {base_c:.0f} cycles ===")
print("  gain% = cycles saved by enlarging.  cost% = cycles LOST by shrinking")
print("  (printed positive).  A structure is free to surrender only if both")
print("  are inside the +-1.00% noise floor.\n")
print(f"  {'structure':<26}{'up probe':<18}{'gain%':>8}   "
      f"{'down probe':<22}{'cost%':>8}")
rows = []
for label, up, dn in PAIRS:
    u, d = pct(up), pct(dn)
    us = f"{u:+.2f}" if u is not None else (why(up) or "--")


    ds = f"{-d:+.2f}" if d is not None else (why(dn) or "--")
    print(f"  {label:<26}{up:<18}{us:>8}   {dn:<22}{ds:>8}")
    rows.append({"structure": label, "up": up, "gain_pct": u,
                 "down": dn, "cost_pct": (-d if d is not None else None),
                 "up_status": why(up), "down_status": why(dn)})

free = [r for r in rows
        if r["gain_pct"] is not None and r["cost_pct"] is not None
        and abs(r["gain_pct"]) < 1.0 and abs(r["cost_pct"]) < 1.0]
print("\n  free to surrender (both directions inside noise):",
      ", ".join(r["structure"] for r in free) or "none")

json.dump({"tile": TILE, "shape": shape, "baseline_cycles": base_c,
           "report": rep, "pairs": rows,
           "cells": {k: {"ok": v.ok,
                         "cycles": (v.stats or {}).get("cycles"),
                         "power_w": (v.stats or {}).get("power_w")}
                     for k, v in res.items()}},
          open(f"{OUT}/probes_both.json", "w"), indent=1, default=str)
print(f"\nwrote {OUT}/probes_both.json")
