"""Replay the recorded 43-probe wave through the *library* path.

probes_both.py computed the paired gradient with its own local PAIRS table and
its own pct()/why() helpers. That logic now lives in accelsim_probes as
pair_gradient(), and codesign_loop/agent_codesign sweep CODESIGN_PROBES rather
than the upward-only CUTLASS_PROBES. This re-derives the same table from the
same cached simulations, so any disagreement is a bug in the library version
and not a new measurement.
"""
import json, os, sys, time
sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
import ray
from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_probes import (
    run_probes, probe_report, format_probe_report)
from chia.simulators.accelsim_dse import CODESIGN_PROBES

TILE = "base"
OUT = f"/raid/sp56429/chia_hackathon/testkernel/dse/out_probes_both/{TILE}"
KL = f"/raid/sp56429/chia_hackathon/testkernel/dse/traces4k/{TILE}/traces/kernelslist.g"

ray.init(num_cpus=200, num_gpus=8,
         resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
         ignore_reinit_error=True, log_to_driver=False)
node = AccelSimNode("/raid/sp56429/chia_hackathon/accel-sim-framework",
                    "/usr/local/cuda-12.8")
print(f"{len(CODESIGN_PROBES)} probes + baseline, replaying {OUT}/sims",
      flush=True)
t0 = time.time()
res = run_probes(node, KL, probes=CODESIGN_PROBES, outroot=f"{OUT}/sims")
print(f"{len(res)} sims in {time.time()-t0:.0f}s "
      f"({sum(1 for r in res.values() if getattr(r,'cached',False))} cached)\n",
      flush=True)
rep = probe_report(res, probes=CODESIGN_PROBES)
print(format_probe_report(rep))
json.dump(rep.get("pairs"), open(f"{OUT}/pairs_library.json", "w"),
          indent=1, default=str)
print(f"\nwrote {OUT}/pairs_library.json")
