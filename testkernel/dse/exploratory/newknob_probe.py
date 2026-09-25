"""Measure the knobs the counter audit found the loop was never driving.

An agent reading the baseline counters named write policy, L1 banking, the
DRAM write queue and L1 bypass as the changes it actually wanted.  All four
were refused as "not expressible"; all four are registered options.  This
measures them, because a knob nobody has run is a claim, not a result.

Every dl1 edit carries -gpgpu_adaptive_cache_config 0 and is read against the
`adaptive_off` control, not against baseline: adaptive cache config recomputes
L1D geometry per launch and silently discards a hand-edited -gpgpu_cache:dl1.
"""
import json, os, sys, time
sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
import ray
from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_probes import Probe, run_probes
from chia.simulators.accelsim_codesign import (
    write_alloc_knobs, l1_bank_knobs, dram_queue_knobs,
    l2_subpartition_knobs, gmem_skip_l1_knobs, ADAPTIVE_OFF)

TILE = "base"
OUT = f"/raid/sp56429/chia_hackathon/testkernel/dse/out_probes_both/{TILE}"
KL = f"/raid/sp56429/chia_hackathon/testkernel/dse/traces4k/{TILE}/traces/kernelslist.g"

Q = "Does the change the counter agent asked for actually pay?"

NEW = (
    Probe("adaptive_off", dict(ADAPTIVE_OFF), "control for every dl1 edit",
          "n/a", "n/a"),
    Probe("no_write_alloc", {**ADAPTIVE_OFF, **write_alloc_knobs("N")}, Q,
          "Write-allocate on a write-once stream was pure miss-queue overhead.",
          "Nothing -- this is a hardware policy, not a tiling change.",
          reference="adaptive_off"),
    Probe("no_write_alloc_l2", {**ADAPTIVE_OFF, **write_alloc_knobs("N", l2=True)},
          Q, "Same, applied at both levels.",
          "Nothing.", reference="adaptive_off"),
    Probe("l1_banks4", {**ADAPTIVE_OFF, **l1_bank_knobs(4)}, Q,
          "0.64 L1 bank conflicts per access on a coalesced stream was the "
          "single-bank L1, not the access pattern.",
          "Swizzle the tile if banking does not fix it.",
          reference="adaptive_off"),
    Probe("l1_banks8_hash", {**ADAPTIVE_OFF, **l1_bank_knobs(8, hashing=1)}, Q,
          "More banks plus a hash index.", "Swizzle the tile.",
          reference="adaptive_off"),
    Probe("gmem_skip_l1", {**ADAPTIVE_OFF, **gmem_skip_l1_knobs()}, Q,
          "Removing the L1 write path entirely; costs the read hits.",
          "Stage reads through shared memory.", reference="adaptive_off"),
    Probe("dram_wq", dram_queue_knobs(), Q,
          "L2 was rejecting on a full downstream queue, not on bandwidth.",
          "Nothing."),
    Probe("l2_subpart2", l2_subpartition_knobs(2), Q,
          "More independent memory queues.", "Nothing."),
)

ray.init(num_cpus=200, num_gpus=8,
         resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
         ignore_reinit_error=True, log_to_driver=False)
node = AccelSimNode("/raid/sp56429/chia_hackathon/accel-sim-framework",
                    "/usr/local/cuda-12.8")
print(f"{len(NEW)} probes (+baseline), out={OUT}/sims", flush=True)
t0 = time.time()
res = run_probes(node, KL, probes=NEW, outroot=f"{OUT}/sims")
print(f"\n{len(res)} sims in {time.time()-t0:.0f}s\n", flush=True)

base = res.get("baseline")
ctl = res.get("adaptive_off")
ref = float(getattr(ctl, "cycles", 0) or getattr(base, "cycles", 0) or 0)
rows = []
print(f"{'probe':<20}{'cycles':>12}{'vs control':>12}  status")
for p in NEW:
    r = res.get(p.name)
    cyc = float(getattr(r, "cycles", 0) or 0)
    ok = bool(getattr(r, "ok", False)) and cyc > 0
    gain = (ref - cyc) / ref * 100.0 if (ok and ref) else None
    rows.append({"probe": p.name, "cycles": cyc, "ok": ok, "gain_pct": gain,
                 "knobs": p.knobs,
                 "unapplied": list(getattr(r, "unapplied", []) or [])})
    g = f"{gain:+11.2f}%" if gain is not None else f"{'-':>12}"
    st = "" if ok else (getattr(r, "error", "") or "FAILED")[:40]
    un = getattr(r, "unapplied", None)
    if un:
        st += f"  UNAPPLIED={list(un)[:3]}"
    print(f"{p.name:<20}{cyc:>12,.0f}{g}  {st}")
json.dump({"reference": ref, "rows": rows},
          open(f"{OUT}/newknobs.json", "w"), indent=1, default=str)
print(f"\nwrote {OUT}/newknobs.json")
