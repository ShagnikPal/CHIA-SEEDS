"""Second wave: the transaction-count lever, after write policy measured inert.

Removing write-allocate applies cleanly and pays exactly 0.00%: the L1 is
write-through, so a store occupies a miss-queue entry forwarding data to L2
whether or not a line is allocated. The entries are spent on *forwarding*,
not on *allocation*, so the only way to need fewer of them is to make each
one carry more.

Both caches are sectored, which is the opposite of that: a warp storing a
full 128 B line becomes four 32 B transactions and four queue entries. This
wave tests de-sectoring, plus the three throughput limits the registrar audit
turned up that nobody had ever set -- the LSU issues one memory transaction
per core cycle (-gpgpu_mem_unit_ports 1), buffers two response packets
(-gpgpu_n_ldst_response_buffer_size 2), and the load request coalescer is off.
"""
import json, os, sys, time
sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
import ray
from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_probes import Probe, run_probes
from chia.simulators.accelsim_codesign import (
    line_cache_knobs, mem_port_knobs, ldst_buffer_knobs, lrc_knobs,
    queue_knobs, cache_type_spec, ADAPTIVE_OFF)


NORMAL_MQ4 = {k: cache_type_spec(v, "N") for k, v in queue_knobs(4).items()}

TILE = "base"
OUT = f"/raid/sp56429/chia_hackathon/testkernel/dse/out_probes_both/{TILE}"
KL = f"/raid/sp56429/chia_hackathon/testkernel/dse/traces4k/{TILE}/traces/kernelslist.g"

Q = "Can anything reduce the number of write transactions, now that\n     write policy is known not to?"

NEW = (
    Probe("adaptive_off", dict(ADAPTIVE_OFF), "control for every dl1 edit",
          "n/a", "n/a"),
    Probe("l1_normal", {**ADAPTIVE_OFF, **line_cache_knobs("N")}, Q,
          "A full-line store became one transaction instead of four, so the "
          "miss queue drained four times faster.",
          "Nothing -- this is the line/sector unit, not a tiling choice.",
          reference="adaptive_off"),
    Probe("l1l2_normal", {**ADAPTIVE_OFF, **line_cache_knobs("N", l2=True)}, Q,
          "Same, carried through L2 so the width is not re-split downstream.",
          "Nothing.", reference="adaptive_off"),
    Probe("mem_ports2", {**ADAPTIVE_OFF, **mem_port_knobs(2)}, Q,
          "One memory transaction per core cycle was the cap, not the queue.",
          "Nothing.", reference="adaptive_off"),
    Probe("mem_ports4", {**ADAPTIVE_OFF, **mem_port_knobs(4)}, Q,
          "Same, with headroom to see whether it saturates.",
          "Nothing.", reference="adaptive_off"),
    Probe("ldst_buf8", {**ADAPTIVE_OFF, **ldst_buffer_knobs(8)}, Q,
          "Two response slots backed up into the queue that was full.",
          "Nothing.", reference="adaptive_off"),
    Probe("lrc_on", {**ADAPTIVE_OFF, **lrc_knobs()}, Q,
          "Coalescing requests merged what the queue was holding separately.",
          "Nothing.", reference="adaptive_off"),
    Probe("l1_normal_ports4",
          {**ADAPTIVE_OFF, **line_cache_knobs("N"), **mem_port_knobs(4)}, Q,
          "Width and issue rate were co-limiting, so neither alone moved.",
          "Nothing.", reference="adaptive_off"),
    Probe("l1_normal_mq4",
          {**ADAPTIVE_OFF, **NORMAL_MQ4}, Q,
          "Wider transactions plus a deeper queue, to see if the known "
          "+1.2% miss-queue ceiling lifts once each entry carries 4x.",
          "Nothing.", reference="adaptive_off"),
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
          open(f"{OUT}/linewave.json", "w"), indent=1, default=str)
print(f"\nwrote {OUT}/linewave.json")
