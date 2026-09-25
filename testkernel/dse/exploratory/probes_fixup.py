"""Re-run probes whose config was corrected after the main wave dispatched.

The first ``2x_subcore`` went out missing -gpgpu_num_reg_banks 64 and
-gpgpu_operand_collector_num_units_gen 16, which shader.cc:5170-5171 require
at 8 schedulers. It would have aborted, and an abort here reads as "4
sub-cores is the architectural ceiling" -- a false negative dressed up as a
finding. This runs the corrected probe into its own directory so the result
is a measurement rather than a config error. Baseline is not re-run: the main
wave's baseline is on the same trace and the same simulator, and its cycle
count is read out of probes_both.json when the two are merged.
"""
import os, sys, time, json
sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
import ray
from chia.simulators import AccelSimNode, sweep_configs
from chia.simulators.accelsim_probes import PROBE_BY_NAME

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
TILE = sys.argv[1] if len(sys.argv) > 1 else "base"
NAMES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2x_subcore"])
OUT = f"{ROOT}/out_probes_both/{TILE}/fixup"
KL = f"{ROOT}/traces4k/{TILE}/traces/kernelslist.g"
assert os.path.exists(KL), KL

probes = [PROBE_BY_NAME[n] for n in NAMES]
ray.init(num_cpus=16, num_gpus=0,
         resources={"accelsim": 8.0}, ignore_reinit_error=True,
         log_to_driver=False)
node = AccelSimNode("/raid/sp56429/chia_hackathon/accel-sim-framework",
                    "/usr/local/cuda-12.8")
t0 = time.time()
res = sweep_configs(
    node, kernelslist=KL,
    candidates={p.name: dict(p.knobs) for p in probes},
    trace_candidates={p.name: dict(p.trace_knobs) for p in probes if p.trace_knobs},
    arch="SM80_A100", outroot=f"{OUT}/sims")
print(f"{len(res)} sims in {time.time()-t0:.0f}s\n", flush=True)

out = {}
for n, r in res.items():
    cyc = (r.stats or {}).get("cycles")
    print(f"{n:<20} ok={r.ok}  cycles={cyc}  "
          f"unapplied={getattr(r, 'unapplied', None)}")
    if not r.ok:
        hay = (r.stderr_tail or "") + (r.stdout_tail or "")
        for ln in hay.splitlines():
            if "ssertion" in ln or "deadlock" in ln.lower():
                print("   ", ln.strip()); break
    out[n] = {"ok": r.ok, "cycles": cyc,
              "power_w": (r.stats or {}).get("power_w"),
              "stderr_tail": (r.stderr_tail or "")[-600:]}
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/fixup.json", "w"), indent=1, default=str)
print(f"\nwrote {OUT}/fixup.json")
