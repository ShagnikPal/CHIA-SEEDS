#!/usr/bin/env /usr/bin/python3.10
"""Add tensor cores, not SMs: sweep the specialized-unit count directly.

``spec_axis.py`` proved HMMA throughput is the live compute lever, by moving
the initiation interval.  But "initiation interval" is not a thing you can
buy -- you buy *tensor cores*.  This sweeps the knob that counts them.

    -specialized_unit_3 <on>,<num_units>,<latency>,<id_oc_w>,<oc_ex_w>,TENSOR

Stock A100 (gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config:17) is
``1,4,12,4,4,TENSOR`` with ``-trace_opcode_latency_initiation_spec_op_3 12,8``.
Note this is NOT the "4,4" option-parser default in trace_driven.cc:903 --
the config file overrides it, and reading the registration default instead of
the applied config is what mis-signed the first pass of the II sweep.

Two hard constraints, both shader.cc:127-150 under the sub-core model:
  * field 4 (id_oc_spec_reg_width) MUST equal -gpgpu_num_sched_per_core (4);
    raising it asserts.  So the ID->OC register stays 4 wide regardless.
  * field 5 (oc_ex_spec_reg_width) is free.  This is "the TC pipeline
    register", and ``n8_w4`` vs ``n8_w8`` asks whether it binds.
  * field 2 (num_units) is free upward.  Downward it hits the same sub-core
    floor that deadlocks the SP/INT units, so the starve direction is taken
    from the II sweep instead, where it is already measured (-16.93%).

Aggregate HMMA throughput is num_units/II per SM, so these should be
substitutes.  ``n8_ii4`` exists to test that: 8 units at II=4 is 4x stock
aggregate, and must land on ``spec3_ii2`` (4 units at II=2, also 4x) at
101761 base / 99031 win.  If the two disagree, the units knob is doing
something other than what its name says, and the whole axis is suspect.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_codesign import ADAPTIVE_OFF, evaluate_point, sm_knobs

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
OUT = os.environ.get("DSE_OUT", f"{ROOT}/out_chia64")
TRACES = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")

BASE = "base"
WIN = "t128x256x16_s3"
UNITS = "-specialized_unit_3"
II = "-trace_opcode_latency_initiation_spec_op_3"

STOCK_UNITS = "1,4,12,4,4,TENSOR"
STOCK_II = "12,8"


def spec(n: int, ocex: int, lat: int = 12) -> str:
    return f"1,{n},{lat},4,{ocex},TENSOR"


POINTS = [
    ("tc_n4_w4", {UNITS: spec(4, 4)}, 1.0,
     "CONTROL: restate stock explicitly -- must reproduce 108864 / 101084"),
    ("tc_n6_w6", {UNITS: spec(6, 6)}, 1.5,
     "1.5x the tensor cores: where is the knee?"),
    ("tc_n8_w8", {UNITS: spec(8, 8)}, 2.0,
     "2x the tensor cores + 2x wide OC_EX -- the II=4 equivalent"),
    ("tc_n8_w4", {UNITS: spec(8, 4)}, 2.0,
     "2x the cores, STOCK pipeline register: does OC_EX width bind?"),
    ("tc_n16_w16", {UNITS: spec(16, 16)}, 4.0,
     "4x the tensor cores -- the II=2 equivalent"),
    ("tc_n8_ii4", {UNITS: spec(8, 8), II: "12,4"}, 4.0,
     "CROSS-CHECK: 8 units at II=4 must equal 4 units at II=2 (101761/99031)"),
    ("tc_n4_w16", {UNITS: spec(4, 16)}, 1.0,
     "CONTROL: stock cores, 4x pipeline register -- register alone buys 0?"),
]


def kl(k: str) -> str:
    return f"{TRACES}/{k}/traces/kernelslist.g"


def main() -> None:
    ray.init(num_cpus=56, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})
    t0 = time.time()
    jobs = []
    for kern in (BASE, WIN):
        for label, tk, mult, _ in POINTS:
            od = os.path.join(OUT, kern, "tcunits", label)
            ref = evaluate_point.chia_remote(
                ACCELSIM, CUDA, kern, f"{kern}/{label}", kl(kern), od,
                {**ADAPTIVE_OFF, **sm_knobs(108)}, True, tk, False, 10800)
            jobs.append((kern, label, mult, ref))

    print(f"dispatched {len(jobs)} points", flush=True)
    idx = {id(r): i for i, (_, _, _, r) in enumerate(jobs)}
    pending = [j[3] for j in jobs]
    results: dict[int, dict] = {}
    n = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for r in ready:
            res = ray.get(r)
            i = idx[id(r)]
            results[i] = res
            n += 1
            kern, label, mult, _ = jobs[i]
            mark = "cache" if res.get("cached") else f"{res.get('wall_s',0):.0f}s"
            bad = "" if res.get("ok") else f"  !! {res.get('note','')[:70]}"
            print(f"[{n:>2}/{len(jobs)}] {kern:<16}{label:<12}{mult:>5.1f}x"
                  f"{res.get('cycles') or '-':>10}  {mark:>7}{bad}", flush=True)

    out = []
    for i, (kern, label, mult, _) in enumerate(jobs):
        r = results[i]
        out.append({"kernel": kern, "label": label, "issue_mult": mult,
                    "cycles": r.get("cycles"), "power_w": r.get("power_w"),
                    "joules": r.get("joules"), "edp": r.get("edp"),
                    "tensor_w": (r.get("components") or {}).get("TENSOR"),
                    "ok": r.get("ok"), "cached": r.get("cached"),
                    "note": r.get("note")})
    path = os.path.join(OUT, "tcunits_axis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\n{len(out)} points in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
