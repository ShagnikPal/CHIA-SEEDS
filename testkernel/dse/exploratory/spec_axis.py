#!/usr/bin/env /usr/bin/python3.10
"""The tensor pipe, via the knob that actually reaches it.

``compute_axis.py`` group B is void for every tensor point, and this file
says why.  In Accel-Sim's trace-driven path (ISA_Def/ampere_opcode.h:49):

    {"HMMA", OpcodeChar(OP_HMMA, SPECIALIZED_UNIT_3_OP)}

HMMA is *not* TENSOR_CORE_OP.  It is specialized unit 3.  So both knobs with
"tensor" in the name -- ``-gpgpu_num_tensor_core_units`` and
``-trace_opcode_latency_initiation_tensor`` -- drive a pipeline that receives
zero instructions from this trace, which is why a 256-cycle tensor latency
changed the cycle count by exactly nothing, twice.

Everything else still reports "tensor", which is the trap:
  * shader.cc:5904 counts SPECIALIZED_UNIT_3_OP into m_tensor_core_inst_issued
  * accelwattch_component_mapping.h:63 maps OP_HMMA -> TENSOR__OP, so
    AccelWattch bills 13.75 W of "TENSOR" power
Instruction counters and the power model both say the tensor unit is busy
while the timing model routes the work somewhere else entirely.

The live lever is trace_driven.cc:955 --

    latency = specialized_unit_latency[spec_id]      # spec_id = 2
    initiation_interval = specialized_unit_initiation[spec_id]

-- fed by ``-trace_opcode_latency_initiation_spec_op_3``, which defaults to
"4,4" and which nothing in this project has ever set.  Initiation interval is
the right proxy for "how many tensor cores": II 4->1 quadruples the HMMA issue
rate per SM, which is what 4x the tensor pipes would buy.  So this sweeps it
in both directions, and leads with a canary that must fire before any of it
is worth reading.
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
KNOB = "-trace_opcode_latency_initiation_spec_op_3"


POINTS = [
    ("spec3_canary", "256,64", "CANARY: is spec_op_3 the live HMMA lever?"),
    ("spec3_ii1", "4,1", "II 4->1: 4x the HMMA issue rate = 4x tensor pipes"),
    ("spec3_ii2", "4,2", "II 4->2: 2x the HMMA issue rate"),
    ("spec3_lat1", "1,4", "latency 4->1 at stock II: is it latency or throughput?"),
    ("spec3_ii8", "4,8", "STARVE: half the HMMA issue rate -- what does it cost?"),
    ("spec3_ii16", "4,16", "STARVE: a quarter of the issue rate"),
    ("spec3_lat16", "16,4", "STARVE: 4x HMMA latency at stock II"),
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
        for label, val, _ in POINTS:
            od = os.path.join(OUT, kern, "spec", label)
            ref = evaluate_point.chia_remote(
                ACCELSIM, CUDA, kern, f"{kern}/{label}", kl(kern), od,
                {**ADAPTIVE_OFF, **sm_knobs(108)}, True, {KNOB: val},
                False, 10800)
            jobs.append((kern, label, val, ref))

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
            kern, label, val, _ = jobs[i]
            mark = "cache" if res.get("cached") else f"{res.get('wall_s',0):.0f}s"
            bad = "" if res.get("ok") else f"  !! {res.get('note','')[:60]}"
            print(f"[{n:>2}/{len(jobs)}] {kern:<16}{label:<14}{val:>8}"
                  f"{res.get('cycles') or '-':>10}  {mark:>7}{bad}", flush=True)

    out = []
    for i, (kern, label, val, _) in enumerate(jobs):
        r = results[i]
        out.append({"kernel": kern, "label": label, "spec_op_3": val,
                    "cycles": r.get("cycles"), "power_w": r.get("power_w"),
                    "joules": r.get("joules"), "edp": r.get("edp"),
                    "tensor_w": (r.get("components") or {}).get("TENSOR"),
                    "ok": r.get("ok"), "cached": r.get("cached"),
                    "note": r.get("note")})
    path = os.path.join(OUT, "spec_axis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\n{len(out)} points in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
