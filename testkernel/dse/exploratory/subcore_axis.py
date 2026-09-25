#!/usr/bin/env /usr/bin/python3.10
"""The only representable way to add tensor cores: add sub-cores.

``tcunits_axis.py`` swept ``-specialized_unit_3``'s ``num_units`` field 4 -> 16
and got ten bit-identical cycle counts.  The reason is *not* the ID_OC_SPEC
width I first claimed -- at stock II=8 the tensor path moves 0.5 instr/cycle,
nowhere near a 4/cycle dispatch limit.  The reason is shader.cc:5176-5179:

    if (sub_core_model) {
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds;
      reg_id = j / cusPerSched;           // reg_id in [0, num_sched_per_core)
    }

Every operand-collector unit carries its scheduler id as ``m_reg_id``, and
``collector_unit_t::dispatch()`` (shader.cc:5431) does

    m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp);

so only OC_EX_SPEC slots 0..num_sched-1 are ever written.  Functional unit k
reads slot k (shader.cc:2044, m_issue_reg_id = k).  Units 4..15 therefore poll
a register that nothing is architecturally able to fill.

**The usable count of any issue-partitioned FU class equals
``-gpgpu_num_sched_per_core``.**  That makes "4 tensor cores" not a knob but a
consequence, and the 4 stock units are genuinely all live: 4/II=8 = 0.5/cycle,
and 1048576 HMMA / 108 SM / 0.5 = 19418 issue cycles/SM, which is exactly the
number the II sweep's arithmetic closed on to 0.1%.

So this file buys the 5th through 8th tensor core the only way the model
allows: eight sub-cores.  That is a coupled change -- it doubles the
schedulers, SP, INT, SFU and LSU issue width too -- so the group is built to
decompose it rather than just report it:

  sched4_ii4   CONTROL.  Stock 4-sub-core machine, II 8->4.  1.0 HMMA/cyc.
  sched8       8 sub-cores, 8 tensor units, II stock 8.      1.0 HMMA/cyc.
               Same tensor throughput as sched4_ii4 by a different mechanism.
               If the model is coherent these two must agree.
  sched8_ii16  8 sub-cores, II 8->16.  0.5 HMMA/cyc == STOCK throughput.
               Isolates what the *rest* of the doubled core width buys with
               the tensor path pinned to its stock rate.  This is the control
               that says whether "more tensor cores" or "more everything" is
               doing the work.
  sched8_ii4   8 sub-cores, II 8->4.  2.0 HMMA/cyc -- 4x stock.  Does the
               tensor path keep paying once the core can actually feed it?

Asserts this configuration has to satisfy, all verified against source first:
  shader.cc:130-150   num_sched == size of every ID_OC_* register set, and ==
                      id_oc_spec_reg_width of every enabled specialized unit
  shader.cc:5170      num_reg_banks % num_sched == 0
  shader.cc:5171-5172 num_sched <= n_collector_units and divides it
Hence pipeline_widths, the four -specialized_unit_N strings, the register
banks and the generic collector pool all move together.  Ratios held at their
stock values: banks/sched 8, collector units/sched 2.
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
II = "-trace_opcode_latency_initiation_spec_op_3"


PIPE_8 = "8,8,8,8,8,8,8,8,8,8,16,8,8"


SUBCORE8 = {
    "-gpgpu_num_sched_per_core": "8",
    "-gpgpu_pipeline_widths": PIPE_8,
    "-gpgpu_num_sp_units": "8",
    "-gpgpu_num_sfu_units": "8",
    "-gpgpu_num_dp_units": "8",
    "-gpgpu_num_int_units": "8",
    "-gpgpu_num_tensor_core_units": "8",
    "-gpgpu_num_reg_banks": "64",
    "-gpgpu_operand_collector_num_units_gen": "16",
    "-gpgpu_operand_collector_num_in_ports_gen": "16",
    "-gpgpu_operand_collector_num_out_ports_gen": "16",
}


SPEC8 = {
    "-specialized_unit_1": "1,8,4,8,8,BRA",
    "-specialized_unit_2": "1,8,200,8,8,TEX",
    "-specialized_unit_3": "1,8,12,8,8,TENSOR",
    "-specialized_unit_4": "1,8,4,8,8,UDP",
}


POINTS: list[tuple[str, dict, dict, float, str]] = [
    ("sched4_ii4", {}, {II: "12,4"}, 1.0,
     "CONTROL: stock core, II 8->4 -- 1.0 HMMA/cyc the cheap way"),
    ("sched8", SUBCORE8, SPEC8, 1.0,
     "8 sub-cores, 8 tensor units, stock II -- 1.0 HMMA/cyc the wide way"),
    ("sched8_ii16", SUBCORE8, {**SPEC8, II: "12,16"}, 0.5,
     "CONTROL: 8 sub-cores at STOCK tensor throughput -- what does the rest buy?"),
    ("sched8_ii4", SUBCORE8, {**SPEC8, II: "12,4"}, 2.0,
     "8 sub-cores + II 8->4 -- 4x stock tensor throughput"),
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
        for label, knobs, tk, rate, _ in POINTS:
            od = os.path.join(OUT, kern, "subcore", label)
            ref = evaluate_point.chia_remote(
                ACCELSIM, CUDA, kern, f"{kern}/{label}", kl(kern), od,
                {**ADAPTIVE_OFF, **sm_knobs(108), **knobs}, True, tk,
                False, 10800)
            jobs.append((kern, label, rate, ref))

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
            kern, label, rate, _ = jobs[i]
            mark = "cache" if res.get("cached") else f"{res.get('wall_s',0):.0f}s"
            bad = "" if res.get("ok") else f"  !! {res.get('note','')[:70]}"
            un = res.get("unapplied") or []
            unm = f"  UNAPPLIED={un}" if un else ""
            print(f"[{n:>2}/{len(jobs)}] {kern:<16}{label:<13}{rate:>4}/cyc"
                  f"{res.get('cycles') or '-':>10}  {mark:>7}{unm}{bad}",
                  flush=True)

    out = []
    for i, (kern, label, rate, _) in enumerate(jobs):
        r = results[i]
        out.append({"kernel": kern, "label": label, "hmma_per_cyc": rate,
                    "cycles": r.get("cycles"), "power_w": r.get("power_w"),
                    "joules": r.get("joules"), "edp": r.get("edp"),
                    "occupancy": r.get("occupancy"),
                    "cta_per_core": r.get("cta_per_core"),
                    "tensor_w": (r.get("components") or {}).get("TENSOR"),
                    "ok": r.get("ok"), "cached": r.get("cached"),
                    "unapplied": r.get("unapplied"), "note": r.get("note")})
    path = os.path.join(OUT, "subcore_axis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\n{len(out)} points in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
