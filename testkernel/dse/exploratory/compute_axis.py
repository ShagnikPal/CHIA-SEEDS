#!/usr/bin/env /usr/bin/python3.10
"""Compute axis: what does *more compute* buy, and what does less of it cost?

Three questions the co-design loop never asked, all at 4096x4096x64:

  A. Fine SM granularity.  ``2x_sm`` said +18.19% and ``half_sm`` said -64.45%,
     but the loop only ever banded {104,106,108,110,112}.  What does ONE more
     SM buy?  Two?  This sweeps 104..116 at 1-SM granularity on both the
     baseline tile and the co-designed tile, so the "+1 SM" question gets a
     number instead of an interpolation.

  B. What kind of compute.  ``many_tensor_units`` (4x the tensor pipes AND a
     4x wider OC_EX_TENSOR_CORE) returned bit-identical cycles.  That is
     either "tensor issue is not the constraint" or "the knob is a no-op in
     trace-driven mode" -- and those are opposite conclusions.  In trace mode
     the tensor pipe is driven by -trace_opcode_latency_initiation_tensor,
     which defaults to "4,1" and which NO probe has ever moved.  ``tc_canary``
     sets it to 256,64: if that does not slow the kernel down, the lever is
     dead and every tensor result here is unmeasurable.  Everything else in
     this group is only interpretable if the canary fires.

     The group also runs the *starve* direction (fewer units, worse
     initiation interval) because "more buys nothing" does not imply "less
     costs nothing", and only the second one frees area to spend elsewhere.

  C. Where the SM gain comes from.  If +18% at 2x SM survives cutting the
     tensor units to 1, the gain is not compute -- it is the per-SM memory
     path (each SM brings its own L1 and its own miss queue, and the miss
     queue is the attributed limiter).  That is the difference between
     "buy more tensor cores" and "buy more SMs", which is the actual
     question on the table.

Writes SM-sweep points into the existing ``band/`` outdirs so the ten already
simulated configurations return from the fingerprint cache instead of being
paid for twice.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_codesign import (
    ADAPTIVE_OFF, evaluate_point, sm_knobs,
)

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
OUT = os.environ.get("DSE_OUT", f"{ROOT}/out_chia64")
TRACES = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")
N_SLOTS = 48

BASE = "base"
WIN = "t128x256x16_s3"
STOCK = dict(ADAPTIVE_OFF)


PIPE_STOCK = "4,4,4,4,4,4,4,4,4,4,8,4,4"
PIPE_TC16 = "4,4,4,4,4,4,4,4,4,4,8,4,16"


def kl(kernel: str) -> str:
    return f"{TRACES}/{kernel}/traces/kernelslist.g"


SM_SWEEP = list(range(104, 117))
BAND_CACHED = {104, 106, 108, 110, 112}


COMPUTE: list[tuple[str, dict, dict, str]] = [
    ("tc_canary", {}, {"-trace_opcode_latency_initiation_tensor": "256,64"},
     "CANARY: is the trace-mode tensor pipe a live lever at all?"),
    ("fast_tensor", {}, {"-trace_opcode_latency_initiation_tensor": "1,1"},
     "If every HMMA retired in 1 cycle at II=1, how much faster?"),
    ("tc_ii4", {}, {"-trace_opcode_latency_initiation_tensor": "4,4"},
     "STARVE: 4x worse tensor initiation interval -- what does it cost?"),
    ("tc1", {"-gpgpu_num_tensor_core_units": "1"}, {},
     "STARVE: 1 tensor pipe per SM instead of 4 -- what does it cost?"),
    ("tc2", {"-gpgpu_num_tensor_core_units": "2"}, {},
     "STARVE: 2 tensor pipes per SM -- the half-way point."),
    ("tc16_wide", {"-gpgpu_num_tensor_core_units": "16",
                   "-gpgpu_pipeline_widths": PIPE_TC16}, {},
     "4x the tensor pipes AND a 4x wider TC pipeline register."),
    ("int_fast", {}, {"-trace_opcode_latency_initiation_int": "1,1"},
     "If integer ops were free, how much faster? (address arithmetic)"),
    ("int_slow", {}, {"-trace_opcode_latency_initiation_int": "16,4"},
     "STARVE: 4x integer latency, 4x II -- what does it cost?"),
    ("sp_fast", {}, {"-trace_opcode_latency_initiation_sp": "1,1"},
     "If FP32 ops were free, how much faster? (the epilogue)"),
    ("sp_slow", {}, {"-trace_opcode_latency_initiation_sp": "16,4"},
     "STARVE: 4x FP32 latency, 4x II -- what does it cost?"),
    ("alu_fast", {}, {"-trace_opcode_latency_initiation_int": "1,1",
                      "-trace_opcode_latency_initiation_sp": "1,1"},
     "Both scalar pipes free -- the fast_alu probe, this time applied."),
]


SCALE_SM = [126, 144, 162, 180, 198, 216]
DECOMP: list[tuple[str, dict, dict, str]] = [
    ("sm216", sm_knobs(216), {}, "the +18% probe, re-measured with power"),
    ("sm216_tc1", {**sm_knobs(216), "-gpgpu_num_tensor_core_units": "1"}, {},
     "216 SMs with 1/4 the tensor pipes: does the SM gain need the compute?"),
    ("sm216_tcslow", sm_knobs(216),
     {"-trace_opcode_latency_initiation_tensor": "16,4"},
     "216 SMs with a 4x worse tensor pipe: same question, latency side."),
    ("sm108_tc1", {**sm_knobs(108), "-gpgpu_num_tensor_core_units": "1"}, {},
     "control for sm216_tc1 at stock SM count"),
]


def boot() -> None:
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": float(N_SLOTS), "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})


def ev(kernel: str, point: str, knobs: dict, trace_knobs: dict, outdir: str):
    return evaluate_point.chia_remote(
        ACCELSIM, CUDA, kernel, f"{kernel}/{point}", kl(kernel), outdir,
        knobs, True, trace_knobs or None, False, 10800)


def main() -> None:
    boot()
    t0 = time.time()
    jobs: list[tuple[str, str, str, object]] = []


    for kern, tag in ((BASE, "base"), (WIN, "ref")):
        for n in SM_SWEEP:
            knobs = {**STOCK, **sm_knobs(n)}
            if n in BAND_CACHED:
                od = os.path.join(OUT, kern, "band", f"band_{tag}_sm{n}")
            else:
                od = os.path.join(OUT, kern, "smfine", f"sm{n}")
            jobs.append(("A", kern, f"sm{n}", ev(kern, f"sm{n}", knobs, {}, od)))


    for kern in (BASE, WIN):
        for label, knobs, tk, _ in COMPUTE:
            kk = {**STOCK, **sm_knobs(108), **knobs}
            od = os.path.join(OUT, kern, "compute", label)
            jobs.append(("B", kern, label, ev(kern, label, kk, tk, od)))


    for n in SCALE_SM:
        kk = {**STOCK, **sm_knobs(n)}
        od = os.path.join(OUT, BASE, "smscale", f"sm{n}")
        jobs.append(("C", BASE, f"sm{n}", ev(BASE, f"sm{n}", kk, {}, od)))
    for label, knobs, tk, _ in DECOMP:
        kk = {**STOCK, **knobs}
        od = os.path.join(OUT, BASE, "smscale", label)
        jobs.append(("C", BASE, label, ev(BASE, label, kk, tk, od)))

    print(f"dispatched {len(jobs)} points", flush=True)
    refs = [j[3] for j in jobs]
    out = []
    done_n = 0
    pending = list(refs)
    idx = {id(r): i for i, r in enumerate(refs)}
    results: dict[int, dict] = {}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        for r in ready:
            res = ray.get(r)
            i = idx[id(r)]
            results[i] = res
            done_n += 1
            g, kern, lab, _ = jobs[i]
            cyc = res.get("cycles")
            mark = "cache" if res.get("cached") else f"{res.get('wall_s',0):.0f}s"
            bad = "" if res.get("ok") else f"  !! {res.get('note','')[:60]}"
            un = res.get("unapplied") or []
            unm = f"  UNAPPLIED={un}" if un else ""
            print(f"[{done_n:>3}/{len(jobs)}] {g} {kern:<16}{lab:<14}"
                  f"{cyc if cyc else '-':>10}  {mark:>7}{unm}{bad}", flush=True)

    for i, (g, kern, lab, _) in enumerate(jobs):
        r = results[i]
        out.append({"group": g, "kernel": kern, "label": lab,
                    "cycles": r.get("cycles"), "power_w": r.get("power_w"),
                    "joules": r.get("joules"), "edp": r.get("edp"),
                    "occupancy": r.get("occupancy"),
                    "cta_per_core": r.get("cta_per_core"),
                    "dram_reads": r.get("dram_reads"),
                    "dram_writes": r.get("dram_writes"),
                    "l2_bw_gbps": r.get("l2_bw_gbps"),
                    "tensor_w": (r.get("components") or {}).get("TENSOR"),
                    "ok": r.get("ok"), "cached": r.get("cached"),
                    "unapplied": r.get("unapplied"), "note": r.get("note")})

    path = os.path.join(OUT, "compute_axis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\n{len(out)} points in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
