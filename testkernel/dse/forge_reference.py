#!/usr/bin/env python3
"""Build the reference the kernel-writing agent is measured against.

``KernelForgeTool`` needs two things bound at construction that no agent may
supply: a *reference dump* (the numbers a candidate has to reproduce) and a
*reference opcode histogram* (the arithmetic it has to actually perform).
Both come from the same naive SGEMM in ``forge/reference.cu``, built against
the same fixed harness every candidate is linked into -- so the gate is the
reference's own behaviour, not a hand-written expectation.

Output-matching alone is not a gate: a kernel can reproduce C by doing less
arithmetic (precomputing, exploiting the seeded inputs, skipping terms) and
every such shortcut reads as a speedup in the simulator. ``work_preserved``
against this histogram is the half of the gate that catches it.
"""
from __future__ import annotations

import json
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray

from chia.simulators.accelsim import AccelSimNode

ROOT = os.path.join(PROJECT, "testkernel")
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
FORGE = f"{ROOT}/forge"
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/dse/out_forge")
N = int(os.environ.get("FORGE_N", "256"))
SEED = int(os.environ.get("FORGE_SEED", "12345"))


def main() -> None:
    ray.init(num_cpus=16, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    d = os.path.join(OUT, "reference")
    os.makedirs(d, exist_ok=True)
    node = AccelSimNode(ACCELSIM, CUDA)

    build = AccelSimNode.build_cuda.chia_remote_blocking(
        node, sources=[f"{FORGE}/harness.cu", f"{FORGE}/reference.cu"],
        binary=os.path.join(d, "harness"), arch_sm="sm_80")
    print("build ok:", build.ok, build.stderr_tail[:300] if not build.ok else "")
    if not build.ok:
        sys.exit(1)


    ver = AccelSimNode.verify_kernel.chia_remote_blocking(
        node, binary=build.binary, dump=os.path.join(d, "reference.bin"),
        args=(str(N), str(SEED)), reference_dump=None, cwd=d)
    print(f"reference dump: ok={ver.ok} reason={ver.reason} "
          f"elems={ver.n_elems} (expect {N * N})")
    if not ver.ok or ver.n_elems != N * N:
        sys.exit(1)

    tr = AccelSimNode.trace_app.chia_remote_blocking(
        node, app_cmd=[build.binary, str(N), str(SEED),
                       os.path.join(d, "trace_out.bin")],
        trace_dir=os.path.join(d, "traces"), max_est_sim_seconds=3600.0,
        cwd=d)
    print(f"trace: ok={tr.ok} kernels={tr.n_kernels} "
          f"est_sim={getattr(tr, 'est_sim_seconds', 0):.0f}s "
          f"{'' if tr.ok else (tr.aborted_reason or tr.stderr_tail[:200])}")
    if not tr.ok:
        sys.exit(1)

    top = sorted(tr.opcodes.items(), key=lambda kv: -kv[1])[:12]
    print("top opcodes: " + ", ".join(f"{k}={v}" for k, v in top))
    print("total warp insts:", sum(tr.opcodes.values()))

    meta = {"n": N, "seed": SEED, "binary": build.binary,
            "reference_dump": os.path.join(d, "reference.bin"),
            "kernelslist": tr.kernelslist, "n_kernels": tr.n_kernels,
            "est_sim_seconds": getattr(tr, "est_sim_seconds", None),
            "opcodes": tr.opcodes}
    path = os.path.join(OUT, "reference.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print("->", path)


if __name__ == "__main__":
    main()
