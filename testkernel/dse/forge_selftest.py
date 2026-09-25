#!/usr/bin/env python3
"""Exercise the kernel-forge gate with hand-written candidates -- no LLM.

The sandbox's value is entirely in what it *refuses*, and a gate that has only
ever seen good input is not evidence. This submits four candidates straight
through ``KernelForgeTool.submit_kernel``, one for each way the gate is
supposed to fire:

  tiled_32   a real shared-memory tiled SGEMM      -> must be ACCEPTED
  wrong_math a tiled kernel with an off-by-one k   -> must fail VERIFY
  subsample  a kernel doing a quarter of the FFMAs, scaled up
                                                   -> must fail VERIFY
  exfil      source that tries to reach outside the sandbox
                                                   -> must fail SCREEN

Only the first is a kernel anyone would want. The other three are the ones that
say whether "accepted" means anything.

``subsample`` is listed as a VERIFY failure, not a work failure, and that is
the honest result rather than a weaker test than intended. For a dense GEMM on
pseudo-random inputs every term contributes, so *any* kernel that skips
arithmetic also gets the numbers wrong and dies at the earlier gate -- I could
not construct a candidate that passes verification while doing materially less
work. Output matching is the binding gate here; ``work_preserved`` is depth
behind it, for a problem or a harness where that stops being true. So the work
gate is exercised directly, on histograms, by ``check_work_gate`` below, which
is the only way to reach it without a contrived kernel.
"""
from __future__ import annotations


import asyncio
import json
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(ROOT))
CHIA_ROOT = os.environ.get("CHIA_ROOT", os.path.join(PROJECT, "chia"))
sys.path.insert(0, CHIA_ROOT)

import ray

from chia.simulators.accelsim import AccelSimNode
from chia.simulators.accelsim_sandbox import KernelForgeTool

ROOT = os.path.join(PROJECT, "testkernel")
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/dse/out_forge")

TILED = r"""
#define TS 32
__global__ void sgemm_tiled(const float *A, const float *B, float *C, int N) {
  __shared__ float As[TS][TS], Bs[TS][TS];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
  float acc = 0.0f;
  for (int t = 0; t < N / TS; ++t) {
    As[ty][tx] = A[row * N + t * TS + tx];
    Bs[ty][tx] = B[(t * TS + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < TS; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = acc;
}
void launch(const float *A, const float *B, float *C, int N) {
  dim3 t(TS, TS), g(N / TS, N / TS);
  sgemm_tiled<<<g, t>>>(A, B, C, N);
}
"""

WRONG = TILED.replace("for (int k = 0; k < TS; ++k)", "for (int k = 1; k < TS; ++k)")\
             .replace("sgemm_tiled", "sgemm_wrong")


SUBSAMPLE = r"""
__global__ void sgemm_cheat(const float *A, const float *B, float *C, int N) {
  int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < N; k += 4) acc += A[row * N + k] * B[k * N + col];
  C[row * N + col] = acc * 4.0f;
}
void launch(const float *A, const float *B, float *C, int N) {
  dim3 t(16, 16), g(N / 16, N / 16);
  sgemm_cheat<<<g, t>>>(A, B, C, N);
}
"""

EXFIL = r"""
#include "/etc/passwd"
void launch(const float *A, const float *B, float *C, int N) {}
"""

CASES = [("tiled_32", TILED, "accepted"),
         ("wrong_math", WRONG, "verify"),
         ("subsample", SUBSAMPLE, "verify"),
         ("exfil", EXFIL, "screen")]


def check_work_gate(ref_opcodes: dict) -> dict:
    """Fire `work_preserved` directly, which no submittable kernel can do.

    Three histograms against the reference: itself (must pass), itself with
    half the FFMAs (must fail -- this is the shortcut the gate exists for), and
    itself with the compiler's immediate-materialisation idiom perturbed (must
    pass; counting `HFMA2.MMA` as arithmetic is what rejected a correct tiled
    SGEMM at ratio 0.9961 before `NON_WORK_FORMS` was introduced).
    """
    from chia.simulators.accelsim import work_preserved

    ident = dict(ref_opcodes)
    halved = {**ident, "FFMA": ident["FFMA"] // 2}
    idiom = {**ident, "HFMA2.MMA": ident.get("HFMA2.MMA", 0) // 3}
    out = {}
    for name, hist, want in (("identity", ident, True),
                             ("half_ffma", halved, False),
                             ("idiom_noise", idiom, True)):
        w = work_preserved(hist, ref_opcodes)
        out[name] = {"expected_ok": want, "ok": w["ok"],
                     "ratio": w["ratio"], "pass": w["ok"] == want}
        print(f"{'PASS' if w['ok'] == want else 'FAIL'}  work:{name:<12} "
              f"expected ok={want!s:<5} got ok={w['ok']!s:<5} ratio={w['ratio']:.4f}")
    return out


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ray.init(num_cpus=16, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    with open(os.path.join(OUT, "reference.json")) as fh:
        ref = json.load(fh)

    tool = KernelForgeTool(
        "forgeselftest", AccelSimNode(ACCELSIM, CUDA),
        harness_cu=f"{ROOT}/forge/harness.cu",
        work_dir=os.path.join(OUT, "selftest"),
        reference_dump=ref["reference_dump"],
        reference_opcodes=ref["opcodes"],
        n=ref["n"], seed=ref["seed"], max_est_sim_seconds=1800.0,


        simulate=bool(int(os.environ.get("FORGE_SELFTEST_SIM", "0"))),
        sim_sm=108, max_sims=2, accelsim_root=ACCELSIM, cuda_path=CUDA,
    )

    results, ok_all = {}, True
    for label, src, expect in CASES:
        rec = asyncio.run(tool.submit_kernel(label, src))
        got = rec["stage"] if not rec["accepted"] else "accepted"
        good = got == expect
        ok_all &= good
        results[label] = {"expected": expect, "got": got, "record": rec}
        print(f"{'PASS' if good else 'FAIL'}  {label:<12} expected {expect:<11} got {got}")
        if rec.get("error"):
            print(f"        {str(rec['error'])[:200]}")
        if rec["accepted"]:
            print(f"        LDG={rec['global_loads']} LDS={rec['shared_loads']} "
                  f"insts={rec['total_warp_insts']} "
                  f"work_ratio={rec['work_check']['ratio']:.4f}")
        if "work_check" in rec and not rec["accepted"]:
            w = rec["work_check"]
            print(f"        work {w['work_ops_candidate']} vs {w['work_ops_reference']} "
                  f"ratio={w['ratio']:.4f}")

    gate = check_work_gate(ref["opcodes"])
    ok_all &= all(v["pass"] for v in gate.values())

    path = os.path.join(OUT, "forge_selftest.json")
    with open(path, "w") as fh:
        json.dump({"all_as_expected": bool(ok_all), "cases": results,
                   "work_gate": gate}, fh, indent=1, default=str)
    print(f"\nall_as_expected={ok_all}  -> {path}")


if __name__ == "__main__":
    main()
