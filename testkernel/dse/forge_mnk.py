#!/usr/bin/env python3
"""Kernel-writing agent at the co-design loop's own problem shape.

``forge_run.py`` runs the same agent on a square NxNxN problem.  Nothing here
replaces it; this is an additional driver whose only purpose is to produce
kernels that can enter the loop's grid, which a square problem cannot:
``trace_shape_ok`` refuses to replay one problem size against another, and it
is right to -- the numbers would look fine and mean nothing.

Two things are done differently from ``forge_reference.py``, both forced by the
size of the problem rather than chosen:

* **The reference is not traced.**  A naive one-thread-per-element SGEMM at
  4096x4096x64 issues ~33.5M FFMA warp-instructions plus twice that in global
  loads.  Tracing it is pointless when the only thing the trace is used for is
  an arithmetic count that the shape already determines exactly.  So the work
  gate's reference histogram is computed analytically -- M*N*K/32 FFMA -- and
  the comment below is the whole derivation, auditable without running it.
  The *output* reference is still produced by really running the reference
  kernel on a real GPU; correctness is never asserted, only measured.

* **Accepted kernels are published into the loop's trace root.**  A kernel is
  only useful to the loop if the loop can find its trace under the tile name
  it asks for, so each accepted candidate is linked in as ``forge_<label>``.
  Publishing is additive: nothing already in the trace root is touched, and a
  name that already exists is refused rather than overwritten.
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
from chia.simulators.accelsim_sandbox import (
    KernelForgeTool, publish_tile, sandboxed_kernel_llm)

ROOT = os.path.join(PROJECT, "testkernel")
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
FORGE = f"{ROOT}/forge"
TRACE_ROOT = os.environ.get("DSE_TRACES", f"{ROOT}/dse/traces4k")
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/dse/out_forge_mnk")
M = int(os.environ.get("FORGE_M", "4096"))
N = int(os.environ.get("FORGE_N", "4096"))
K = int(os.environ.get("FORGE_K", "64"))
SEED = int(os.environ.get("FORGE_SEED", "12345"))
BUDGET = int(os.environ.get("FORGE_BUDGET", "8"))
MODEL = os.environ.get("FORGE_MODEL", "claude-sonnet-4-6")

SYSTEM = """You are a CUDA kernel engineer working through a single submission
tool. You cannot read or write files, run commands, or search the web; the only
thing you can do is submit complete .cu source text and read back what the
build, the correctness check, the hardware trace and a cycle-accurate
simulation say about it.

Work empirically. Submit, read the returned counters, and let them decide what
to change next. Do not submit a batch of variants you have not reasoned about,
and do not stop after the first kernel that is accepted -- an accepted kernel is
a correct kernel, not a fast one.

Rank on the measured cycle count and nothing else. Instruction counts, load
counts and occupancy are diagnostics that explain a cycle count; they are not
substitutes for it, and on this problem they disagree with it."""


def brief() -> str:
    ffma = M * N * K // 32
    return f"""Target: C = A x B, single-precision, row-major, on an NVIDIA A100
(sm_80). A is {M}x{K}, B is {K}x{N}, C is {M}x{N}. You write
`void launch(const float *A, const float *B, float *C, int M, int N, int K)`;
the harness owns allocation, initialisation and checking.

The arithmetic the problem requires is fixed: M*N*K = {M * N * K} fused
multiply-adds = {ffma} FFMA warp-instructions. A kernel that produces the
right numbers with materially fewer than that is rejected as not having done
the work.

What makes this shape interesting, and what an earlier run of this project
established by measurement on exactly this problem:

  * K is only {K}. The reduction is short, so there is very little arithmetic
    to hide memory latency behind. The measured baseline is bandwidth- and
    issue-bound, not compute-bound: it retires about 0.75 warp-instructions
    per SM per cycle, roughly 19% of issue capacity.
  * The dominant stall reason is the L1 data cache's miss queue on the WRITE
    path -- 93.9% of reservation failures, 92.6% of them writes. C is
    {M}x{N} = {M * N} floats and every one of them is written once; with K
    this short, the write traffic is a large fraction of all traffic.
  * Hardware knobs that widen that path buy a few percent. The best software
    change found so far -- a CUTLASS 128x128x16 tile -- buys more.

You are not restricted to what CUTLASS emits. You may choose any
decomposition: threadblock tile shape, register blocking, vectorised access,
split-K, a different layout of the epilogue, anything that is correct.

Budget: {BUDGET} simulations. Submit a kernel, read its cycles, and let the
counters choose the next one."""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ray.init(num_cpus=220, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 100.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})
    d = os.path.join(OUT, "reference")
    os.makedirs(d, exist_ok=True)
    node = AccelSimNode(ACCELSIM, CUDA)

    build = AccelSimNode.build_cuda.chia_remote_blocking(
        node, sources=[f"{FORGE}/harness_mnk.cu", f"{FORGE}/reference_mnk.cu"],
        binary=os.path.join(d, "harness"), arch_sm="sm_80")
    print("reference build ok:", build.ok,
          "" if build.ok else build.stderr_tail[:400], flush=True)
    if not build.ok:
        return 1

    ver = AccelSimNode.verify_kernel.chia_remote_blocking(
        node, binary=build.binary, dump=os.path.join(d, "reference.bin"),
        args=(str(M), str(N), str(K), str(SEED)), reference_dump=None, cwd=d)
    print(f"reference dump: ok={ver.ok} reason={ver.reason} "
          f"elems={ver.n_elems} (expect {M * N})", flush=True)
    if not ver.ok or ver.n_elems != M * N:
        return 1


    ref_opcodes = {"FFMA": M * N * K // 32}
    print("reference FFMA warp-insts (analytic):", ref_opcodes["FFMA"],
          flush=True)


    if len(sys.argv) > 1 and sys.argv[1] == "ref":
        print("reference ready; stopping before the agent (mode=ref)")
        return 0

    tool = KernelForgeTool(
        "forge_mnk", node,
        harness_cu=f"{FORGE}/harness_mnk.cu",
        work_dir=os.path.join(OUT, "candidates"),
        reference_dump=os.path.join(d, "reference.bin"),
        reference_opcodes=ref_opcodes,
        mnk=(M, N, K), seed=SEED,
        simulate=True, sim_sm=108, max_sims=BUDGET,
        max_est_sim_seconds=3600.0,
        accelsim_root=ACCELSIM, cuda_path=CUDA)
    print(f"forge tool up; shape {M}x{N}x{K}; budget {BUDGET}", flush=True)

    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        """Put one hand-written kernel through the same gate as the agent's.

        Same compile, same bit-exact output check against the reference, same
        work check, same trace and same simulation.  A kernel that arrives
        this way is indistinguishable downstream from one the agent wrote.
        """
        src = os.environ.get("FORGE_KERNEL") or (
            sys.argv[2] if len(sys.argv) > 2 else "")
        if not src or not os.path.exists(src):
            print("set FORGE_KERNEL=<path to .cu>, or pass it as argv[2]")
            return 2
        label = os.environ.get("FORGE_LABEL") or os.path.splitext(
            os.path.basename(src))[0]
        print(f"submitting {src} as {label!r}", flush=True)
        rec = tool._submit_kernel_sync(label, open(src).read())
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("cuda_source",)},
                         indent=1, default=str), flush=True)
        if not rec.get("accepted"):
            print(f"\nREJECTED at the {rec.get('stage','?')} stage: "
                  f"{rec.get('error') or '?'}")
            return 1
        t = publish_tile(label, rec.get("kernelslist") or "",
                         TRACE_ROOT, M, N, K)
        cyc = rec.get("cycles")
        print(f"\naccepted. "
              + (f"{cyc:,.0f} cycles, cta/sm={rec.get('cta_per_core', '-')}"
                 if cyc else
                 f"not timed: {(rec.get('sim') or {}).get('note', 'no sim')}"))
        if t:
            print(f"enter it in the grid with:  DSE_FORGE_TILES={t}")
        return 0


    sessions = os.path.join(OUT, "claude_sessions")
    os.makedirs(sessions, exist_ok=True)
    llm = sandboxed_kernel_llm(
        [tool], model=MODEL, system_message=SYSTEM, timeout_seconds=3600,
        projects_cwd=sessions, log_dir=sessions)
    print(f"tool up at {tool.hostname}:{tool.port}; briefing the agent",
          flush=True)
    r = llm.prompt(brief(), tools=[tool])
    print(f"agent finished: success={r.success} rc={r.returncode}", flush=True)
    res = {"success": r.success, "returncode": r.returncode}


    subs = tool.all_submissions()
    with open(os.path.join(OUT, "forge_mnk.json"), "w") as fh:
        json.dump({"shape": [M, N, K], "budget": BUDGET,
                   "submissions": subs, "agent": res}, fh,
                  indent=1, default=str)

    accepted = {k: v for k, v in (subs or {}).items() if v.get("accepted")}
    print(f"\naccepted {len(accepted)} of {len(subs or {})} submissions",
          flush=True)
    published = []
    for label, rec in sorted(accepted.items(),
                             key=lambda kv: kv[1].get("cycles") or 9e18):
        print(f"  {label:<28} {rec.get('cycles', 0):>12,.0f} cyc  "
              f"cta/sm={rec.get('cta_per_core', '-')}")
        t = publish_tile(label, rec.get("kernelslist") or "",
                         TRACE_ROOT, M, N, K)
        if t:
            published.append(t)
    with open(os.path.join(OUT, "published_tiles.json"), "w") as fh:
        json.dump(published, fh, indent=1)
    print("\npublished tiles:", published)
    print("->", os.path.join(OUT, "forge_mnk.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
