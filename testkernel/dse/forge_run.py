#!/usr/bin/env python3
"""Drive the sandboxed kernel-writing agent: the software half of the loop.

Every other software move in this project came from a fixed menu -- CUTLASS
tile shapes enumerated by ``propose_tiles``. This is the axis where the
software is *written* rather than selected, and it is the only place an LLM
touches anything executable, which is why it is the only place that needs a
sandbox (``accelsim_sandbox.py``: no Bash/Read/Write/Edit, one MCP tool, a
source screener, a fixed harness the candidate cannot edit).

The brief handed to the agent is measured, not asserted: the two numbers in it
come from the reference's own trace (``forge_reference.py`` ->
``out_forge/reference.json``). The agent is told what the machine costs, not
what to write; the tile shape, the blocking and the use of shared memory are
its decisions, and the gate -- output match + arithmetic-op count -- is the
same for every candidate.

Nothing here scores a candidate -- but the tool now does. The first run of
this driver returned only static opcode counts, and the agent, correctly
optimising the one number it was given, shipped a kernel that is +69.8%
slower in simulation than the one it ranked last: a 256x128 tile issues the
fewest instructions at N=512 and launches 8 threadblocks onto a 108-SM part.
So ``submit_kernel`` now simulates every accepted kernel and returns cycles,
and the brief below ranks on cycles. ``forge_rank.py`` still re-measures all
of them independently afterwards, because the agent's own ranking is a claim
and not a result.
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
from chia.simulators.accelsim_sandbox import KernelForgeTool, sandboxed_kernel_llm

ROOT = os.path.join(PROJECT, "testkernel")
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
FORGE = f"{ROOT}/forge"
OUT = os.environ.get("FORGE_OUT", f"{ROOT}/dse/out_forge")
BUDGET = int(os.environ.get("FORGE_BUDGET", "6"))

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


def brief(ref: dict) -> str:
    ops = ref["opcodes"]
    ffma = sum(v for k, v in ops.items() if k.split(".")[0] == "FFMA")
    ldg = sum(v for k, v in ops.items() if k.startswith("LDG"))
    lds = sum(v for k, v in ops.items() if k.startswith("LDS"))
    n = ref["n"]
    return f"""Target: C = A x B, single-precision, {n}x{n}x{n}, row-major, on an
NVIDIA A100 (sm_80). You write `void launch(const float *A, const float *B,
float *C, int N)`; the harness owns allocation, initialisation and checking.

The reference implementation is one thread per output element. Its trace, from
the real GPU, measured:

  FFMA  {ffma:>9} warp-instructions   ({ffma} x 32 = {ffma * 32} = {n}^3, the
                                        arithmetic the problem requires)
  LDG   {ldg:>9} warp-instructions   (global loads)
  LDS   {lds:>9} warp-instructions   (shared-memory loads)

So it issues {ldg / ffma:.2f} global loads per fused multiply-add and never
touches shared memory. On this machine a global load that misses is hundreds of
cycles and an FFMA is one; that ratio is the whole cost of this kernel.

The arithmetic is fixed -- the gate compares your FFMA count against the
reference's and rejects a kernel that produces the right numbers with
materially less arithmetic, so there is no credit in computing less. What you
can move is the {ldg} global loads: each element of A and B is re-read many
times, and reuse is the only lever.

**The objective is the cycle count.** Every accepted kernel is simulated on
this trace and `cycles` comes back with it; that is the number to rank on. The
load and instruction counts come back too, and they are worth reading, but
they are explanations rather than the objective -- on this problem they point
the other way. Two things make them diverge:

  - Occupancy. The GPU has 108 SMs. A {n}x{n} problem tiled at 256x128 is
    {n // 256}x{n // 128} = {(n // 256) * (n // 128)} threadblocks, which
    leaves most of the machine idle no matter how few instructions each block
    issues. A smaller tile issues more instructions and may still finish
    sooner. Split-K costs strictly more loads and buys parallelism; it may
    win.
  - Latency hiding. Double buffering, deeper software pipelining and
    independent load streams all *add* instructions. What they buy is
    overlap, which an instruction count cannot see and a cycle count can.

Budget: about {BUDGET} submissions, and a simulation budget that the tool
reports as `sims_left`. Report at the end which candidate you would ship, its
cycle count, and be explicit about anything you tried that did not work --
including anything the cycle count contradicted."""


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ray.init(num_cpus=16, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": 48.0, "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})

    with open(os.path.join(OUT, "reference.json")) as fh:
        ref = json.load(fh)

    node = AccelSimNode(ACCELSIM, CUDA)
    tool = KernelForgeTool(
        "kernelforge", node,
        harness_cu=f"{FORGE}/harness.cu",
        work_dir=os.path.join(OUT, "candidates"),
        reference_dump=ref["reference_dump"],
        reference_opcodes=ref["opcodes"],
        n=ref["n"], seed=ref["seed"],
        max_est_sim_seconds=1800.0,


        simulate=True, sim_sm=108,
        max_sims=int(os.environ.get("FORGE_SIMS", "12")),
        accelsim_root=ACCELSIM, cuda_path=CUDA,
    )

    sessions = os.path.join(OUT, "claude_sessions")
    os.makedirs(sessions, exist_ok=True)
    llm = sandboxed_kernel_llm(
        [tool], system_message=SYSTEM, timeout_seconds=2700,
        projects_cwd=sessions, log_dir=sessions,
    )

    print(f"tool up at {tool.hostname}:{tool.port}; briefing the agent")
    res = llm.prompt(brief(ref), tools=[tool])
    print(f"agent finished: success={res.success} rc={res.returncode}")


    subs = tool.all_submissions()
    acc = tool.accepted()
    print(f"\nsubmissions: {len(subs)}  accepted: {len(acc)}")
    for label, rec in subs.items():
        if rec.get("accepted"):
            cyc = rec.get("cycles")
            print(f"  [ok]   {label:<24} "
                  f"cyc={f'{cyc:,.0f}' if cyc else '-':>11}  "
                  f"LDG={rec['global_loads']:<9} "
                  f"LDS={rec['shared_loads']:<9} "
                  f"insts={rec['total_warp_insts']}")
        else:
            print(f"  [fail] {label:<24} {rec.get('stage')}: "
                  f"{str(rec.get('error'))[:120]}")

    path = os.path.join(OUT, "forge_run.json")
    with open(path, "w") as fh:
        json.dump({"submissions": subs, "accepted": list(acc),
                   "agent_report": res.result,
                   "success": res.success}, fh, indent=1, default=str)
    print("\n->", path)
    print("\n--- agent report ---\n" + (res.result or "")[-3000:])


if __name__ == "__main__":
    main()
