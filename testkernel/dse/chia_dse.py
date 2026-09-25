#!/usr/bin/env python3
"""Run the autonomous co-design loop.  This file holds no decisions.

Everything that decides anything lives in ``chia.simulators.accelsim_codesign``.
What is left here is what a driver should be: where the tools are installed,
how big the cluster is, and which shape to study.

    chia_dse.py trace [K]      capture the baseline trace (do this first)
    chia_dse.py smoke          one cached point, to prove dispatch works
    chia_dse.py run [K]        the full chain (default K=64)
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

from chia.simulators.accelsim_codesign import (
    ADAPTIVE_OFF, BASELINE_TILE, LoopConfig, build_and_trace, codesign_loop,
    evaluate_point, sm_knobs, trace_name,
)

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
CUTLASS = os.environ.get(
    "CUTLASS_ROOT",
    os.path.join(PROJECT, "cutlass"))


N_SLOTS = 100


def boot() -> None:
    """Custom resources, because the ChiaFunctions are pinned to them.

    ``accelsim`` is CPU-only simulation; ``accelsim_gpu`` gates the tracer,
    which needs a real device.  Declaring them separately is what stops 100
    simulations from queueing behind 8 GPUs.
    """
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": float(N_SLOTS), "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})


def forge_tiles() -> tuple:
    """Tiles written by the kernel-forge agent, if it has published any.

    Purely additive.  ``codesign_loop`` appends these to whatever
    ``propose_tiles`` already returned and still picks the winner by the same
    EDP-then-cycles rule over the union, so a forge tile can only win by
    being faster than every tile already in the grid -- it cannot displace a
    recorded result.  An empty tuple reproduces the deterministic run exactly,
    which is what happens when the agent has never been run.

    ``DSE_FORGE_TILES`` (comma separated) overrides the file;
    ``DSE_FORGE_TILES=-`` disables the forge entirely.
    """
    env = os.environ.get("DSE_FORGE_TILES")
    if env is not None:
        return tuple(t for t in (x.strip() for x in env.split(",")) if t
                     and t != "-")
    path = os.environ.get(
        "DSE_FORGE_OUT", f"{ROOT}/out_forge_mnk") + "/published_tiles.json"
    if not os.path.exists(path):
        return ()
    try:
        with open(path) as fh:
            names = json.load(fh)
    except Exception as exc:
        print(f"forge tiles: {path} unreadable ({exc}); "
              "running without them")
        return ()
    return tuple(str(n) for n in names if n)


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    traces = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")
    out = os.environ.get("DSE_OUT", f"{ROOT}/out_chia_k{k}")
    boot()

    if what == "trace":
        """Capture the baseline trace, which every later step replays.

        Step 1 of the loop reads this trace; it does not create it, because
        every other tile is traced on demand and the baseline is the one that
        has to exist before there is anything to compare against.  Needs a
        GPU.  Re-running is safe: an existing trace is left alone.
        """
        r = ray.get(build_and_trace.chia_remote(
            ACCELSIM, CUDA, CUTLASS, f"{ROOT}/src/gemm_tpl.cu",
            json.dumps(BASELINE_TILE.__dict__, default=list),
            4096, 4096, k, f"{ROOT}/build_chia_k{k}", traces, 3600))
        print(json.dumps(r, indent=1, default=str))
        if not r.get("ok"):
            print(f"\ntrace FAILED at the {r.get('stage')} stage")
            return
        print(f"\nbaseline trace -> {traces}/"
              f"{trace_name(BASELINE_TILE)}; now run:  chia_dse.py run {k}")
        return

    if what == "smoke":


        ref = evaluate_point.chia_remote(
            ACCELSIM, CUDA, "base", "base/smoke",
            f"{traces}/base/traces/kernelslist.g",
            f"{ROOT}/out_chia_smoke/base_adaptive_off",
            {**ADAPTIVE_OFF, **sm_knobs(108)}, True, None, False, 10800)
        r = ray.get(ref)
        print(json.dumps({x: r[x] for x in
                          ("ok", "cycles", "power_w", "joules", "cached",
                           "unapplied", "note")}, indent=1))
        return

    ft = forge_tiles()
    print(f"forge tiles entering the grid: {list(ft) or 'none'}")

    cfg = LoopConfig(
        accelsim_root=ACCELSIM, cuda_path=CUDA, cutlass_root=CUTLASS,
        template=f"{ROOT}/src/gemm_tpl.cu", trace_root=traces,
        out_root=out, work_dir=f"{ROOT}/build_chia_k{k}", m=4096, n=4096, k=k,
        agent_step4=os.environ.get("DSE_AGENT", "1") != "0",
        agent_sim_budget=int(os.environ.get("DSE_AGENT_BUDGET", "0")),
        forge_tiles=ft,
        area_budget_mm2=float(
            os.environ.get("DSE_AREA_BUDGET", "0")))
    j = codesign_loop(cfg)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/journal.json", "w") as fh:
        json.dump(j, fh, indent=1, default=str)
    print(f"\njournal -> {out}/journal.json")


if __name__ == "__main__":
    main()
