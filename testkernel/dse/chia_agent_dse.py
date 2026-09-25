#!/usr/bin/env python3
"""Run the *agentic* arm of the co-design loop.  This file holds no decisions.

``chia_dse.py`` runs the deterministic chain: every choice in it is a
threshold, and it is kept untouched so it can serve as the control.  This
driver runs the same problem with an LLM where those thresholds are.  Both
arms are handed the same measurements, both are verified by the same
three-arm sign-stability band, and the two journals have the same shape, so
the comparison is between two decision procedures and nothing else.

    chia_agent_dse.py smoke        serve the tool, prove the budget binds
    chia_agent_dse.py run [K]      measure, let the agent design, band it

``smoke`` does not invoke a model: it drives the tool's own surface directly,
which is the half of this that can go wrong silently.
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

from chia.simulators.accelsim_codesign import LoopConfig

ROOT = os.environ.get("DSE_ROOT", ROOT)
ACCELSIM = os.environ.get("ACCELSIM_ROOT", os.path.join(PROJECT, "accel-sim-framework"))
CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.8")
CUTLASS = os.environ.get(
    "CUTLASS_ROOT",
    os.path.join(PROJECT, "cutlass"))
N_SLOTS = 48
BUDGET = int(os.environ.get("AGENT_BUDGET", "16"))


def boot() -> None:
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             resources={"accelsim": float(N_SLOTS), "accelsim_gpu": 8.0},
             runtime_env={"env_vars": {
                 "PYTHONPATH": CHIA_ROOT}})


def cfg_for(k: int, traces: str, out: str) -> LoopConfig:
    return LoopConfig(
        accelsim_root=ACCELSIM, cuda_path=CUDA, cutlass_root=CUTLASS,
        template=f"{ROOT}/src/gemm_tpl.cu", trace_root=traces,
        out_root=out, work_dir=f"{ROOT}/build_agent_k{k}", m=4096, n=4096, k=k)


def smoke(k: int, traces: str) -> None:
    """Exercise the whole tool surface with a budget of 1, and no model.

    What this checks is not that a simulation runs -- ``chia_dse.py smoke``
    already covers dispatch -- but the parts that are new and would fail
    quietly: that the three free reads render, that an illegal knob string is
    refused with a usable reason rather than crashing the actor, that the
    budget actually binds, and that the ledger survives the actor boundary.
    """
    from chia.simulators.accelsim_agent import CoDesignTool
    import asyncio


    def sim(points):
        return asyncio.run(tool.simulate(points))

    out = os.environ.get("AGENT_OUT", f"{ROOT}/out_agent_smoke")
    tool = CoDesignTool(
        "codesign", accelsim_root=ACCELSIM, cuda_path=CUDA,
        trace_root=traces, out_root=out, m=4096, n=4096, k=k, budget=1)
    print(f"tool up at {tool.hostname}:{tool.port}\n")
    for nm in ("get_measurements", "knob_vocabulary", "tile_vocabulary"):
        print(f"--- {nm} " + "-" * (60 - len(nm)))
        print(getattr(tool, nm)())
        print()
    print("--- simulate: 1 legal, 1 illegal, 1 over budget " + "-" * 20)
    print(sim(["base/stock", "base/occ3", "base/queue16"]))
    print("\n--- propose_design " + "-" * 50)
    capped = tool.propose_design("base", "queue99999", "an absurd multiplier")
    print(f"  [capped]   {capped}")
    first = tool.propose_design("base", "queue4",
                                "smoke test: the first commitment")
    print(f"  [accepted] {first}")
    second = tool.propose_design("base", "stock", "a revision, after the fact")
    print(f"  [one-shot] {second}")
    print(f"  [closed]   {sim(['base/queue4'])}")

    led = tool.read_ledger()
    d = led["design"] or {}
    print(f"\nledger: spent={led['spent']} batches={len(led['batches'])} "
          f"design={d.get('tile')} + {d.get('knobs')}")


    sims = [r for b in led["batches"] for r in b.get("records", [])
            if r.get("ok")]
    charged_correctly = led["spent"] == sum(1 for r in sims
                                            if not r.get("cached"))
    checks = {
        "exactly one point was dispatched; two were refused free":
            len(sims) == 1,
        "budget was charged for the point that ran, not the one from cache":
            charged_correctly,
        "the batch survived the Ray actor boundary":
            len(led["batches"]) == 1,
        "the measured baseline is the recorded one (108,864 cycles)":
            any(r.get("cycles") == 108864
                for b in led["batches"] for r in b.get("records", [])),
        "an out-of-range multiplier is refused, not silently built":
            capped.startswith("'queue99999'"),
        "the first design commits":
            d.get("knobs") == "queue4",
        "a later proposal does not overwrite it":
            second.startswith("already committed"),
        "simulation stops once a design is committed":
            "run is closed" in sim(["base/wb4"]),
    }
    print()
    for why, got in checks.items():
        print(f"  [{'ok  ' if got else 'FAIL'}] {why}")
    ok = all(checks.values())
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    traces = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")
    boot()

    if what == "smoke":
        smoke(k, traces)
        return

    from chia.simulators.accelsim_agent import agent_codesign
    out = os.environ.get("AGENT_OUT", f"{ROOT}/out_agent_k{k}")
    j = agent_codesign(cfg_for(k, traces, out), budget=BUDGET, out_root=out,
                       timeout_seconds=int(os.environ.get("AGENT_TIMEOUT_S", 14400)))
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/journal.json", "w") as fh:
        json.dump(j, fh, indent=1, default=str)
    d = j.get("design")
    if d:
        print(f"\nagent arm: {d['tile']} + {d['knobs']}  "
              f"{d['verdict']} [{d['lo']:+.2f}%, {d['hi']:+.2f}%]  "
              f"quotable={d['quotable']}  ({d['sims_spent']}/{d['budget']} sims)")
    print(f"journal -> {out}/journal.json")


if __name__ == "__main__":
    main()
