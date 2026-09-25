#!/usr/bin/env /usr/bin/python3.10
"""Full n_sm x l2 co-design surface at fixed clock, across every admitted tile.

The 19-point grid was cross-shaped: one dimension varied from the centre plus
two cross terms. That is cheap but it never crosses n_sm with l2, so the two
findings it produced (the sm216/sm432 preference is kernel-dependent; 2x L2
beats 4x) sit on separate arms with no cell where both move. And it spent a
third of its points on the clock axis, which is provably degenerate here
(EDP ~ f^-1.28, top bin always wins) and therefore carries no design
information.

This runs the surface those points should have bought: 4 SM counts x 4 L2
capacities at stock 1410 MHz, for every tile the gate admitted. n=7 kernels
instead of n=2.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_dse import HW_DIMENSIONS

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
TRACES = f"{ROOT}/traces4k"
OUT = f"{ROOT}/out4k/surface"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
N_SLOTS = 88
CLOCK = "clk1410"

SM = ["sm54", "sm108", "sm216", "sm432"]
L2 = ["l2_half", "l2_base", "l2_2x", "l2_4x"]


def kl(name: str) -> str:
    return f"{TRACES}/{name}/traces/kernelslist.g"


def point_knobs(sm: str, l2: str) -> dict:
    return {
        HW_DIMENSIONS["n_sm"]["knob"]: HW_DIMENSIONS["n_sm"]["values"][sm],
        HW_DIMENSIONS["clock"]["knob"]: HW_DIMENSIONS["clock"]["values"][CLOCK],
        HW_DIMENSIONS["l2"]["knob"]: HW_DIMENSIONS["l2"]["values"][l2],
        "-gpgpu_n_cores_per_cluster": "1",
    }


@ray.remote(num_cpus=1)
def sim(kernel: str, point: str, kernelslist: str, outdir: str, knobs: dict) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_power import run_with_power
    t0 = time.time()
    try:
        r, v, s = run_with_power(
            AccelSimNode(ACCELSIM, CUDA), kernelslist, outdir=outdir,
            knob_overrides=dict(knobs), sync_xml=True,
            collect_counters=False, per_kernel=False, timeout_s=14400,
        )
    except Exception as exc:
        return {"kernel": kernel, "point": point, "ok": False,
                "note": f"{type(exc).__name__}: {exc}"[:160],
                "wall_s": time.time() - t0}
    st = r.stats
    ok = bool(r.ok) and bool(st.get("cycles")) and bool(v.avg_power_w)
    log = os.path.join(outdir, "sim.log")
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    if "deadlock detected" in txt:
        ok = False

    applied = {k: r.knobs.get(k) for k in knobs}
    unapplied = [k for k, want in knobs.items()
                 if str(applied.get(k, "")).strip() != str(want).strip()]
    return {
        "kernel": kernel, "point": point, "ok": ok and not unapplied,
        "cycles": st.get("cycles"), "insts": st.get("insts"),
        "occupancy": st.get("occupancy"), "issued_ctas": st.get("issued_ctas"),
        "cta_per_core": st.get("cta_per_core"),
        "dram_reads": st.get("dram_reads"), "dram_writes": st.get("dram_writes"),
        "l2_bw_gbps": st.get("l2_bw_gbps"),
        "power_w": v.avg_power_w, "joules": s.joules, "edp": s.edp,
        "inert_power": v.inert, "unapplied": unapplied,
        "effective": applied, "wall_s": time.time() - t0,
        "note": "" if ok and not unapplied else (
            "deadlock" if "deadlock detected" in txt
            else (f"UNAPPLIED {unapplied}" if unapplied
                  else (r.stderr_tail or "no cycles")[:150])),
    }


def main() -> None:
    tiles = sys.argv[1:] or json.load(open(f"{ROOT}/out4k/gate.json"))["admitted"]
    tiles = [t for t in tiles if os.path.exists(kl(t))]
    os.makedirs(OUT, exist_ok=True)
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             runtime_env={"env_vars":
                          {"PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})

    jobs = []
    for t, sm, l2 in itertools.product(tiles, SM, L2):
        pt = f"{sm}/{CLOCK}/{l2}"
        jobs.append((t, pt, kl(t), f"{OUT}/{t}/{sm}_{l2}", point_knobs(sm, l2)))

    print(f"{len(tiles)} tiles x {len(SM)}x{len(L2)} hw points = {len(jobs)} sims"
          f" across {N_SLOTS} slots", flush=True)
    print(f"tiles: {tiles}", flush=True)

    refs = [sim.remote(*j) for j in jobs]
    out, pend, n = [], list(refs), 0
    t0 = time.time()
    while pend:
        ready, pend = ray.wait(pend, num_returns=1)
        for rf in ready:
            r = ray.get(rf); out.append(r); n += 1
            c = f"{r['cycles']:.0f}" if r.get("cycles") else "-"
            p = f"{r['power_w']:.1f}W" if r.get("power_w") else ""
            bad = "" if r["ok"] else f"  FAILED {r.get('note','')[:40]}"
            print(f"  [{n:>3}/{len(jobs)}] {r['kernel']:<18}{r['point']:<22}"
                  f" cyc={c:>10} {p:>8} [{r['wall_s']:.0f}s]{bad}", flush=True)

            json.dump({"cells": out}, open(f"{OUT}/surface.json", "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s  ok={sum(1 for r in out if r['ok'])}"
          f"/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
