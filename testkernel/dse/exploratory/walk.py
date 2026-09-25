#!/usr/bin/env /usr/bin/python3.10
"""Drive the HW/SW co-design walk on CUTLASS TF32 GEMM.

Every simulation is a Ray task holding one CPU slot. Traces are made once per
kernel on a GPU slot and replayed across the whole hardware grid, which is the
economics the whole design rests on: ~45 s of GPU buys a row of ~130 s CPU
cells that fan out across cores.

Usage:
    walk.py probe            round-0 probe wave on the baseline kernel
    walk.py grid             build + run the hardware grid for the baseline
    walk.py xmlcheck         verify power responds to the XML co-generator
    walk.py sw               build/trace/screen software candidates
    walk.py all              the whole walk
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim import AccelSimNode, config_fingerprint
from chia.simulators.accelsim_dse import (
    cells_from_records,
    format_grid,
    format_rows,
    BASELINE_TILE,
    CUTLASS_PROBES,
    Cell,
    KernelRow,
    Tile,
    build_grid,
    licensed_dimensions,
    pareto_front,
    resident_ctas,
    suggest_tiles,
)
from chia.simulators.accelsim_power import (
    core_clock_hz,
    edp,
    run_with_power,
)
from chia.simulators.accelsim_probes import format_probe_report, probe_report

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
CUTLASS = "/raid/sp56429/Accel-sim/sp_accelsim/final_workloads/cutlass_gemms/cutlass"
GEMM_N = 1024
N_SLOTS = 48

os.makedirs(f"{ROOT}/out", exist_ok=True)


def _node() -> AccelSimNode:
    return AccelSimNode(ACCELSIM, CUDA)


@ray.remote(num_cpus=1)
def sim_cell(kernel: str, point: str, kernelslist: str, outdir: str,
             knobs: dict, sync_xml: bool = True) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_power import run_with_power

    t0 = time.time()
    try:
        r, v, s = run_with_power(
            AccelSimNode(ACCELSIM, CUDA), kernelslist,
            outdir=outdir, knob_overrides=knobs, sync_xml=sync_xml,
            collect_counters=False, per_kernel=False,
        )
    except Exception as exc:
        return {"kernel": kernel, "point": point, "ok": False,
                "note": f"{type(exc).__name__}: {exc}"[:200], "wall_s": time.time() - t0}

    st = r.stats
    out = {
        "kernel": kernel, "point": point, "ok": bool(r.ok),
        "cycles": st.get("cycles"), "insts": st.get("insts"),
        "occupancy": st.get("occupancy"),
        "issued_ctas": st.get("issued_ctas"), "cta_per_core": st.get("cta_per_core"),
        "power_w": v.avg_power_w if v else None,
        "components": v.components if v else {},
        "inert_power": v.inert if v else [],
        "power_consistent": v.consistent if v else False,
        "joules": s.joules, "seconds": s.seconds, "edp": s.edp,
        "clock_hz": s.clock_hz,
        "knobs_effective": {k: r.knobs.get(k) for k in knobs},
        "cached": r.cached, "wall_s": time.time() - t0,
        "note": "" if r.ok else r.stderr_tail[:200],
    }
    want_power = knobs.get("-power_simulation_enabled", "1") != "0"
    if not out["cycles"] or (want_power and not out["power_w"]):
        out["ok"] = False
        out["note"] = out["note"] or (
            "no cycles" if not out["cycles"] else "power enabled but no report")
    return out


def fan_out(jobs: list[tuple], desc: str) -> list[dict]:
    """Dispatch every job at once and report as they land."""
    print(f"\n[{desc}] dispatching {len(jobs)} simulations across {N_SLOTS} slots")
    refs = [sim_cell.remote(*j) for j in jobs]
    done, results = 0, []
    pending = list(refs)
    t0 = time.time()
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        for rf in ready:
            res = ray.get(rf)
            results.append(res)
            done += 1
            flag = "" if res["ok"] else f"  FAILED {res.get('note','')[:60]}"
            cyc = f"{res['cycles']:.0f}" if res.get("cycles") else "-"
            pw = f"{res['power_w']:.2f}" if res.get("power_w") else "-"
            print(f"  [{done:>3}/{len(jobs)}] {res['point']:<28} "
                  f"cycles={cyc:>10} P={pw:>7}W [{res['wall_s']:.0f}s]{flag}",
                  flush=True)
    print(f"[{desc}] {len(results)} done in {time.time()-t0:.0f}s")
    return results


@ray.remote(num_cpus=2)
def build_tile(tile_json: str, n: int) -> dict:
    tile = Tile(**json.loads(tile_json))
    ok, why = tile.feasible()
    if not ok:
        return {"tile": tile.name, "ok": False, "stage": "feasibility", "note": why}
    out = f"{ROOT}/build/{tile.name}"
    cmd = ["/usr/local/cuda-12.8/bin/nvcc", "-arch=sm_80",
           f"-I{CUTLASS}/include", f"-I{CUTLASS}/tools/util/include",
           "-O3", "--expt-relaxed-constexpr", "--use_fast_math",
           *tile.defines(n), f"{ROOT}/src/gemm_tpl.cu", "-o", out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        return {"tile": tile.name, "ok": False, "stage": "nvcc",
                "note": p.stderr[-400:], "wall_s": time.time() - t0}
    return {"tile": tile.name, "ok": True, "bin": out, "wall_s": time.time() - t0}


@ray.remote(num_cpus=1, num_gpus=1)
def trace_tile(tile_name: str, binary: str) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    node = AccelSimNode(ACCELSIM, CUDA)
    t0 = time.time()


    p = subprocess.run([binary], capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not p.stdout.startswith("OK "):
        return {"tile": tile_name, "ok": False, "stage": "correctness",
                "note": (p.stdout + p.stderr)[:200]}
    r = AccelSimNode.trace_app(node, app_cmd=[binary],
                               trace_dir=f"{ROOT}/traces/{tile_name}", cwd=ROOT)
    return {"tile": tile_name, "ok": bool(r.ok), "stage": "trace",
            "kernelslist": r.kernelslist, "est_insts": r.est_thread_insts,
            "ref": p.stdout.strip(), "wall_s": time.time() - t0}


def stage_probe(kernelslist: str, kernel: str = "baseline") -> dict:
    """Probe wave, deliberately **without** the power model.

    Two reasons, and the first is not about speed. A probe is an
    *idealization*: infinite L1, zero DRAM latency, a 4x register file. Those
    are not buildable machines, so AccelWattch's watts for them are fiction --
    McPAT would dutifully price a cache geometry that cannot exist. Probes
    exist to bound *cycles*, which is what prunes the hardware axis; EDP is
    measured on the grid, where every point is a real configuration.

    The speedup is a bonus: power modelling costs ~3x wall-clock per cell, so
    dropping it takes the wave from ~20 min to ~5.
    """
    nopow = {"-power_simulation_enabled": "0"}
    jobs = [(kernel, "baseline", kernelslist,
             f"{ROOT}/out/{kernel}/probe/baseline", dict(nopow), False)]
    for p in CUTLASS_PROBES:
        jobs.append((kernel, p.name, kernelslist,
                     f"{ROOT}/out/{kernel}/probe/{p.name}",
                     {**p.knobs, **nopow}, False))
    results = fan_out(jobs, f"probe wave: {kernel}")


    class _R:
        def __init__(self, d):
            self.stats = {k: v for k, v in d.items()
                          if k in ("cycles", "insts", "occupancy", "issued_ctas",
                                   "cta_per_core") and v is not None}
            self.knobs = d.get("knobs_effective") or {}
            self.trace_knobs = {}
            self.returncode = 0 if d["ok"] else 1
            self.timed_out = False
            self.counters = {}
            self.ok = d["ok"]
            self.cycles = d.get("cycles")
            self.stall_mix = {}


            self.stderr_tail = d.get("note", "")
            self.stdout_tail = ""

    by_name = {r["point"]: _R(r) for r in results}

    base_knobs = next((r["knobs_effective"] for r in results
                       if r["point"] == "baseline"), {})
    by_name["baseline"].knobs = {
        "-gpgpu_n_clusters": "108", "-gpgpu_n_cores_per_cluster": "1", **base_knobs}

    rep = probe_report(by_name, probes=CUTLASS_PROBES)
    json.dump({"report": rep, "cells": results},
              open(f"{ROOT}/out/{kernel}/probe.json", "w"), indent=1, default=str)
    return {"report": rep, "cells": results}


def stage_grid(kernelslist: str, report: dict, kernel: str = "baseline") -> list[dict]:
    dims = licensed_dimensions(report)
    grid = build_grid(dims)
    print(f"\nlicensed dimensions: {dims}")
    print(f"grid: {len(grid)} points")
    jobs = [(kernel, g.name, kernelslist,
             f"{ROOT}/out/{kernel}/grid/{g.name.replace('/', '_')}", g.knobs, True)
            for g in grid]
    results = fan_out(jobs, f"hw grid: {kernel}")
    json.dump({"dims": dims, "cells": results},
              open(f"{ROOT}/out/{kernel}/grid.json", "w"), indent=1, default=str)
    return results


TRACED = {
    "base": BASELINE_TILE,
    "tb128x128x16_s3": Tile((128, 128, 16), (2, 2, 1), 3),
    "tb128x128x32_s3": Tile((128, 128, 32), (2, 2, 1), 3),
    "tb128x64x32_s3": Tile((128, 64, 32), (2, 2, 1), 3),
    "tb64x128x32_s3": Tile((64, 128, 32), (2, 2, 1), 3),
    "tb64x64x16_s10": Tile((64, 64, 16), (2, 2, 1), 10),
    "tb64x64x32_s5": Tile((64, 64, 32), (2, 2, 1), 5),
}


def _kl(name: str) -> str:
    return f"{ROOT}/traces/{name}/traces/kernelslist.g"


@ray.remote(num_cpus=1)
def gate_cell(name: str, kernelslist: str, outdir: str) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_dse import simulable
    t0 = time.time()
    ok, why = simulable(AccelSimNode(ACCELSIM, CUDA), kernelslist, outdir)
    return {"kernel": name, "ok": ok, "why": why, "wall_s": time.time() - t0}


def stage_gate(names: list[str]) -> list[str]:
    """Admit only traces that actually simulate.

    Runs before any probe wave or grid row. A trace that deadlocks costs one
    simulation here instead of one per grid cell, and the failure mode is
    invisible to the correctness gate -- an align-1 CUTLASS build computes the
    right answer and wedges the memory model every time.
    """
    ready = [n for n in names if os.path.exists(_kl(n))]
    missing = [n for n in names if n not in ready]
    if missing:
        print(f"  no trace yet, skipping: {missing}")
    refs = {n: gate_cell.remote(n, _kl(n), f"{ROOT}/gate/{n}") for n in ready}
    admitted = []
    print(f"\n[admission gate] {len(refs)} traces, no-power simulation each")
    for n, rf in refs.items():
        r = ray.get(rf)
        mark = "ADMIT" if r["ok"] else "REJECT"
        print(f"  {mark:<7}{n:<22}{r['why']}  [{r['wall_s']:.0f}s]", flush=True)
        if r["ok"]:
            admitted.append(n)
    json.dump({"admitted": admitted}, open(f"{ROOT}/out/gate.json", "w"))
    return admitted


def stage_screen(names: list[str]) -> list[dict]:
    """One simulation per candidate at the baseline hardware point.

    A full row per candidate costs ~len(grid) sims; this costs one. The
    approximation is that a kernel's ranking at baseline hardware predicts its
    ranking at its own best hardware -- which the survivors' full rows then
    test, since if ranking were strongly hardware-dependent the screen and the
    rows would disagree.
    """
    jobs = [(n, "screen", _kl(n), f"{ROOT}/out/{n}/screen", {}, True)
            for n in names if os.path.exists(_kl(n))]
    results = fan_out(jobs, "sw screen @ baseline hw")
    for r in results:
        r["point"] = r["kernel"]
    json.dump(results, open(f"{ROOT}/out/screen.json", "w"), indent=1, default=str)
    return results


def stage_rows(names: list[str], report: dict) -> dict:
    """Full hardware grid for each surviving kernel."""
    dims = licensed_dimensions(report)
    grid = build_grid(dims)
    jobs = []
    for n in names:
        for g in grid:
            jobs.append((n, g.name, _kl(n),
                         f"{ROOT}/out/{n}/grid/{g.name.replace('/', '_')}",
                         g.knobs, True))
    results = fan_out(jobs, f"rows: {len(names)} kernels x {len(grid)} points")
    json.dump({"dims": dims, "cells": results},
              open(f"{ROOT}/out/rows.json", "w"), indent=1, default=str)
    return {"dims": dims, "grid": [g.name for g in grid], "cells": results}


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})

    base_kl = f"{ROOT}/traces/base/traces/kernelslist.g"
    assert os.path.exists(base_kl), f"no baseline trace at {base_kl}"

    if what in ("gate", "all"):
        stage_gate(list(TRACED))
    if what in ("probe", "all"):
        admitted = (json.load(open(f"{ROOT}/out/gate.json"))["admitted"]
                    if os.path.exists(f"{ROOT}/out/gate.json") else ["base"])
        assert "base" in admitted, "baseline trace did not pass the admission gate"
        pr = stage_probe(base_kl)
        print("\n" + format_probe_report(pr["report"]))
    if what in ("grid", "all"):
        pr = json.load(open(f"{ROOT}/out/baseline/probe.json"))
        cells = stage_grid(base_kl, pr["report"])
        print("\n" + format_grid(cells_from_records(cells)))
    if what in ("screen", "sw", "all"):
        res = stage_screen(list(TRACED))
        ranked = sorted([r for r in res if r["ok"]], key=lambda r: r["edp"])
        print("\n" + format_grid(cells_from_records(ranked), baseline_point="base"))
        keep = [r["kernel"] for r in ranked[:3]]
        print(f"\nsurvivors -> full rows: {keep}")
        json.dump(keep, open(f"{ROOT}/out/survivors.json", "w"))
    if what in ("rows", "all"):
        pr = json.load(open(f"{ROOT}/out/baseline/probe.json"))
        keep = json.load(open(f"{ROOT}/out/survivors.json"))
        out = stage_rows(keep, pr["report"])
        rows = []
        for n in keep:
            cs = cells_from_records([c for c in out["cells"] if c["kernel"] == n])
            rows.append(KernelRow(kernel=n, tile=TRACED.get(n), cells=cs))
        print("\n" + format_rows(rows, baseline_kernel="base"))


if __name__ == "__main__":
    main()
