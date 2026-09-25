#!/usr/bin/env /usr/bin/python3.10
"""Gate / probe / grid the 4096x4096x64 shape.

Same M*N*K as the 1024^3 study (1.07e9) so mainloop cost is unchanged, but the
working set is 66 MiB against a 40 MiB L2 -- so the memory hierarchy is
actually exercised -- and block counts are 512-4096 against 108 SMs, which puts
the SM-count probes in the throughput regime instead of the CTA tail.

    run4k.py gate          admission-gate every traced tile
    run4k.py probe         probe wave on the stock tile
    run4k.py grid <name>   hardware grid for one tile
    run4k.py starve [tiles] downward probes: what does *removing* each
                            resource cost? The upward wave only says whether
                            more of X helps; an iso-area reallocation needs to
                            know whether less of X hurts, and those are
                            different questions with different answers.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

import ray

from chia.simulators.accelsim_dse import (
    CUTLASS_PROBES,
    licensed_dimensions,
    build_grid,
    cells_from_records,
    format_grid,
    pareto_front,
)
from chia.simulators.accelsim_probes import (
    STARVE_PROBES,
    format_probe_report,
    probe_report,
)

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"


TRACES = os.environ.get("DSE_TRACES", f"{ROOT}/traces4k")
OUT = os.environ.get("DSE_OUT", f"{ROOT}/out4k")
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
N_SLOTS = 48

TILES = ["base", "t128x128x32_s3", "t128x128x16_s3", "t128x128x16_s4",
         "t256x128x16_s3", "t128x64x16_s6", "t64x128x16_s6", "t64x64x16_s10"]
NOPOWER = {"-power_simulation_enabled": "0"}


def kl(name: str) -> str:
    return f"{TRACES}/{name}/traces/kernelslist.g"


@ray.remote(num_cpus=1)
def sim(kernel: str, point: str, kernelslist: str, outdir: str,
        knobs: dict, power: bool, trace_knobs: dict | None = None) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    from chia.simulators.accelsim_power import run_with_power
    t0 = time.time()
    try:
        r, v, s = run_with_power(
            AccelSimNode(ACCELSIM, CUDA), kernelslist, outdir=outdir,
            knob_overrides=dict(knobs), sync_xml=power,
            trace_knob_overrides=dict(trace_knobs or {}),
            collect_counters=False, per_kernel=False, timeout_s=10800,
        )
    except Exception as exc:
        return {"kernel": kernel, "point": point, "ok": False,
                "note": f"{type(exc).__name__}: {exc}"[:160],
                "wall_s": time.time() - t0}
    st = r.stats
    ok = bool(r.ok) and bool(st.get("cycles"))
    if power and not v.avg_power_w:
        ok = False
    log = os.path.join(outdir, "sim.log")
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    if "deadlock detected" in txt:
        ok = False
    return {
        "kernel": kernel, "point": point, "ok": ok,
        "cycles": st.get("cycles"), "insts": st.get("insts"),
        "occupancy": st.get("occupancy"), "issued_ctas": st.get("issued_ctas"),
        "cta_per_core": st.get("cta_per_core"),
        "dram_reads": st.get("dram_reads"), "dram_writes": st.get("dram_writes"),
        "l2_bw_gbps": st.get("l2_bw_gbps"),
        "power_w": v.avg_power_w if power else None,
        "components": v.components if power else {},
        "inert_power": v.inert if power else [],
        "joules": s.joules if power else None,
        "edp": s.edp if power else None,
        "effective": {k: r.knobs.get(k) for k in knobs},
        "effective_trace": {k: r.trace_knobs.get(k) for k in (trace_knobs or {})},
        "wall_s": time.time() - t0,
        "note": "" if ok else ("deadlock" if "deadlock detected" in txt
                               else (r.stderr_tail or "no cycles")[:150]),
    }


def fan(jobs, desc):
    print(f"\n[{desc}] {len(jobs)} sims across {N_SLOTS} slots", flush=True)
    refs = [sim.remote(*j) for j in jobs]
    out, pend, n = [], list(refs), 0
    t0 = time.time()
    while pend:
        ready, pend = ray.wait(pend, num_returns=1)
        for rf in ready:
            r = ray.get(rf); out.append(r); n += 1
            c = f"{r['cycles']:.0f}" if r.get("cycles") else "-"
            p = f"{r['power_w']:.2f}W" if r.get("power_w") else ""
            bad = "" if r["ok"] else f"  FAILED {r.get('note','')[:45]}"
            print(f"  [{n:>3}/{len(jobs)}] {r['point']:<24} cyc={c:>10} {p:>9}"
                  f" [{r['wall_s']:.0f}s]{bad}", flush=True)
    print(f"[{desc}] done in {time.time()-t0:.0f}s", flush=True)
    return out


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "gate"
    os.makedirs(OUT, exist_ok=True)
    ray.init(num_cpus=N_SLOTS + 8, num_gpus=8, ignore_reinit_error=True,
             log_to_driver=False,
             runtime_env={"env_vars":
                          {"PYTHONPATH": "/raid/sp56429/chia_hackathon/chia"}})

    if what == "gate":
        have = [t for t in TILES if os.path.exists(kl(t))]
        res = fan([(t, t, kl(t), f"{OUT}/gate/{t}", NOPOWER, False) for t in have],
                  "admission gate")
        ok = [r["kernel"] for r in res if r["ok"]]
        json.dump({"admitted": ok, "cells": res},
                  open(f"{OUT}/gate.json", "w"), indent=1)
        print("\nadmitted:", ok)
        for r in sorted(res, key=lambda r: r.get("cycles") or 9e18):
            if r["ok"]:
                print(f"  {r['kernel']:<18}{r['cycles']:>10.0f} cycles  "
                      f"occ={r.get('occupancy') or 0:>6.2f}%  "
                      f"ctas={r.get('issued_ctas') or 0:>6.0f}  "
                      f"dram_rd={r.get('dram_reads') or 0:>10.0f}")

    elif what == "probe":
        jobs = [("base", "baseline", kl("base"), f"{OUT}/probe/baseline",
                 NOPOWER, False)]
        for p in CUTLASS_PROBES:
            jobs.append(("base", p.name, kl("base"), f"{OUT}/probe/{p.name}",
                         {**p.knobs, **NOPOWER}, False, dict(p.trace_knobs)))
        res = fan(jobs, "probe wave @ 4096x4096x64")

        class R:
            def __init__(s, d):
                s.stats = {k: v for k, v in d.items()
                           if k in ("cycles", "insts", "occupancy",
                                    "issued_ctas", "cta_per_core") and v is not None}
                s.knobs = d.get("effective") or {}
                s.trace_knobs = d.get("effective_trace") or {}
                s.counters = {}; s.stall_mix = {}
                s.returncode = 0 if d["ok"] else 1
                s.timed_out = False; s.ok = d["ok"]; s.cycles = d.get("cycles")
                s.stderr_tail = d.get("note", ""); s.stdout_tail = ""
        by = {r["point"]: R(r) for r in res}
        by["baseline"].knobs = {"-gpgpu_n_clusters": "108",
                                "-gpgpu_n_cores_per_cluster": "1",
                                **by["baseline"].knobs}
        rep = probe_report(by, probes=CUTLASS_PROBES)
        import dataclasses
        ser = dict(rep)
        ser["ranked"] = [dataclasses.asdict(h) if dataclasses.is_dataclass(h) else h
                         for h in rep.get("ranked", [])]
        json.dump({"report": ser, "cells": res},
                  open(f"{OUT}/probe.json", "w"), indent=1, default=str)
        print("\n" + format_probe_report(rep))
        print("\nhw_targets:", rep.get("hw_targets"))

    elif what == "starve":
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
        probes = [p for p in STARVE_PROBES if not only or p.name in only]
        jobs = []
        for t in tiles:
            jobs.append((t, f"{t}/baseline", kl(t),
                         f"{OUT}/starve/{t}/baseline", NOPOWER, False))


            jobs.append((t, f"{t}/adaptive_off", kl(t),
                         f"{OUT}/starve/{t}/adaptive_off",
                         {"-gpgpu_adaptive_cache_config": "0", **NOPOWER}, False))
            for p in probes:
                jobs.append((t, f"{t}/{p.name}", kl(t),
                             f"{OUT}/starve/{t}/{p.name}",
                             {**p.knobs, **NOPOWER}, False, dict(p.trace_knobs)))
        res = fan(jobs, "starve wave @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/starve.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            base = (got.get("baseline") or {}).get("cycles")
            if not base:
                print(f"\n[{t}] no baseline -- nothing to normalise against")
                continue
            print(f"\n[{t}] baseline = {base:.0f} cycles")
            print(f"  {'probe':<20}{'cycles':>10}{'cost%':>9}"
                  f"{'vs ctrl%':>10}  verdict")
            rows = []
            for p in probes:
                r = got.get(p.name)
                if not r or not r.get("ok"):
                    print(f"  {p.name:<20}{'FAILED':>10}"
                          f"          {(r or {}).get('note','missing')[:40]}")
                    continue
                ref = (got.get(p.reference) or {}).get("cycles") or base
                cost = 100.0 * (r["cycles"] - base) / base
                ctrl = 100.0 * (r["cycles"] - ref) / ref


                bad = [k for k, v in p.knobs.items()
                       if str(r.get("effective", {}).get(k, "")).strip()
                       != str(v).strip()]
                verdict = ("INERT: " + ",".join(bad)) if bad else (
                    "free to surrender" if abs(ctrl) < 1.0 else "load-bearing")
                rows.append((p.name, ctrl, verdict))
                print(f"  {p.name:<20}{r['cycles']:>10.0f}{cost:>8.2f}%"
                      f"{ctrl:>9.2f}%  {verdict}")
            free = [n for n, c, v in rows if v == "free to surrender"]
            print(f"  FREE TO SURRENDER: {free}")

    elif what == "occ":


        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        SHM = "-gpgpu_shmem_size"
        SHMD = "-gpgpu_shmem_sizeDefault"
        SHMO = "-gpgpu_shmem_option"
        UNI = "-gpgpu_unified_l1d_size"
        REG = "-gpgpu_shader_registers"


        def _shm(kb: int) -> str:
            return str(kb * 1024)
        points = {


            "ctl_shmem_only": {UNI: "384", SHM: _shm(384), SHMD: _shm(384),
                               SHMO: "0,8,16,32,64,164,288,352"},
            "ctl_reg_only": {REG: "131072"},

            "occ2": {UNI: "384", SHM: _shm(384), SHMD: _shm(384),
                     SHMO: "0,8,16,32,64,164,288,352", REG: "131072"},
            "occ4": {UNI: "640", SHM: _shm(640), SHMD: _shm(640),
                     SHMO: "0,8,16,32,64,164,288,352,576,608", REG: "262144"},
        }
        jobs = []
        for t in tiles:
            jobs.append((t, f"{t}/baseline", kl(t),
                         f"{OUT}/occ/{t}/baseline", NOPOWER, False))
            for name, kn in points.items():
                jobs.append((t, f"{t}/{name}", kl(t),
                             f"{OUT}/occ/{t}/{name}", {**kn, **NOPOWER}, False))
        res = fan(jobs, "occupancy co-limitation @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/occ.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            base = (got.get("baseline") or {}).get("cycles")
            if not base:
                print(f"\n[{t}] no baseline")
                continue
            print(f"\n[{t}] baseline = {base:.0f} cycles, "
                  f"occ={(got.get('baseline') or {}).get('occupancy','?')}")
            print(f"  {'point':<18}{'cycles':>10}{'gain%':>9}"
                  f"{'occ%':>8}{'CTA/SM':>8}")
            for name in ("ctl_shmem_only", "ctl_reg_only", "occ2", "occ4"):
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<18}{'FAILED':>10}  "
                          f"{(r or {}).get('note','missing')[:40]}")
                    continue
                gain = 100.0 * (base - r["cycles"]) / base
                print(f"  {name:<18}{r['cycles']:>10.0f}{gain:>8.2f}%"
                      f"{r.get('occupancy', 0):>7.2f}{r.get('cta_per_core', 0):>8}")

    elif what == "iso":


        SHM, SHMD = "-gpgpu_shmem_size", "-gpgpu_shmem_sizeDefault"
        SHMO, UNI = "-gpgpu_shmem_option", "-gpgpu_unified_l1d_size"
        REG, NSM = "-gpgpu_shader_registers", "-gpgpu_n_clusters"
        OCC2 = {UNI: "384", SHM: str(384 * 1024), SHMD: str(384 * 1024),
                SHMO: "0,8,16,32,64,164,288,352", REG: "131072"}
        OCC4 = {UNI: "640", SHM: str(640 * 1024), SHMD: str(640 * 1024),
                SHMO: "0,8,16,32,64,164,288,352,576,608", REG: "262144"}
        points = {
            "stock": {},
            "isoA_occ2_sm104": {**OCC2, NSM: "104"},
            "isoA_detex_sm160": {NSM: "160"},
            "isoA_occ2_detex_sm151": {**OCC2, NSM: "151"},
            "isoA_occ4_detex_sm139": {**OCC4, NSM: "139"},
        }
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = []
        for t in tiles:
            for name, kn in points.items():
                jobs.append((t, f"{t}/{name}", kl(t),
                             f"{OUT}/iso/{t}/{name}", dict(kn), True))
        res = fan(jobs, "iso-area reallocation @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/iso.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            base = got.get("stock")
            if not base or not base.get("ok"):
                print(f"\n[{t}] no stock reference")
                continue
            bc = base["cycles"]
            be = base.get("energy_mj") or 0.0
            print(f"\n[{t}] stock = {bc:.0f} cycles, {be:.3f} mJ")


            print(f"  {'design':<24}{'cycles':>10}{'speedup':>9}"
                  f"{'W':>8}{'mJ':>9}{'EDP':>8}{'occ%':>7}{'CTA':>5}"
                  f"{'wave':>7}{'util':>7}")
            for name in points:
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<24}{'FAILED':>10}  "
                          f"{(r or {}).get('note','missing')[:38]}")
                    continue
                sp = bc / r["cycles"]
                e = r.get("energy_mj") or 0.0
                edp = (e * r["cycles"]) / (be * bc) if be and bc else 0.0
                ctas = r.get("issued_ctas") or 0
                nsm = int(points[name].get(NSM, 108))
                cpc = r.get("cta_per_core") or 1
                slots = max(1, nsm * int(cpc))
                waves = ctas / slots if ctas else 0.0
                import math
                util = (ctas / (math.ceil(waves) * slots)
                        if waves else 0.0)
                flag = " <- tail" if 0 < util < 0.8 else ""
                print(f"  {name:<24}{r['cycles']:>10.0f}{sp:>8.3f}x"
                      f"{r.get('power_w', 0):>8.1f}{e:>9.3f}{edp:>8.3f}"
                      f"{r.get('occupancy', 0):>7.2f}{int(cpc):>5}"
                      f"{waves:>7.2f}{util:>7.2f}{flag}")

    elif what == "mq":


        def _dl1(missq: int, wr_alloc: str = "L") -> str:
            return f"S:4:128:64,L:T:m:{wr_alloc}:L,A:512:64,{missq}:0,32"

        def _dl2(missq: int, wr_alloc: str = "L") -> str:
            return f"S:128:128:16,L:B:m:{wr_alloc}:X,A:192:4,{missq}:0,32"

        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"


        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        points = {
            "adaptive_off": dict(AOFF),
            "mq_l1_4x": {**AOFF, D1: _dl1(64)},
            "mq_l2_4x": {D2: _dl2(128)},
            "mq_both_4x": {**AOFF, D1: _dl1(64), D2: _dl2(128)},
            "mq_both_16x": {**AOFF, D1: _dl1(256), D2: _dl2(512)},


            "l1_no_write_alloc": {**AOFF, D1: _dl1(16, "N")},
            "l2_no_write_alloc": {D2: _dl2(32, "N")},
        }
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = []
        for t in tiles:
            jobs.append((t, f"{t}/baseline", kl(t),
                         f"{OUT}/mq/{t}/baseline", NOPOWER, False))
            for name, kn in points.items():
                jobs.append((t, f"{t}/{name}", kl(t),
                             f"{OUT}/mq/{t}/{name}", {**kn, **NOPOWER}, False))
        res = fan(jobs, "miss-queue depth @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/mq.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            base = (got.get("baseline") or {}).get("cycles")
            if not base:
                print(f"\n[{t}] no baseline")
                continue
            print(f"\n[{t}] baseline = {base:.0f} cycles")
            print(f"  {'point':<22}{'cycles':>10}{'gain%':>9}{'vs ctrl%':>10}")
            for name in points:
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<22}{'FAILED':>10}  "
                          f"{(r or {}).get('note','missing')[:38]}")
                    continue


                ref = ((got.get("adaptive_off") or {}).get("cycles") or base
                       if "-gpgpu_adaptive_cache_config" in points[name]
                       else base)
                print(f"  {name:<22}{r['cycles']:>10.0f}"
                      f"{100*(base-r['cycles'])/base:>8.2f}%"
                      f"{100*(ref-r['cycles'])/ref:>9.2f}%")

    elif what == "smsweep":


        def _dl1(q): return f"S:4:128:64,L:T:m:L:L,A:512:64,{q}:0,32"
        def _dl2(q): return f"S:128:128:16,L:B:m:L:X,A:192:4,{q}:0,32"
        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        NSM = "-gpgpu_n_clusters"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}
        SMS = list(range(150, 171, 2))
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["t128x128x16_s4"])
        points = {}
        for n in SMS:
            points[f"sm{n}_stockq"] = {**AOFF, NSM: str(n)}
            points[f"sm{n}_mq16"] = {**AOFF, **MQ16, NSM: str(n)}
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/smsweep/{t}/{n}",
                 {**kn, **NOPOWER}, False)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "SM-count sweep @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/smsweep.json", "w"), indent=1)

        import math
        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            print(f"\n[{t}]")
            print(f"  {'SMs':>5}{'gcd40':>7}{'waves':>7}{'tail%':>7}"
                  f"{'stock q':>10}{'mq16':>10}{'queue':>9}")
            for n in SMS:
                a = got.get(f"sm{n}_stockq")
                b = got.get(f"sm{n}_mq16")
                if not (a and b and a.get("ok") and b.get("ok")):
                    print(f"  {n:>5}{'FAILED':>40}")
                    continue
                ac, bc = a["cycles"], b["cycles"]
                slots = n * 2
                w = math.ceil(1024 / slots)
                tail = 1024 - (w - 1) * slots
                print(f"  {n:>5}{math.gcd(n, 40):>7}{w:>7}"
                      f"{100 * tail / slots:>6.0f}%{ac:>10.0f}{bc:>10.0f}"
                      f"{100 * (ac - bc) / ac:>8.2f}%")
            for q in ("stockq", "mq16"):
                v = [got[f"sm{n}_{q}"]["cycles"] for n in SMS
                     if got.get(f"sm{n}_{q}", {}).get("ok")]
                if v:
                    lo = min(v)
                    best = [n for n in SMS
                            if got.get(f"sm{n}_{q}", {}).get("cycles") == lo]
                    print(f"  [{q}] best {lo:.0f} at {best} SMs, "
                          f"spread {100 * (max(v) - lo) / lo:.2f}%")

    elif what == "smloc":


        def _dl1(q): return f"S:4:128:64,L:T:m:L:L,A:512:64,{q}:0,32"
        def _dl2(q): return f"S:128:128:16,L:B:m:L:X,A:192:4,{q}:0,32"
        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        NSM = "-gpgpu_n_clusters"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}
        SMS = [104, 106, 108, 110, 112]
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        points = {}
        for n in SMS:
            points[f"sm{n}_stockq"] = {**AOFF, NSM: str(n)}
            points[f"sm{n}_mq16"] = {**AOFF, **MQ16, NSM: str(n)}
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/smloc/{t}/{n}",
                 {**kn, **NOPOWER}, False)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "local SM robustness 104-112 @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/smloc.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            print(f"\n[{t}]")
            print(f"  {'SMs':>5}{'stock q':>10}{'mq16':>10}"
                  f"{'queue':>9}{'cta/SM':>8}")
            gains = []
            for n in SMS:
                a, b = got.get(f"sm{n}_stockq"), got.get(f"sm{n}_mq16")
                if not (a and b and a.get("ok") and b.get("ok")):
                    print(f"  {n:>5}{'FAILED':>10}")
                    continue
                ac, bc = a["cycles"], b["cycles"]
                g = 100.0 * (ac - bc) / ac
                gains.append(g)
                print(f"  {n:>5}{ac:>10.0f}{bc:>10.0f}{g:>8.2f}%"
                      f"{b.get('cta_per_core', 0):>8.1f}")
            for q in ("stockq", "mq16"):
                v = [got[f"sm{n}_{q}"]["cycles"] for n in SMS
                     if got.get(f"sm{n}_{q}", {}).get("ok")]
                if v:
                    print(f"  [{q}] cycles spread over 104-112 SMs: "
                          f"{100 * (max(v) - min(v)) / min(v):.2f}%")
            if gains:
                lo, hi = min(gains), max(gains)
                print(f"  [queue gain] {lo:+.2f}% .. {hi:+.2f}% "
                      f"(mean {sum(gains) / len(gains):+.2f}%)")
                print("  [verdict] " + ("LOCAL EFFECT ROBUST: queue gain "
                                        "positive at every SM count in band"
                                        if lo > 0 else
                                        "NOT ROBUST: queue gain changes sign "
                                        "within 8 SMs of stock"))

    elif what == "wbq":


        def _dl1(q): return f"S:4:128:64,L:T:m:L:L,A:512:64,{q}:0,32"
        def _dl2(q): return f"S:128:128:16,L:B:m:L:X,A:192:4,{q}:0,32"
        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        PW = "-gpgpu_pipeline_widths"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}

        def widths(wb):
            v = [4] * 13
            v[10] = wb
            return {PW: ",".join(str(x) for x in v)}

        WBS = [8, 4, 2, 1]
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        points = {}
        for wb in WBS:
            points[f"wb{wb}_stockq"] = {**AOFF, **widths(wb)}
            points[f"wb{wb}_mq16"] = {**AOFF, **MQ16, **widths(wb)}
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/wbq/{t}/{n}",
                 {**kn, **NOPOWER}, False)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "writeback x miss-queue @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/wbq.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            print(f"\n[{t}]")
            print(f"  {'EX_WB':>6}{'stock q':>10}{'mq16':>10}"
                  f"{'queue':>9}{'wb gain':>9}")
            ref = {}
            for wb in WBS:
                a, b = got.get(f"wb{wb}_stockq"), got.get(f"wb{wb}_mq16")
                if not (a and b and a.get("ok") and b.get("ok")):
                    print(f"  {wb:>6}{'FAILED':>10}")
                    continue
                ac, bc = a["cycles"], b["cycles"]
                ref[wb] = (ac, bc)
                base_ac = ref.get(8, (ac, bc))[0]
                print(f"  {wb:>6}{ac:>10.0f}{bc:>10.0f}"
                      f"{100 * (ac - bc) / ac:>8.2f}%"
                      f"{100 * (base_ac - ac) / base_ac:>8.2f}%")


            if 8 in ref and 4 in ref:
                s8, q8 = ref[8]
                s4, q4 = ref[4]
                wb_alone = 100 * (s8 - s4) / s8
                q_alone = 100 * (s8 - q8) / s8
                both = 100 * (s8 - q4) / s8
                best_single = max(wb_alone, q_alone)
                print(f"  [compose] wb4 alone {wb_alone:+.2f}%, "
                      f"mq16 alone {q_alone:+.2f}%, both {both:+.2f}%")
                if both > best_single + 1.0:
                    print("  [compose] INDEPENDENT: both beats the better "
                          "single by >1pp -- writeback is a separate axis")
                elif both < best_single - 1.0:
                    print("  [compose] INTERFERING: both is worse than the "
                          "better single alone")
                else:
                    print("  [compose] REDUNDANT: both ~= better single -- "
                          "same bottleneck reached two ways, as predicted")

    elif what == "band":


        def _dl1(q): return f"S:4:128:64,L:T:m:L:L,A:512:64,{q}:0,32"
        def _dl2(q): return f"S:128:128:16,L:B:m:L:X,A:192:4,{q}:0,32"
        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        PW, NSM = "-gpgpu_pipeline_widths", "-gpgpu_n_clusters"
        UNI, SHM = "-gpgpu_unified_l1d_size", "-gpgpu_shmem_size"
        SHMD, SHMO = "-gpgpu_shmem_sizeDefault", "-gpgpu_shmem_option"
        REG = "-gpgpu_shader_registers"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}
        OCC2 = {UNI: "384", SHM: str(384 * 1024), SHMD: str(384 * 1024),
                SHMO: "0,8,16,32,64,164,288,352", REG: "131072"}
        WB4 = {PW: ",".join(["4"] * 13)}
        SMS = [104, 106, 108, 110, 112]
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])


        VARIANTS = {"stockq": {}, "mq16": MQ16, "wb4": WB4,
                    "occ2_mq16": {**OCC2, **MQ16}, "wb4_mq16": {**WB4, **MQ16}}
        prior = {}
        try:
            for r in json.load(open(f"{OUT}/smloc.json"))["cells"]:
                if r.get("ok"):
                    prior[r["point"]] = r["cycles"]
            print(f"  (reusing {len(prior)} cached points from smloc.json)")
        except FileNotFoundError:
            print("  (no smloc.json cache -- simulating every column)")
        jobs = [(t, f"{t}/sm{n}_{v}", kl(t), f"{OUT}/band/{t}/sm{n}_{v}",
                 {**AOFF, **kn, NSM: str(n), **NOPOWER}, False)
                for t in tiles for n in SMS for v, kn in VARIANTS.items()
                if f"{t}/sm{n}_{v}" not in prior]
        res = fan(jobs, "band 104-112 x {stockq, mq16, wb4, occ2_mq16, wb4_mq16}")
        json.dump({"cells": res}, open(f"{OUT}/band.json", "w"), indent=1)

        got = {r["point"]: r["cycles"] for r in res if r.get("ok")}
        got.update(prior)

        for t in tiles:
            print(f"\n[{t}]")
            cols = list(VARIANTS)
            print("  " + f"{'SMs':>5}" + "".join(f"{c:>11}" for c in cols))
            for n in SMS:
                cells = [got.get(f"{t}/sm{n}_{c}") for c in cols]
                print("  " + f"{n:>5}" + "".join(
                    f"{c:>11.0f}" if c else f"{'--':>11}" for c in cells))
            ref = [got.get(f"{t}/sm{n}_stockq") for n in SMS]
            print(f"  {'claim':<22}{'mean':>8}{'min':>9}{'max':>9}  stable")
            for c in cols[1:]:
                v = [100 * (r - got[f"{t}/sm{n}_{c}"]) / r
                     for n, r in zip(SMS, ref)
                     if r and got.get(f"{t}/sm{n}_{c}")]
                if not v:
                    print(f"  {c:<22}{'no data':>26}")
                    continue
                lo, hi = min(v), max(v)


                verdict = ("YES" if lo > 0 else
                           "YES (stable LOSS)" if hi < 0 else
                           "NO -- straddles zero")
                print(f"  {c + ' vs stock q':<22}{sum(v) / len(v):>+7.2f}%"
                      f"{lo:>+8.2f}%{hi:>+8.2f}%  {verdict}")

    elif what == "smq":


        def _dl1(missq: int) -> str:
            return f"S:4:128:64,L:T:m:L:L,A:512:64,{missq}:0,32"

        def _dl2(missq: int) -> str:
            return f"S:128:128:16,L:B:m:L:X,A:192:4,{missq}:0,32"

        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        NSM = "-gpgpu_n_clusters"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}
        SMS = (156, 158, 160)
        points = {}
        for n in SMS:
            points[f"sm{n}_stockq"] = {**AOFF, NSM: str(n)}
            points[f"sm{n}_mq16"] = {**AOFF, **MQ16, NSM: str(n)}
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/smq/{t}/{n}", kn, False)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "SM count x queue depth @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/smq.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}

            def cyc(n, q):
                r = got.get(f"sm{n}_{q}")
                return r["cycles"] if r and r.get("ok") else None

            print(f"\n[{t}]")
            print(f"  {'SMs':>5}{'stock q':>10}{'mq16':>10}"
                  f"{'queue gain':>12}")
            for n in SMS:
                a, b = cyc(n, "stockq"), cyc(n, "mq16")
                if a is None or b is None:
                    print(f"  {n:>5}{'FAILED':>22}")
                    continue
                print(f"  {n:>5}{a:>10.0f}{b:>10.0f}"
                      f"{100 * (a - b) / a:>11.2f}%")
            for q in ("stockq", "mq16"):
                vals = [cyc(n, q) for n in SMS]
                if all(v is not None for v in vals):
                    lo, hi = min(vals), max(vals)
                    print(f"  spread across {SMS[0]}-{SMS[-1]} SMs "
                          f"[{q}]: {100 * (hi - lo) / lo:.2f}% "
                          f"({lo:.0f}-{hi:.0f})")

    elif what == "isomq":


        def _dl1(missq: int) -> str:
            return f"S:4:128:64,L:T:m:L:L,A:512:64,{missq}:0,32"

        def _dl2(missq: int) -> str:
            return f"S:128:128:16,L:B:m:L:X,A:192:4,{missq}:0,32"

        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        NSM = "-gpgpu_n_clusters"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}
        points = {
            "stock": {},
            "adaptive_off": dict(AOFF),
            "isoA_detex_sm160": {**AOFF, NSM: "160"},
            "isoA_mq16_sm158": {**AOFF, **MQ16, NSM: "158"},
            "isoA_mq16_sm107": {**AOFF, **MQ16, NSM: "107"},
        }
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/isomq/{t}/{n}", kn, True)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "iso-area: miss queue + SMs @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/isomq.json", "w"), indent=1)

        CLK = 1.41e9
        AREA = {"stock": 4866.95, "adaptive_off": 4866.95,
                "isoA_detex_sm160": 4859.0, "isoA_mq16_sm158": 4860.9,
                "isoA_mq16_sm107": 4855.1}
        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            ref = got.get("stock") or {}
            rc = ref.get("cycles")
            if not rc:
                print(f"\n[{t}] no stock reference")
                continue
            re_ = rc / CLK * ref.get("power_w", 0.0) * 1000.0
            redp = rc * re_
            print(f"\n[{t}] stock = {rc:.0f} cyc, {ref.get('power_w', 0):.1f} W,"
                  f" {re_:.3f} mJ, {AREA['stock']:.0f} mm^2")
            print(f"  {'design':<20}{'SM':>4}{'cycles':>9}{'speedup':>9}"
                  f"{'W':>8}{'mJ':>8}{'EDP':>7}{'mm^2':>8}{'CTA':>5}")
            for name in points:
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<20}{'FAILED':>13}  "
                          f"{(r or {}).get('note', 'missing')[:36]}")
                    continue
                c = r["cycles"]
                e = c / CLK * r.get("power_w", 0.0) * 1000.0
                nsm = int(points[name].get(NSM, "108"))
                print(f"  {name:<20}{nsm:>4}{c:>9.0f}{rc / c:>8.3f}x"
                      f"{r.get('power_w', 0):>8.1f}{e:>8.3f}"
                      f"{c * e / redp if redp else 0:>7.3f}"
                      f"{AREA.get(name, 0):>8.0f}{r.get('cta_per_core', 0):>5.0f}")
            a = (got.get("isoA_detex_sm160") or {}).get("cycles")
            b = (got.get("isoA_mq16_sm158") or {}).get("cycles")
            if a and b:
                print(f"  [control] SMs-only(160)={a:.0f} vs "
                      f"mq16+SMs(158)={b:.0f}  -> queue is worth "
                      f"{100 * (a - b) / a:+.2f}% at equal area")

    elif what == "mqinf":


        def _dl1(missq: int) -> str:
            return f"S:4:128:64,L:T:m:L:L,A:512:64,{missq}:0,32"

        def _dl2(missq: int) -> str:
            return f"S:128:128:16,L:B:m:L:X,A:192:4,{missq}:0,32"

        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}

        points = {"mq_1x": dict(AOFF)}
        for mult in (2, 4, 8, 16, 64, 256):
            points[f"mq_{mult}x"] = {**AOFF, D1: _dl1(16 * mult),
                                     D2: _dl2(32 * mult)}
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = [(t, f"{t}/{n}", kl(t), f"{OUT}/mqinf/{t}/{n}",
                 {**kn, **NOPOWER}, False)
                for t in tiles for n, kn in points.items()]
        res = fan(jobs, "miss-queue depth -> infinity @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/mqinf.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            ctrl = (got.get("mq_1x") or {}).get("cycles")
            if not ctrl:
                print(f"\n[{t}] no control")
                continue
            print(f"\n[{t}] mq_1x (stock depth, adaptive off) = {ctrl:.0f}")
            print(f"  {'depth':<8}{'L1':>6}{'L2':>7}{'cycles':>10}"
                  f"{'gain%':>8}{'marginal%':>11}")
            prev = ctrl
            for name in points:
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<8}{'FAILED':>30}  "
                          f"{(r or {}).get('note', 'missing')[:34]}")
                    continue
                m = int(name[3:-1])
                c = r["cycles"]
                print(f"  {name:<8}{16 * m:>6}{32 * m:>7}{c:>10.0f}"
                      f"{100 * (ctrl - c) / ctrl:>7.2f}%"
                      f"{100 * (prev - c) / prev:>10.2f}%")
                prev = c

    elif what == "mqocc":


        REG = "-gpgpu_shader_registers"
        SHM, SHMD = "-gpgpu_shmem_size", "-gpgpu_shmem_sizeDefault"
        SHMO, UNI = "-gpgpu_shmem_option", "-gpgpu_unified_l1d_size"
        OCC2 = {UNI: "384", SHM: str(384 * 1024), SHMD: str(384 * 1024),
                SHMO: "0,8,16,32,64,164,288,352", REG: "131072"}
        OCC4 = {UNI: "640", SHM: str(640 * 1024), SHMD: str(640 * 1024),
                SHMO: "0,8,16,32,64,164,288,352,576,608", REG: "262144"}

        def _dl1(missq: int, wr_alloc: str = "L") -> str:
            return f"S:4:128:64,L:T:m:{wr_alloc}:L,A:512:64,{missq}:0,32"

        def _dl2(missq: int, wr_alloc: str = "L") -> str:
            return f"S:128:128:16,L:B:m:{wr_alloc}:X,A:192:4,{missq}:0,32"

        D1, D2 = "-gpgpu_cache:dl1", "-gpgpu_cache:dl2"
        AOFF = {"-gpgpu_adaptive_cache_config": "0"}
        MQ4 = {D1: _dl1(64), D2: _dl2(128)}
        MQ16 = {D1: _dl1(256), D2: _dl2(512)}


        points = {
            "adaptive_off": dict(AOFF),
            "occ2": {**AOFF, **OCC2},
            "mq16": {**AOFF, **MQ16},
            "occ2_mq4": {**AOFF, **OCC2, **MQ4},
            "occ2_mq16": {**AOFF, **OCC2, **MQ16},
            "occ4_mq16": {**AOFF, **OCC4, **MQ16},
        }
        tiles = (sys.argv[2].split(",") if len(sys.argv) > 2
                 else ["base", "t128x128x16_s4"])
        jobs = []
        for t in tiles:
            for name, kn in points.items():
                jobs.append((t, f"{t}/{name}", kl(t),
                             f"{OUT}/mqocc/{t}/{name}", kn, True))
        res = fan(jobs, "occupancy x miss-queue @ 4096x4096x64")
        json.dump({"cells": res}, open(f"{OUT}/mqocc.json", "w"), indent=1)

        for t in tiles:
            got = {r["point"].split("/", 1)[1]: r for r in res
                   if r["point"].startswith(f"{t}/")}
            ctrl = got.get("adaptive_off") or {}
            base = ctrl.get("cycles")
            if not base:
                print(f"\n[{t}] no control")
                continue
            bedp = base * ctrl.get("energy_mj", 0.0)
            print(f"\n[{t}] control (adaptive_off) = {base:.0f} cycles")
            print(f"  {'point':<14}{'cycles':>9}{'gain%':>8}{'W':>8}"
                  f"{'mJ':>9}{'EDP':>7}{'occ%':>7}{'CTA':>5}")
            for name in points:
                r = got.get(name)
                if not r or not r.get("ok"):
                    print(f"  {name:<14}{'FAILED':>9}  "
                          f"{(r or {}).get('note', 'missing')[:44]}")
                    continue
                edp = r["cycles"] * r.get("energy_mj", 0.0)
                print(f"  {name:<14}{r['cycles']:>9.0f}"
                      f"{100 * (base - r['cycles']) / base:>7.2f}%"
                      f"{r.get('power_w', 0.0):>8.2f}"
                      f"{r.get('energy_mj', 0.0):>9.4f}"
                      f"{(edp / bedp if bedp else 0):>7.3f}"
                      f"{r.get('occupancy', 0.0):>7.2f}"
                      f"{r.get('cta_per_core', 0):>5}")

            a = (got.get("occ2") or {}).get("cycles")
            b = (got.get("mq16") or {}).get("cycles")
            c = (got.get("occ2_mq16") or {}).get("cycles")
            if a and b and c:
                best_part = min(a, b)
                print(f"  [compose] occ2={a:.0f} mq16={b:.0f} "
                      f"together={c:.0f}  "
                      f"{'SUPER-ADDITIVE' if c < best_part else 'no synergy'}"
                      f" ({100 * (best_part - c) / best_part:+.2f}% "
                      f"vs best single knob)")

    elif what == "grid":
        name = sys.argv[2]


        dims = (sys.argv[3].split(",") if len(sys.argv) > 3
                else licensed_dimensions(json.load(open(f"{OUT}/probe.json"))["report"]))
        print(f"licensed dims: {dims}")
        grid = build_grid(dims)
        res = fan([(name, g.name, kl(name),
                    f"{OUT}/{name}/grid/{g.name.replace('/', '_')}", g.knobs,
                    True, g.trace_knobs)
                   for g in grid], f"grid: {name}")
        json.dump({"cells": res}, open(f"{OUT}/{name}_grid.json", "w"), indent=1)
        cells = cells_from_records(res)
        print("\n" + format_grid(cells))
        print("\nPARETO:")
        for c in pareto_front(cells):
            print(f"  {c.point:<22}{c.cycles:>10.0f} cyc  {c.joules*1e3:>9.4f} mJ")


if __name__ == "__main__":
    main()
