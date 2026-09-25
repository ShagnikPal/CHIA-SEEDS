#!/usr/bin/env /usr/bin/python3.10
"""Iso-area reallocation: spend what the starve wave proved surrenderable.

The upward probe wave said which resources are live. The starve wave says
which are free to give up. The area model says what giving one up buys. This
script closes the loop: it holds total chip area at the stock A100's value and
asks what the best (n_sm, L2) pair is once the dead structures are deleted.

Everything here is priced by McPAT from the same knobs the simulator saw. Two
caveats travel with every number:

  * McPAT is configured at ``core_tech_node = 23`` nm against a real 7 nm
    A100, so absolute mm^2 are meaningless (the core array alone prices at
    ~4.6e3 mm^2 against an 826 mm^2 die). Only ratios are used, and even those
    are biased: at 23 nm McPAT puts SM array : L2 at roughly 19:1 where a 7 nm
    A100 is nearer 7:1, which tilts any area search towards selling SMs to buy
    cache. Results that survive despite that bias are the trustworthy ones.

  * The texture-cache credit is an upper bound. CACTI's area for that
    structure is dominated by its declared port and bank geometry rather than
    capacity, which is also why the capacity knob frees so little of it. So the
    credit is reported as a band -- full deletion, half, and the knob-only
    shrink -- and the iso-area conclusion is checked at each.
"""
import json
import subprocess
import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
from chia.simulators.accelsim_area import area_report, read_counters, removal_is_justified
from chia.simulators.accelsim import read_config_knobs
from chia.simulators.accelsim_power import render_power_xml

BASE_CFG = ("/raid/sp56429/chia_hackathon/accel-sim-framework/gpu-simulator/"
            "gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config")
SRC_XML = ("/raid/sp56429/chia_hackathon/accel-sim-framework/gpu-simulator/"
           "gpgpu-sim/configs/tested-cfgs/SM7_GV100/accelwattch_sass_sim.xml")
WORK = "/raid/sp56429/chia_hackathon/testkernel/dse/area_xml"
REPORT = ("/raid/sp56429/chia_hackathon/testkernel/dse/out4k/base/grid/"
          "sm108_clk1410_l2_base/accelwattch_power_report.log")

N_SM_STOCK, N_L2_STOCK = 108, 160


def _l2(sets: int, assoc: int, mshr: int) -> str:
    """Mirror of accelsim_dse._l2 so L2 levels are priced exactly as simulated."""
    return (f"S:{sets}:128:{assoc},L:B:m:L:P,A:{mshr}:{mshr//2},32:0,32")


L2_LEVELS = {
    "l2_half": _l2(64, 16, 192),
    "l2_base": None,
    "l2_2x": _l2(256, 16, 384),
    "l2_4x": _l2(512, 16, 768),
    "l2_8x": _l2(1024, 16, 1536),
}


def price(n_sm: int, l2_level: str) -> tuple[float, float, float, dict]:
    """(total, per_sm, per_l2_slice, sm_parts) for a design point."""
    knobs = dict(read_config_knobs(BASE_CFG))
    if L2_LEVELS[l2_level] is not None:
        knobs["-gpgpu_cache:dl2"] = L2_LEVELS[l2_level]
    xml = f"{WORK}/iso_{n_sm}_{l2_level}.xml"
    render_power_xml(SRC_XML, xml, knobs)
    r = area_report(xml, n_sm, N_L2_STOCK)
    if not r.ok:
        raise RuntimeError(f"{n_sm}/{l2_level}: {r.note}")
    return r.total_mm2, r.per_sm_mm2, r.per_l2_mm2, r.parts


if __name__ == "__main__":
    counters = read_counters(REPORT)
    knobs = dict(read_config_knobs(BASE_CFG))

    print("=== removal certification (from the stock run's own counters) ===")
    for part in ("Texture Cache", "Constant Cache", "Instruction Cache"):
        ok, why = removal_is_justified(part, counters, knobs)
        print(f"  {'CERTIFIED' if ok else 'refused  '}  {why}")

    print("\n=== pricing L2 levels at the stock SM count ===")
    priced = {}
    for lvl in L2_LEVELS:
        tot, per_sm, per_l2, parts = price(N_SM_STOCK, lvl)
        priced[lvl] = (tot, per_sm, per_l2, parts)
        print(f"  {lvl:<10} total={tot:9.1f}  perSM={per_sm:7.3f}  "
              f"perL2slice={per_l2:6.3f}  L2total={per_l2*N_L2_STOCK:8.1f}")

    base_total, base_sm, base_l2, base_parts = priced["l2_base"]
    tex = base_parts.get("Texture Cache", 0.0)
    fixed = base_total - N_SM_STOCK * base_sm - N_L2_STOCK * base_l2
    print(f"\nstock: {N_SM_STOCK} SM x {base_sm:.3f} + {N_L2_STOCK} L2 x "
          f"{base_l2:.3f} + fixed {fixed:.2f} = {base_total:.1f} mm^2")
    print(f"texture cache = {tex:.3f} mm^2/SM = {100*tex/base_sm:.1f}% of an SM"
          f"  -> {tex*N_SM_STOCK:.0f} mm^2 chip-wide if deleted")

    print("\n=== iso-area reallocation (budget = stock total) ===")
    print("credit    L2 level     max SMs   L2 mm^2   total     vs stock")
    print("-" * 64)
    plan = []
    for credit_name, credit in (("full", tex), ("half", tex / 2), ("knob", 1.62)):
        for lvl in L2_LEVELS:
            _, _, per_l2, _ = priced[lvl]
            sm_cost = base_sm - credit
            l2_cost = per_l2 * N_L2_STOCK
            budget = base_total - fixed - l2_cost
            n = int(budget // sm_cost)
            tot = fixed + n * sm_cost + l2_cost
            if n < 54 or n > 512:
                continue
            plan.append({"credit": credit_name, "credit_mm2": round(credit, 3),
                         "l2": lvl, "n_sm": n, "total_mm2": round(tot, 1)})
            print(f"{credit_name:<9} {lvl:<12} {n:6d}   {l2_cost:8.1f}  "
                  f"{tot:8.1f}  {100*(tot/base_total-1):+6.2f}%")
    json.dump({"stock_total": base_total, "per_sm": base_sm,
               "per_l2": base_l2, "fixed": fixed, "tex_mm2": tex,
               "plan": plan}, open("out4k/iso_plan.json", "w"), indent=2)
    print("\nwrote out4k/iso_plan.json")
