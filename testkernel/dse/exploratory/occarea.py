#!/usr/bin/env /usr/bin/python3.10
"""Price the occupancy design point and test it against the iso-area budget.

The starve wave certified the texture cache removable (TEX_ACC = TC_H = TC_M
= 0, no perfecting flag). The question this answers is whether that credit
covers the cost of the only thing the workload actually wants: enough shared
memory and registers to fit a second CTA per SM.

Accounting convention, and the one place it is loose. McPAT prices the L1 data
cache and the shared memory as two structures, but Ampere builds them as one
unified SRAM array sized by ``-gpgpu_unified_l1d_size``. At stock the two add
to 32 + 164 = 196 KB against a true unified 192 KB, so the existing convention
already tracks the array to within 2%. Carrying it forward to the doubled
point gives 32 + 384 = 416 KB against a true 384 KB -- an 8% *overcharge*,
which biases against the design being argued for. Left deliberately
conservative rather than tuned.
"""
import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
from chia.simulators.accelsim import read_config_knobs
from chia.simulators.accelsim_area import area_report, removal_credit
from chia.simulators.accelsim_power import render_power_xml
from iso4k import BASE_CFG, SRC_XML, WORK, N_SM_STOCK, N_L2_STOCK

POINTS = {
    "stock":          {},
    "occ2":           {"-gpgpu_shmem_size": str(384 * 1024),
                       "-gpgpu_shader_registers": "131072"},
    "occ4":           {"-gpgpu_shmem_size": str(640 * 1024),
                       "-gpgpu_shader_registers": "262144"},
}


def price(name, over):
    knobs = dict(read_config_knobs(BASE_CFG))
    knobs.update(over)
    xml = f"{WORK}/occ_{name}.xml"
    render_power_xml(SRC_XML, xml, knobs)
    r = area_report(xml, N_SM_STOCK, N_L2_STOCK)
    if not r.ok:
        raise RuntimeError(f"{name}: {r.note}")
    return r


if __name__ == "__main__":
    rs = {n: price(n, o) for n, o in POINTS.items()}
    base = rs["stock"]
    fixed = base.total_mm2 - N_SM_STOCK * base.per_sm_mm2 - N_L2_STOCK * base.per_l2_mm2

    print(f"{'point':<8}{'per-SM':>9}{'dcache':>9}{'shmem':>9}{'regfile':>9}"
          f"{'tex':>9}{'d/SM':>9}")
    for n, r in rs.items():
        p = r.parts
        d = r.per_sm_mm2 - base.per_sm_mm2
        print(f"{n:<8}{r.per_sm_mm2:>9.3f}{p.get('Data Cache',0):>9.3f}"
              f"{p.get('Shared Memory',0):>9.3f}{p.get('Register Files',0):>9.3f}"
              f"{p.get('Texture Cache',0):>9.3f}{d:>+9.3f}")

    tex_sm, tex_chip = removal_credit(base, ["Texture Cache"])
    print(f"\ntexture credit: {tex_sm:.3f} mm^2/SM  ({tex_chip:.0f} chip-wide)")

    print(f"\n{'design':<26}{'per-SM':>9}{'n_sm':>7}{'total':>10}{'vs stock':>10}")
    print("-" * 62)
    budget = base.total_mm2
    rows = []
    for n in ("occ2", "occ4"):
        for detex in (False, True):
            per_sm = rs[n].per_sm_mm2 - (tex_sm if detex else 0.0)

            n_sm = int((budget - fixed - N_L2_STOCK * base.per_l2_mm2) // per_sm)
            tot = fixed + n_sm * per_sm + N_L2_STOCK * base.per_l2_mm2
            label = f"{n}{' + de-texture' if detex else ''}"
            rows.append((label, per_sm, n_sm, tot))
            print(f"{label:<26}{per_sm:>9.3f}{n_sm:>7}{tot:>10.1f}"
                  f"{100*(tot/budget-1):>+9.2f}%")
    print(f"\n{'stock reference':<26}{base.per_sm_mm2:>9.3f}{N_SM_STOCK:>7}"
          f"{budget:>10.1f}{0.0:>+9.2f}%")
