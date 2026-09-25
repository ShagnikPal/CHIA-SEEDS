#!/usr/bin/env /usr/bin/python3.10
"""Price every starve probe in silicon, so "at the cost of what?" has a number.

The upward probe wave says which resources are live. The starve wave says
which are free to surrender. Neither says how much surrendering one *buys* --
that needs an area model, and McPAT has one: the same CACTI instances that
produce AccelWattch's watts produce an area for every array and functional
unit. ``accelsim_area`` runs McPAT standalone against an XML rendered from the
same knobs the simulator saw, so the cycles cost and the area credit for a
probe come from one consistent description of the machine.

Output is an exchange rate: mm^2 of SM freed per probe, and what that buys
when converted into SMs or L2 slices at the A100 baseline.

    area4k.py            exchange rates for every starve probe
    area4k.py plan       plus the iso-area designs the rates imply
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

from chia.simulators.accelsim import read_config_knobs
from chia.simulators.accelsim_area import area_report, format_area
from chia.simulators.accelsim_power import render_power_xml
from chia.simulators.accelsim_probes import STARVE_PROBES

ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CFG = f"{ACCELSIM}/gpu-simulator/gpgpu-sim/configs/tested-cfgs"
BASE_CFG = f"{CFG}/SM80_A100/gpgpusim.config"
SRC_XML = f"{CFG}/SM7_GV100/accelwattch_sass_sim.xml"
WORK = "/raid/sp56429/chia_hackathon/testkernel/dse/area_xml"

N_SM = 108
N_L2 = 160


def price(name: str, knobs: dict, n_sm: int = N_SM, n_l2: int = N_L2):
    import os
    os.makedirs(WORK, exist_ok=True)
    eff = read_config_knobs(BASE_CFG)
    eff.update(knobs)
    xml = f"{WORK}/{name}.xml"
    render_power_xml(SRC_XML, xml, eff)
    return area_report(xml, n_sm, n_l2)


def main() -> None:
    base = price("baseline", {})
    if not base.ok:
        print("baseline area failed:", base.note)
        return
    print("=== A100 baseline, priced by McPAT (23 nm; ratios only) ===")
    print(format_area(base))

    print("\n=== what each starve probe frees ===")
    print(f"  {'probe':<20}{'per-SM':>9}{'freed':>9}{'% SM':>7}"
          f"{'chip':>10}{'buys SMs':>10}{'buys L2 MB':>12}")
    rows = []
    for p in STARVE_PROBES:
        r = price(p.name, p.knobs)
        if not r.ok:
            print(f"  {p.name:<20}{'FAILED':>9}  {r.note[:50]}")
            continue
        freed_sm = base.per_sm_mm2 - r.per_sm_mm2
        freed_chip = freed_sm * N_SM


        buys_sm = freed_chip / r.per_sm_mm2 if r.per_sm_mm2 else 0.0
        buys_l2 = (freed_chip / base.per_l2_mm2) * 0.25 if base.per_l2_mm2 else 0.0
        rows.append((p.name, freed_sm, freed_chip, buys_sm, buys_l2))
        print(f"  {p.name:<20}{r.per_sm_mm2:>9.3f}{freed_sm:>9.3f}"
              f"{100*freed_sm/base.per_sm_mm2:>6.1f}%{freed_chip:>10.1f}"
              f"{buys_sm:>10.1f}{buys_l2:>12.1f}")
    print("\n  'buys L2 MB' converts the freed area into extra 256 KB L2 slices"
          "\n  at the baseline slice price; 'buys SMs' builds the extra SMs"
          "\n  without the structure that was removed.")
    print("\n  Probes absent from this table move no structure McPAT prices"
          "\n  (register banks, tensor/LSU issue width, NoC flit size, FP64"
          "\n  pipelines). Their cycle cost is still a real result; their area"
          "\n  credit is simply outside this model and is reported as 0.000.")
    return rows


if __name__ == "__main__":
    main()
