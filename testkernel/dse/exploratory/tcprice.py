#!/usr/bin/env /usr/bin/python3.10
"""What would the tensor cores cost, given that McPAT prices them at zero?

``tcunits_axis.py`` measures what more tensor throughput buys.  This asks the
other half of the question, and the honest answer up front is: **the project's
area model cannot price it.**  Grepping the AccelWattch McPAT build for
"tensor" finds only power activity counters (`gpgpu_sim_wrapper.cc:494`,
derived from `sfu_coeff`) -- there is no `FunctionalUnit` of tensor type, no
`TENSOR_per_core` XML field, and nothing contributing to core *area*.  Tensor
cores are modelled as zero silicon.  Verified on the artifacts: the 4x-tensor
point `out4k/probe/many_tensor_units/accelwattch_sass_sim.xml` carries
byte-identical ALU/MUL/FPU counts to baseline, so it prices at exactly the
stock 42.7287 mm^2 per SM.

So a tensor result reported against this model is a free lunch, which is the
one thing the brief forbids.  This builds an off-model bound instead, the same
move `mqprice.py` makes for miss queues.

The proxy: a tensor core is a dense FMA array, and McPAT *does* price dense FP
datapath via `FPU_per_core` (`core.cc:6713` -> `logic.cc:686-690`, area
\\propto FPU_per_core/2).  `accelsim_power.py:226` already uses that field as
the destination for `-gpgpu_num_sp_units` at 8 lanes per pipeline.  So charge
"2x the HMMA issue rate" as "2x the FP datapath" and report the delta as an
upper-ish bound, stated as a proxy and never as a measurement.

`alu64` is a control, not a proposal: McPAT's ALU path is **quadratic** in unit
count (`logic.cc:747` sets area_t \\propto num_fu, then `:820` multiplies by
num_fu again).  If alu64 comes back at ~4x the Integer-ALU area while fpu64
comes back at ~2x the FP area, the two scaling laws are confirmed and the FPU
proxy is safe to quote linearly.  If FPU is *not* linear, the proxy is void.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

from chia.simulators.accelsim_area import area_report

SRC = ("/raid/sp56429/chia_hackathon/accel-sim-framework/gpu-simulator/"
       "gpgpu-sim/configs/tested-cfgs/SM7_GV100/accelwattch_sass_sim.xml")
N_SM, N_L2 = 108, 160
WORK = "/raid/sp56429/chia_hackathon/testkernel/dse/tc_price_xml"


POINTS = [
    ("stock", {}, "reference: ALU 32 / FPU 32 / MUL 4"),
    ("fpu16", {"FPU_per_core": 16}, "half the FP datapath -- linearity check down"),
    ("fpu64", {"FPU_per_core": 64}, "PROXY for 2x HMMA issue rate"),
    ("fpu128", {"FPU_per_core": 128}, "PROXY for 4x HMMA issue rate"),
    ("alu64", {"ALU_per_core": 64}, "CONTROL: is the ALU path really quadratic?"),
]


def patch(src: str, dst: str, fields: dict) -> None:
    s = open(src).read()
    for k, v in fields.items():
        s2 = re.sub(rf'(<param name="{k}" value=")[^"]*(")', rf'\g<1>{v}\g<2>', s)
        if s2 == s:
            raise SystemExit(f"field {k} not found in {src}")
        s = s2
    open(dst, "w").write(s)


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for label, fields, why in POINTS:
        dst = os.path.join(WORK, f"{label}.xml")
        patch(SRC, dst, fields)
        r = area_report(dst, N_SM, N_L2)
        if not r.ok:
            print(f"{label:<10} FAILED  {r.note}")
            continue
        rows.append((label, r, why))

    if not rows:
        return
    base = rows[0][1]
    keys = ["Integer ALUs", "Floating Point Units", "Complex ALUs"]
    print(f"\n{'point':<10}{'per-SM mm2':>12}{'vs stock':>10}"
          + "".join(f"{k[:14]:>16}" for k in keys))
    for label, r, why in rows:
        d = (r.per_sm_mm2 / base.per_sm_mm2 - 1) * 100 if base.per_sm_mm2 else 0
        parts = "".join(f"{(r.parts or {}).get(k, 0.0):>16.4f}" for k in keys)
        print(f"{label:<10}{r.per_sm_mm2:>12.4f}{d:>9.2f}%{parts}   {why}")

    print(f"\nwhole-die at {N_SM} SM:")
    for label, r, _ in rows:
        print(f"  {label:<10}{r.total_mm2:>12.2f} mm2   "
              f"delta {r.total_mm2 - base.total_mm2:>+10.2f}")


if __name__ == "__main__":
    main()
