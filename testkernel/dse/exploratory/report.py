#!/usr/bin/env /usr/bin/python3.10
"""Assemble every stage's JSON into the walk's final numbers."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")

from chia.simulators.accelsim_dse import (
    HW_DIMENSIONS,
    KernelRow,
    Tile,
    cells_from_records,
    format_grid,
    format_rows,
    licensed_dimensions,
    pareto_front,
    resident_ctas,
)
from chia.simulators.accelsim_probes import format_probe_report

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
TILES = {
    "base": Tile((128, 256, 32), (2, 4, 1), 3),
    "tb128x128x16_s3": Tile((128, 128, 16), (2, 2, 1), 3),
    "tb128x128x32_s3": Tile((128, 128, 32), (2, 2, 1), 3),
    "tb128x64x32_s3": Tile((128, 64, 32), (2, 2, 1), 3),
    "tb64x128x32_s3": Tile((64, 128, 32), (2, 2, 1), 3),
    "tb64x64x16_s10": Tile((64, 64, 16), (2, 2, 1), 10),
    "tb64x64x32_s5": Tile((64, 64, 32), (2, 2, 1), 5),
}


def load(path: str):
    p = f"{ROOT}/out/{path}"
    return json.load(open(p)) if os.path.exists(p) else None


def hdr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def power_breakdown(cell: dict, top: int = 10) -> str:
    comps = cell.get("components") or {}
    tot = cell.get("power_w") or 1.0
    live = sorted(((k, v) for k, v in comps.items() if v > 0), key=lambda x: -x[1])
    out = [f"  {k:<12}{v:>8.2f} W{100 * v / tot:>7.1f}%" for k, v in live[:top]]
    zero = [k for k, v in comps.items() if v == 0]
    out.append(f"  zero: {', '.join(zero)}")
    if cell.get("inert_power"):
        out.append(f"  INERT (zero but activity recorded): {cell['inert_power']}")
    return "\n".join(out)


def main() -> None:
    probe = load("baseline/probe.json")
    if probe:
        hdr("ROUND 0 - PROBE WAVE (CUTLASS tb128x256x32_s3, 1024^3)")
        print(format_probe_report(probe["report"]))
        dims = licensed_dimensions(probe["report"])
        print(f"\nlicensed hardware dimensions: {dims}")
        print("  (always-on: " + ", ".join(
            d for d, s in HW_DIMENSIONS.items() if s["always"]) + ")")
        base = next((c for c in probe["cells"] if c["point"] == "baseline"), None)
        if base:
            hdr("BASELINE POWER BREAKDOWN")
            print(f"  total {base['power_w']:.3f} W   "
                  f"cycles {base['cycles']:.0f}   "
                  f"consistent={base['power_consistent']}")
            print(power_breakdown(base))

    grid = load("baseline/grid.json")
    if grid:
        hdr("HARDWARE GRID - baseline kernel")
        print(format_grid(cells_from_records(grid["cells"])))

    screen = load("screen.json")
    if screen:
        hdr("SOFTWARE SCREEN - one sim per kernel at baseline hardware")
        cells = cells_from_records([r for r in screen if r["ok"]])
        print(format_grid(cells, baseline_point="base"))
        print()
        print(f"{'kernel':<20}{'blocks':>8}{'smem KB':>9}{'CTA/SM est':>11}")
        for r in sorted(screen, key=lambda r: r.get("edp") or 9e99):
            t = TILES.get(r["kernel"])
            if t:
                print(f"{r['kernel']:<20}{t.blocks(1024):>8}"
                      f"{t.smem_bytes / 1024:>9.0f}{resident_ctas(t):>11}")

    rows = load("rows.json")
    if rows:
        hdr("FULL ROWS - survivors x hardware grid")
        krows = []
        for n in {c["kernel"] for c in rows["cells"]}:
            cs = cells_from_records([c for c in rows["cells"] if c["kernel"] == n])
            krows.append(KernelRow(kernel=n, tile=TILES.get(n), cells=cs))
        print(format_rows(krows, baseline_kernel="base"))
        hdr("PARETO FRONT - (cycles, energy) over every cell")
        front = pareto_front(cells_from_records(rows["cells"]))
        print(f"{'kernel':<20}{'point':<22}{'cycles':>10}{'mJ':>9}{'EDP':>11}")
        for c in front:
            print(f"{c.kernel:<20}{c.point:<22}{c.cycles:>10.0f}"
                  f"{c.joules * 1e3:>9.4f}{c.edp:>11.3e}")


if __name__ == "__main__":
    main()
