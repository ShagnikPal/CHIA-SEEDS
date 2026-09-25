#!/usr/bin/env python3.10
"""Price the miss-queue design points at iso-area and solve for SM count.

The budget is the stock chip: 4866.95 mm^2 at 108 SM + 160 L2 slices.  A
design point may spend the certified texture credit (TEX_ACC/TC_H/TC_M zero in
all 161 power reports, no suppressing flag) and must pay for whatever it buys.
Whatever is left over buys SMs.

Two things are priced by running McPAT on the design point's OWN XML rather
than scaling the stock number:

  * occ2/occ4 enlarge shared memory and the register file, so their per-SM
    area is genuinely larger and the XML says by how much.
  * adaptive_cache_config 0 changes the modelled L1/shmem split, so an
    AOFF point is not comparable to the adaptive-on stock XML part-by-part.
    Every point here is AOFF, and each is priced against its own XML.

The miss queue is the one thing McPAT cannot see: GPGPU-Sim's miss_queue_size
is not an XML parameter.  It is priced with McPAT's own cache-controller miss
buffer, which sits at 16 entries for the dcache -- exactly the stock L1
miss_queue_size -- by bumping that field and re-running.  Only the first field
(miss buffer) moves; fill/prefetch/writeback are left alone.
"""
import json
import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
from chia.simulators.accelsim_area import area_report

OUT = "out4k"
STOCK_TOTAL = 4866.95
N_L2 = 160
FIXED = 15.731


def bump_miss_buffer(xml: str, anchor: str, new: str) -> str:
    i = xml.index(anchor)
    j = xml.index('name="buffer_sizes"', i)
    k = xml.index('value="', j) + len('value="')
    e = xml.index('"', k)
    f = [x.strip() for x in xml[k:e].split(",")]
    f[0] = new
    return xml[:k] + ", ".join(f) + xml[e:]


def price(point_dir: str, l1_mq: int | None, l2_mq: int | None,
          tag: str) -> tuple[float, float]:
    """Return (per-SM mm^2, per-L2-slice mm^2) for one design point."""
    src = f"{point_dir}/accelwattch_sass_sim.xml"
    xml = open(src).read()
    if l1_mq:
        xml = bump_miss_buffer(xml, '<component id="system.core0.dcache"',
                               str(l1_mq))
    if l2_mq:
        xml = bump_miss_buffer(xml, '<component id="system.L20"', str(l2_mq))
    dst = f"/tmp/isomq_{tag}.xml"
    open(dst, "w").write(xml)
    r = area_report(dst, n_sm=108, n_l2=N_L2)
    if not r.ok:
        raise SystemExit(f"area_report failed for {tag}: {r.note}")
    return r.per_sm_mm2, r.per_l2_mm2, r.parts.get("Texture Cache", 0.0)


def solve_nsm(per_sm: float, per_l2: float) -> int:
    """Largest SM count that still fits inside the stock chip."""
    room = STOCK_TOTAL - FIXED - N_L2 * per_l2
    return int(room // per_sm)


def main() -> None:

    CANDIDATES = [
        ("adaptive_off", None, None),
        ("occ2", None, None),
        ("mq16", 256, 512),
        ("occ2_mq4", 64, 128),
        ("occ2_mq16", 256, 512),
        ("occ4_mq16", 256, 512),
    ]
    tile = sys.argv[1] if len(sys.argv) > 1 else "base"
    rows = []
    for name, l1, l2 in CANDIDATES:
        d = f"{OUT}/mqocc/{tile}/{name}"
        try:
            per_sm, per_l2, tex = price(d, l1, l2, f"{tile}_{name}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  {name:<14} skipped ({type(e).__name__})")
            continue


        net_sm = per_sm - tex
        rows.append({
            "point": name, "per_sm_raw": per_sm, "tex_credit": tex,
            "per_sm_net": net_sm, "per_l2": per_l2,
            "n_sm_isoarea": solve_nsm(net_sm, per_l2),
            "n_sm_no_credit": solve_nsm(per_sm, per_l2),
        })

    print(f"\niso-area budget = {STOCK_TOTAL:.2f} mm^2 "
          f"(stock: 108 SM, {N_L2} L2 slices)   tile={tile}\n")
    print(f"  {'point':<14}{'raw/SM':>9}{'tex cr':>9}{'net/SM':>9}"
          f"{'/L2':>7}{'SMs':>6}{'SMs*':>6}")
    for r in rows:
        print(f"  {r['point']:<14}{r['per_sm_raw']:>9.3f}"
              f"{-r['tex_credit']:>9.3f}{r['per_sm_net']:>9.3f}"
              f"{r['per_l2']:>7.3f}{r['n_sm_isoarea']:>6}"
              f"{r['n_sm_no_credit']:>6}")
    print("\n  SMs  = iso-area SM count spending the texture credit")
    print("  SMs* = iso-area SM count WITHOUT the credit (headline must not")
    print("         depend on the de-texture assumption to be a win)")
    json.dump(rows, open(f"{OUT}/isomq_plan_{tile}.json", "w"), indent=1)
    print(f"\nwrote {OUT}/isomq_plan_{tile}.json")


if __name__ == "__main__":
    main()
