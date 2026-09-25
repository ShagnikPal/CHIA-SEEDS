#!/usr/bin/env python3.10
"""Consolidate every wave into one results view.

Cycles only where a wave ran without power; EDP where it ran with.  Nothing
here re-simulates -- it reads the JSON each wave already wrote.
"""
import json
import os

CLK = 1.41e9
OUT = "out4k"


def cells(name):
    p = f"{OUT}/{name}.json"
    if not os.path.exists(p):
        return {}
    out = {}
    for r in json.load(open(p))["cells"]:
        t, pt = r["point"].split("/", 1)
        out.setdefault(t, {})[pt] = r
    return out


def mj(r):
    return r["cycles"] / CLK * r.get("power_w", 0.0) * 1000.0


def line(ch="-", n=78):
    print(ch * n)


TILE = "t128x128x16_s4"
line("=")
print("CUTLASS TF32 GEMM 4096x4096x64 on SM80_A100 -- consolidated")
line("=")


iso = cells("isomq")
stock = iso.get(TILE, {}).get("stock")
bstock = iso.get("base", {}).get("stock")
if stock and bstock:
    print(f"\nstock chip 4866.95 mm^2 = 108 SM x 42.729 + 160 L2 x 1.478 "
          f"+ 15.731 fixed")
    print(f"  base tile   {bstock['cycles']:>8.0f} cyc  "
          f"{bstock['power_w']:>6.1f} W  {mj(bstock):>7.3f} mJ   1 CTA/SM")
    print(f"  tuned tile  {stock['cycles']:>8.0f} cyc  "
          f"{stock['power_w']:>6.1f} W  {mj(stock):>7.3f} mJ   2 CTA/SM")


print("\n" + "CONCURRENCY IS THE AXIS (cycles, deep queue unless noted)")
line()
mo = cells("mqocc")
rows = []
for t, got in mo.items():
    for p, r in got.items():
        if r.get("ok"):
            rows.append((r.get("cta_per_core", 0), t, p,
                         "deep" if "mq" in p else "stock", r["cycles"]))
best = min(r[4] for r in rows) if rows else 0
print(f"  {'CTA/SM':>6}  {'tile':<16}{'point':<14}{'queue':<7}{'cycles':>9}")
for cta, t, p, q, c in sorted(rows):
    print(f"  {cta:>6.0f}  {t:<16}{p:<14}{q:<7}{c:>9.0f}"
          f"{'  <-- best' if c == best else ''}")


print("\n" + "TWO LOSSES THAT COMPOSE INTO A WIN (base tile, 108 SM)")
line()
g = mo.get("base", {})
if g.get("adaptive_off"):
    ref = g["adaptive_off"]["cycles"]
    for p in ("occ2", "mq16", "occ2_mq16"):
        if g.get(p):
            c = g[p]["cycles"]
            print(f"  {p:<14}{c:>9.0f}{100 * (ref - c) / ref:>+8.2f}%")


mi = cells("mqinf").get(TILE, {})
if mi.get("mq_1x"):
    print("\n" + "DEPTH CURVE, TUNED TILE (knee, then decline)")
    line()
    ref = mi["mq_1x"]["cycles"]
    for m in (1, 2, 4, 8, 16, 64, 256):
        r = mi.get(f"mq_{m}x")
        if r and r.get("ok"):
            print(f"  L1={16 * m:<5} L2={32 * m:<5}{r['cycles']:>9.0f}"
                  f"{100 * (ref - r['cycles']) / ref:>+8.2f}%")


print("\n" + "ISO-AREA, TUNED TILE (budget 4866.95 mm^2)")
line()
AREA = {"stock": 4867, "isoA_detex_sm160": 4859,
        "isoA_mq16_sm158": 4861, "isoA_mq16_sm107": 4855}

NSM = {"stock": 108, "isoA_detex_sm160": 160,
       "isoA_mq16_sm158": 158, "isoA_mq16_sm107": 107}
got = iso.get(TILE, {})
if got.get("stock"):
    rc, re_ = got["stock"]["cycles"], mj(got["stock"])
    print(f"  {'design':<20}{'SM':>4}{'cycles':>9}{'speedup':>9}"
          f"{'mJ':>8}{'EDP':>7}{'mm^2':>7}")
    for p in ("stock", "isoA_mq16_sm107", "isoA_detex_sm160",
              "isoA_mq16_sm158"):
        r = got.get(p)
        if not r or not r.get("ok"):
            continue
        e = mj(r)
        print(f"  {p:<20}{NSM.get(p, 108):>4}"
              f"{r['cycles']:>9.0f}{rc / r['cycles']:>8.3f}x"
              f"{e:>8.3f}{r['cycles'] * e / (rc * re_):>7.3f}"
              f"{AREA.get(p, 0):>7}")


sq = cells("smq").get(TILE, {})
if sq:
    print("\n" + "SM COUNT x QUEUE, SM-MATCHED (de-confounds the above)")
    line()
    print(f"  {'SMs':>5}{'stock q':>10}{'mq16':>10}{'queue':>10}")
    for n in (156, 158, 160):
        a = sq.get(f"sm{n}_stockq")
        b = sq.get(f"sm{n}_mq16")
        if a and b and a.get("ok") and b.get("ok"):
            ac, bc = a["cycles"], b["cycles"]
            print(f"  {n:>5}{ac:>10.0f}{bc:>10.0f}"
                  f"{100 * (ac - bc) / ac:>9.2f}%")
    for q in ("stockq", "mq16"):
        v = [sq[f"sm{n}_{q}"]["cycles"] for n in (156, 158, 160)
             if sq.get(f"sm{n}_{q}", {}).get("ok")]
        if len(v) == 3:
            print(f"  spread across 156-160 SMs [{q}]: "
                  f"{100 * (max(v) - min(v)) / min(v):.2f}%")
else:
    print("\n  [smq de-confounding grid still running]")
