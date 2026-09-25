#!/usr/bin/env python3.10
"""Re-report the mqocc / mqinf waves from their JSON, with EDP.

sim() records power_w but not energy, so derive it: the SM clock is 1.41 GHz
(checked against the grid wave, where 108864 cyc @ 186.99 W printed 14.4369 mJ
-- 108864/1.41e9 * 186.99 * 1000 = 14.437).
"""
import json

CLK = 1.41e9


def load(name):
    return json.load(open(f"out4k/{name}.json"))["cells"]


def energy_mj(r):
    return r["cycles"] / CLK * r.get("power_w", 0.0) * 1000.0


def by_tile(cells):
    out = {}
    for r in cells:
        t, p = r["point"].split("/", 1)
        out.setdefault(t, {})[p] = r
    return out


print("=" * 74)
print("MISS-QUEUE DEPTH -> INFINITY   (stock L1=16 L2=32; both scaled together)")
print("=" * 74)
for tile, got in by_tile(load("mqinf")).items():
    ctrl = got["mq_1x"]["cycles"]
    print(f"\n[{tile}]  mq_1x = {ctrl:.0f} cycles")
    print(f"  {'depth':<8}{'L1':>6}{'L2':>7}{'cycles':>10}{'gain%':>8}"
          f"{'marginal%':>11}")
    prev = ctrl
    for m in (1, 2, 4, 8, 16, 64, 256):
        r = got.get(f"mq_{m}x")
        if not r or not r.get("ok"):
            continue
        c = r["cycles"]
        print(f"  {'mq_%dx' % m:<8}{16 * m:>6}{32 * m:>7}{c:>10.0f}"
              f"{100 * (ctrl - c) / ctrl:>7.2f}%{100 * (prev - c) / prev:>10.2f}%")
        prev = c

print("\n" + "=" * 74)
print("OCCUPANCY x MISS QUEUE  (EDP = cycles * energy, vs each tile's control)")
print("=" * 74)
ORDER = ["adaptive_off", "occ2", "mq16", "occ2_mq4", "occ2_mq16", "occ4_mq16"]
for tile, got in by_tile(load("mqocc")).items():
    ctrl = got["adaptive_off"]
    bedp = ctrl["cycles"] * energy_mj(ctrl)
    print(f"\n[{tile}]  control = {ctrl['cycles']:.0f} cycles, "
          f"{ctrl['power_w']:.2f} W, {energy_mj(ctrl):.4f} mJ")
    print(f"  {'point':<14}{'cycles':>9}{'gain%':>8}{'W':>8}{'mJ':>9}"
          f"{'EDP':>7}{'occ%':>7}{'CTA':>5}")
    for p in ORDER:
        r = got.get(p)
        if not r or not r.get("ok"):
            continue
        e = energy_mj(r)
        print(f"  {p:<14}{r['cycles']:>9.0f}"
              f"{100 * (ctrl['cycles'] - r['cycles']) / ctrl['cycles']:>7.2f}%"
              f"{r.get('power_w', 0):>8.2f}{e:>9.4f}"
              f"{r['cycles'] * e / bedp:>7.3f}"
              f"{r.get('occupancy', 0):>7.2f}{r.get('cta_per_core', 0):>5.0f}")


print("\n" + "=" * 74)
print("CYCLES BY ACHIEVED CONCURRENCY  (deep miss queues unless noted)")
print("=" * 74)
tiles = by_tile(load("mqocc"))
rows = []
for tile, got in tiles.items():
    for p, r in got.items():
        if not r.get("ok"):
            continue
        rows.append((r.get("cta_per_core", 0), tile, p, r["cycles"],
                     "deep" if "mq" in p else "stock"))
print(f"  {'CTA/SM':>7}  {'tile':<16}{'point':<14}{'queue':<7}{'cycles':>9}")
for cta, tile, p, c, q in sorted(rows):
    star = "  <-- best" if c == min(x[3] for x in rows) else ""
    print(f"  {cta:>7.0f}  {tile:<16}{p:<14}{q:<7}{c:>9.0f}{star}")
