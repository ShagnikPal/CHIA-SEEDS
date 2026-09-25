#!/usr/bin/env /usr/bin/python3.10
"""Co-design coupling table at 4096x4096x64.

Answers one question: does the best hardware point depend on which kernel you
run?  If it does, optimizing the chip for the stock kernel and *then* tuning
the kernel (hardware-first alternation) lands somewhere the joint search would
have rejected.  Compare that policy against marginalizing the hardware axis.
"""
from __future__ import annotations
import json, sys, itertools

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse/out4k"
STOCK = "sm108/clk1410/l2_base"


def load(name):
    d = json.load(open(f"{ROOT}/{name}_grid.json"))
    out = {}
    for c in d["cells"]:
        if not c.get("ok") or not c.get("edp"):
            continue
        out[c["point"]] = c
    return out


def main():
    names = sys.argv[1:] or ["base", "t128x128x16_s4"]
    grids = {n: load(n) for n in names}
    for n, g in grids.items():
        print(f"{n}: {len(g)} valid cells")


    ref = grids[names[0]].get(STOCK)
    if not ref:
        print(f"!! {STOCK} missing from {names[0]}"); return
    E0 = ref["edp"]

    points = sorted(set(itertools.chain.from_iterable(g for g in grids.values())))
    w = max(len(p) for p in points) + 2
    print(f"\nEDP normalized to {names[0]} @ {STOCK}  (lower = better)\n")
    print(f"{'hw point':<{w}}" + "".join(f"{n:>22}" for n in names))
    for p in points:
        row = f"{p:<{w}}"
        for n in names:
            c = grids[n].get(p)
            row += f"{c['edp']/E0:>22.4f}" if c else f"{'-':>22}"
        print(row)


    print("\n" + "=" * 70)
    best_hw_for_stock = min(grids[names[0]], key=lambda p: grids[names[0]][p]["edp"])
    print(f"1. hardware-first: best chip for stock kernel = {best_hw_for_stock}")
    alt = {}
    for n in names:
        c = grids[n].get(best_hw_for_stock)
        if c:
            alt[n] = c["edp"] / E0
            print(f"     then kernel {n:<20} -> {alt[n]:.4f}")
    a_best = min(alt.values())

    print(f"\n2. marginalize: score(kernel) = min over the whole grid")
    marg = {}
    for n in names:
        p = min(grids[n], key=lambda p: grids[n][p]["edp"])
        marg[n] = grids[n][p]["edp"] / E0
        print(f"     {n:<20} best at {p:<24} -> {marg[n]:.4f}")
    m_best = min(marg.values())

    print(f"\nhardware-first best : {a_best:.4f}  ({(a_best-1)*100:+.1f}% vs stock)")
    print(f"marginalized  best : {m_best:.4f}  ({(m_best-1)*100:+.1f}% vs stock)")
    if a_best > 1 and m_best < 1:
        print("hardware-first found NO improvement; marginalizing did.")
    elif a_best != 1:
        print(f"ratio of gains     : {(1-m_best)/(1-a_best):.1f}x")


    print("\n" + "=" * 70)
    print("fixed-clock slice @ 1410 MHz (clock axis is degenerate -- see caveats)\n")
    fixed = [p for p in points if "clk1410" in p]
    print(f"{'hw point':<{w}}" + "".join(f"{n:>22}" for n in names))
    for p in fixed:
        row = f"{p:<{w}}"
        for n in names:
            c = grids[n].get(p)
            row += f"{c['edp']/E0:>22.4f}" if c else f"{'-':>22}"
        print(row)
    fam = {}
    for n in names:
        cand = {p: grids[n][p] for p in fixed if p in grids[n]}
        if cand:
            b = min(cand, key=lambda p: cand[p]["edp"])
            fam[n] = b
            print(f"  argmin @1410  {n:<22} -> {b}  ({cand[b]['edp']/E0:.4f})")
    print("  COUPLED at fixed clock" if len(set(fam.values())) > 1
          else "  same chip wins for both at fixed clock")


    print("\ncoupling check -- argmin hardware per kernel:")
    argmins = {}
    for n in names:
        p = min(grids[n], key=lambda p: grids[n][p]["edp"])
        argmins[n] = p
        print(f"  {n:<22} -> {p}")
    print("  COUPLED (optimal chip depends on kernel)" if len(set(argmins.values())) > 1
          else "  not coupled at this granularity -- same chip wins for both")

    clock_sensitivity(grids, names)


def clock_sensitivity(grids, names):
    """How steeply would power have to grow with f for the clock axis to bite?

    EDP = P(f) * (C(f)/f)^2.  Fit P ~ f^alpha and C ~ f^beta from the grid; the
    axis is degenerate whenever alpha + 2*beta < 2.  This is assumption-free --
    no V/f curve is invented, we just report the measured exponent and the
    threshold.  (-dvfs_enabled cannot fix this: power_interface.cc:64 calls
    set_model_voltage(1) with a literal, so voltage does not track frequency.)
    """
    import math
    print("\n" + "=" * 70)
    print("clock-axis sensitivity\n")
    for n in names:
        g = grids[n]
        pts = sorted((int(p.split("clk")[1].split("/")[0]), g[p])
                     for p in g if "sm108" in p and "l2_base" in p)
        if len(pts) < 2:
            continue
        (f0, c0), (f1, c1) = pts[0], pts[-1]
        alpha = math.log(c1["power_w"] / c0["power_w"]) / math.log(f1 / f0)
        beta = math.log(c1["cycles"] / c0["cycles"]) / math.log(f1 / f0)
        print(f"  {n:<22} P ~ f^{alpha:.2f}   cycles ~ f^{beta:.2f}   "
              f"EDP ~ f^{alpha + 2*beta - 2:+.2f}")
    print("\n  EDP exponent < 0 => the top clock bin always wins => axis is")
    print("  degenerate and contributes no design information. Real DVFS gives")
    print("  V ~ f so dynamic P ~ f^3, which would flip the preference. Quote")
    print("  co-design results at fixed clock until the model supports V(f).")


if __name__ == "__main__":
    main()
