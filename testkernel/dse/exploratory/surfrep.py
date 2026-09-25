#!/usr/bin/env /usr/bin/python3.10
"""Read the n_sm x l2 surface and answer the co-design question at n=7.

Three things, in order of how much they matter:

1. Does the optimal chip depend on the kernel?  (the coupling)
2. What does that cost a hardware-first search, versus marginalizing?
3. Does CTA wave quantization *predict* where the coupling appears?

(3) is stated as a HYPOTHESIS UNDER TEST, not a mechanism being illustrated.
Checked against the n=2 grid it already has counterexamples: the rule predicts
sm108 for `base` where sm216 measured best.  An earlier version of this claim
also computed waves as ctas/n_sm, silently assuming one resident CTA per SM --
with cta_per_core=2 the tuned tile has the SAME 1.19 waves at sm432 as the
stock tile, not double.  Capacity is n_sm * cta_per_core.  The table below
reports the rule's hit rate honestly; a low hit rate is a real result (EDP
trades cycles against the power cost of more SMs, so the argmin need not sit
at a wave boundary at all).
"""
from __future__ import annotations

import json
import math
import sys

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse/out4k/surface"
SM = ["sm54", "sm108", "sm216", "sm432"]
L2 = ["l2_half", "l2_base", "l2_2x", "l2_4x"]
STOCK = "sm108/clk1410/l2_base"


def load(path=None):
    d = json.load(open(path or f"{ROOT}/surface.json"))
    by = {}
    bad = []
    for c in d["cells"]:
        if not c.get("ok") or not c.get("edp"):
            bad.append((c.get("kernel"), c.get("point"), c.get("note", "")[:50]))
            continue
        by.setdefault(c["kernel"], {})[c["point"]] = c
    return by, bad


def waves(ctas: int, nsm: int, cta_per_core: int = 1) -> tuple[int, float]:
    """(rounds, utilisation) against the *resident* capacity.

    Capacity is n_sm * cta_per_core, not n_sm. Using n_sm alone understates a
    high-occupancy tile's capacity and invents a wave cliff that is not there.
    """
    cap = max(1, nsm * max(1, int(cta_per_core or 1)))
    rounds = math.ceil(ctas / cap)
    return rounds, ctas / (rounds * cap)


def main() -> None:
    by, bad = load(sys.argv[1] if len(sys.argv) > 1 else None)
    if bad:
        print(f"!! {len(bad)} unusable cells (excluded, NOT treated as data):")
        for k, p, n in bad[:12]:
            print(f"   {k:<18}{p:<24}{n}")
        print()

    kernels = sorted(by, key=lambda k: by[k].get(STOCK, {}).get("edp") or 9e18)
    ref = None
    for k in kernels:
        if STOCK in by[k] and k == "base":
            ref = by[k][STOCK]["edp"]
    if ref is None:
        ref = by[kernels[0]][STOCK]["edp"]
        print(f"note: 'base' missing stock cell; normalizing to {kernels[0]}\n")


    print("=" * 78)
    print(f"EDP surface @ {STOCK.split('/')[1]}, normalized to base @ {STOCK}")
    print("=" * 78)
    for k in kernels:
        g = by[k]
        n = len(g)
        ctas = next((c.get("issued_ctas") for c in g.values() if c.get("issued_ctas")), 0)
        print(f"\n{k}   ({int(ctas or 0)} CTAs, {n}/16 cells)")
        print(f"  {'':<10}" + "".join(f"{s:>11}" for s in SM))
        for l2 in L2:
            row = f"  {l2:<10}"
            for sm in SM:
                c = g.get(f"{sm}/clk1410/{l2}")
                row += f"{c['edp']/ref:>11.4f}" if c else f"{'-':>11}"
            print(row)
        best = min(g, key=lambda p: g[p]["edp"])
        print(f"  argmin -> {best}  ({g[best]['edp']/ref:.4f})")


    print("\n" + "=" * 78)
    print("COUPLING: optimal hardware per kernel")
    print("=" * 78)
    argmin = {k: min(by[k], key=lambda p: by[k][p]["edp"]) for k in kernels}
    for k in kernels:
        g = by[k]
        print(f"  {k:<20}{argmin[k]:<26}{g[argmin[k]]['edp']/ref:>8.4f}")
    distinct = set(argmin.values())
    print(f"\n  {len(distinct)} distinct optimal chips across {len(kernels)} kernels")
    print("  => COUPLED" if len(distinct) > 1 else "  => NOT coupled")


    print("\n" + "=" * 78)
    print("SEARCH POLICY")
    print("=" * 78)
    if "base" in by:
        hw = argmin["base"]
        cand = {k: by[k][hw]["edp"] / ref for k in kernels if hw in by[k]}
        hw_first = min(cand.values())
        print(f"  hardware-first: lock chip to base's optimum ({hw}),")
        print(f"                  then pick the best kernel on it -> {hw_first:.4f}")
        for k in sorted(cand, key=cand.get):
            print(f"      {k:<20}{cand[k]:>8.4f}")
    else:
        hw_first = None

    marg = {k: by[k][argmin[k]]["edp"] / ref for k in kernels}
    m_best = min(marg.values())
    m_k = min(marg, key=marg.get)
    print(f"\n  marginalize:    each kernel at its own best chip")
    print(f"                  best -> {m_best:.4f}  ({m_k} @ {argmin[m_k]})")
    if hw_first:
        print(f"\n  hardware-first {hw_first:.4f}  ({(hw_first-1)*100:+.1f}% vs stock)")
        print(f"  marginalized   {m_best:.4f}  ({(m_best-1)*100:+.1f}% vs stock)")
        if hw_first > 1.0 and m_best < 1.0:
            print("  hardware-first found no improvement at all; marginalizing did")
        elif abs(hw_first - 1.0) > 1e-9:
            print(f"  ratio of gains {(1-m_best)/(1-hw_first):.2f}x")


    print("\n" + "=" * 78)
    print("MECHANISM: does CTA wave quantization predict the optimal SM count?")
    print("=" * 78)
    print("  rule under test: argmin SM = largest chip with >=2 rounds at >=80%")
    print("  capacity = n_sm * cta_per_core  (NOT n_sm -- that invents cliffs)\n")
    print(f"  {'kernel':<20}{'CTAs':>6}{'c/SM':>5}   " + "".join(f"{s:>15}" for s in SM))
    hits = 0
    for k in kernels:
        g = by[k]
        ctas = int(next((c.get("issued_ctas") for c in g.values()
                         if c.get("issued_ctas")), 0) or 0)
        cpc = int(next((c.get("cta_per_core") for c in g.values()
                        if c.get("cta_per_core")), 1) or 1)
        row = f"  {k:<20}{ctas:>6}{cpc:>5}   "
        best_sm = argmin[k].split("/")[0]
        pred = None
        for sm in SM:
            r, u = waves(ctas, int(sm[2:]), cpc)
            row += f"{r:>3}w/{u*100:>3.0f}%{'*' if sm == best_sm else ' '}{'':<5}"
            if r >= 2 and u >= 0.80:
                pred = sm
        print(row)
        hit = pred == best_sm
        hits += hit
        print(f"  {'':<31}predicted {str(pred):<7} measured {best_sm:<7}"
              f"{'HIT' if hit else 'MISS'}")
    print(f"\n  rule predicts {hits}/{len(kernels)} argmins")
    if hits < len(kernels):
        print("  -> quantization alone does NOT determine the optimum. EDP trades")
        print("     cycle gain against the power cost of more SMs, so the argmin")
        print("     can sit past a wave boundary. Report the coupling as measured,")
        print("     not as predicted-by-quantization.")
    print("  (* = measured optimum;  Nw/U% = rounds / resident-slot utilisation)")


if __name__ == "__main__":
    main()
