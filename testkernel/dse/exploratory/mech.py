"""Attribute cache reservation failures by *reason*, not by count.

Counting reservation failures names the cache; splitting them by cause names
the structure inside it.  That distinction is the whole reason this study
landed on the miss queue rather than on MSHRs: at 4096x4096x64 every single
L1 and L2 RESERVATION_FAIL was a MISS_QUEUE_FULL and none was an MSHR
shortage, which is why every MSHR knob measured 0.00%.

Reads the final row of perf_counter.csv.gz (cumulative counters) and sums the
per-bank / per-SM columns into one row per (cache, access type, reason).

Usage:  mech.py <sim_outdir> [<sim_outdir> ...]
"""
import csv
import gzip
import sys

REASONS = ("RESERVATION_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
           "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING", "LINE_ALLOC_FAIL")
UNITS = ("L1D", "L2")
RW = ("R", "W")


def attribute(path: str) -> dict:
    """Sum cumulative counters from the last row, keyed (unit, rw, reason)."""
    with gzip.open(f"{path}/perf_counter.csv.gz", "rt") as fh:
        rdr = csv.reader(fh)
        head = next(rdr)
        last = None
        for row in rdr:
            if len(row) >= len(head):
                last = row
    if last is None:
        raise SystemExit(f"{path}: no data rows")


    want = {}
    for i, c in enumerate(head):
        for u in UNITS:
            if not c.startswith(u + "_"):
                continue
            for rw in RW:
                for r in REASONS:
                    if c.endswith(f"GLOBAL_ACC_{rw}_{r}"):
                        want.setdefault((u, rw, r), []).append(i)
    out = {}
    for k, idx in want.items():
        tot = 0.0
        for i in idx:
            try:
                tot += float(last[i])
            except (ValueError, IndexError):
                pass
        out[k] = tot
    return out


def report(path: str) -> None:
    a = attribute(path)
    print(f"\n=== {path}")
    print(f"  {'cache':<6}{'op':<4}{'RESERVATION_FAIL':>18}"
          f"{'MISS_QUEUE_FULL':>18}{'share':>8}{'MSHR-related':>14}{'share':>8}")
    for u in UNITS:
        for rw in RW:
            rf = a.get((u, rw, "RESERVATION_FAIL"), 0.0)
            mq = a.get((u, rw, "MISS_QUEUE_FULL"), 0.0)
            mshr = sum(a.get((u, rw, r), 0.0) for r in
                       ("MSHR_ENRTY_FAIL", "MSHR_MERGE_ENRTY_FAIL",
                        "MSHR_RW_PENDING"))
            if rf == 0 and mq == 0:
                continue
            print(f"  {u:<6}{rw:<4}{rf:>18,.0f}{mq:>18,.0f}"
                  f"{(100 * mq / rf if rf else 0):>7.1f}%"
                  f"{mshr:>14,.0f}{(100 * mshr / rf if rf else 0):>7.1f}%")

    wf = sum(a.get((u, "W", "RESERVATION_FAIL"), 0.0) for u in UNITS)
    rfa = sum(a.get((u, "R", "RESERVATION_FAIL"), 0.0) for u in UNITS)
    tot = wf + rfa
    if tot:
        print(f"  -> write path is {100 * wf / tot:.1f}% of all "
              f"reservation failures ({wf:,.0f} of {tot:,.0f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)
