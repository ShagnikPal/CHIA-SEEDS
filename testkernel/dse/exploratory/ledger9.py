"""Step 9 standalone: price the winner the running loop could not.

The loop process imported accelsim_codesign before the area ledger existed,
so it will name a winner but not price it.  This reads the journal the loop
wrote and runs the ledger against the winner's own output directory.
"""
import json
import sys

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
from chia.simulators.accelsim_codesign import area_ledger, format_ledger

OUT = "/raid/sp56429/chia_hackathon/testkernel/dse/out_chia64"


def main():
    journal = json.load(open(f"{OUT}/journal.json"))
    design = journal.get("design") or {}
    base_dir = journal.get("baseline", {}).get("outdir") or f"{OUT}/base/pts/baseline"
    win_dir = design.get("outdir")
    if not win_dir:
        print("journal has no design.outdir yet -- loop still running?")
        print(json.dumps({k: v for k, v in journal.items() if k != "points"},
                         indent=2)[:2000])
        return 1
    n_sm = design.get("n_sm", journal.get("cfg", {}).get("n_sm", 108))
    print(f"baseline dir : {base_dir}")
    print(f"winner  dir  : {win_dir}   (n_sm={n_sm})\n")


    lg = area_ledger(base_dir, win_dir, n_sm=n_sm, n_l2=160)
    print(format_ledger(lg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
