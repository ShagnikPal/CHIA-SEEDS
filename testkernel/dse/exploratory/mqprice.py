#!/usr/bin/env python3.10
"""Price deeper miss queues by upper bound, because McPAT cannot price them.

REJECTED METHOD. McPAT's cache controller does expose
`buffer_sizes = "miss, fill, prefetch, writeback"`, and the dcache sits at 16
-- exactly GPGPU-Sim's stock L1 miss_queue_size, which made it look like the
right dial.  It is not.  Bumping it 16->256 (L1) and 16->512 (L2) on an
otherwise byte-identical XML moves per-SM area by **-0.2447 mm^2**: enlarging
a buffer makes the SM *smaller*.  CACTI is re-optimising the organisation
under the new constraint rather than adding storage, so the number is not a
cost.  Non-monotonic means unusable, in either direction.

METHOD USED. Bound it from above and show the conclusion does not depend on
the bound being tight.

  entries added   L1: 16 -> 256 = +240/SM      L2: 32 -> 512 = +480/slice
  bytes/entry     16 B.  A miss-queue entry holds a request descriptor:
                  physical address (~40b), rw/type (4b), size (8b), sector
                  and byte mask (~32b), warp/SM/TPC ids (~24b), MSHR back-
                  pointer (~8b) = ~116b.  Round to 128b = 16 B.
  rate            5.64e-6 mm^2/B, the modelled L2 slice's own areal rate
                  (1.4782 mm^2 / 256 KiB).  A tagged, banked, multi-ported
                  cache costs MORE per byte than a FIFO, so charging the
                  FIFO at cache rates is already conservative.
  safety factor   x4 on top, for porting.  The texture-cache result showed
                  CACTI area is port-dominated rather than capacity-driven
                  (128 KB -> 16 KB freed only ~1.6 mm^2), so a small, heavily
                  ported structure is exactly where per-byte scaling would
                  understate.  x4 is there to make the bound indefensible to
                  argue with, not to be accurate.

Even so the knob costs under 1% of the chip.  That is the finding: the
structure gating this workload is among the cheapest on the die.
"""
KIB = 1024
RATE = 1.4782 / (256 * KIB)
BYTES_PER_ENTRY = 16
SAFETY = 4.0

N_SM, N_L2 = 108, 160
PER_SM, PER_L2, FIXED = 42.7287, 1.4782, 15.731
TEX = 13.936
STOCK_TOTAL = 4866.95


def cost(entries_added: int) -> float:
    return entries_added * BYTES_PER_ENTRY * RATE * SAFETY


def main() -> None:
    l1 = cost(256 - 16)
    l2 = cost(512 - 32)
    print(f"rate {RATE * 1e6:.4f} um^2/B  x{SAFETY:g} safety  "
          f"{BYTES_PER_ENTRY} B/entry\n")
    print(f"  L1 miss queue  16 -> 256  (+240 entries)  {l1:+.4f} mm^2/SM")
    print(f"  L2 miss queue  32 -> 512  (+480 entries)  {l2:+.4f} mm^2/slice")
    chip = N_SM * l1 + N_L2 * l2
    print(f"  chip-wide at stock 108 SM / 160 slices    {chip:+.2f} mm^2"
          f"   ({100 * chip / STOCK_TOTAL:.2f}% of chip)")
    print(f"  texture credit available                  {N_SM * TEX:.0f} mm^2")
    print(f"  queue costs {100 * l1 / TEX:.2f}% of the per-SM credit"
          f"  ({TEX / l1:.0f}:1)\n")

    print(f"iso-area solve against stock {STOCK_TOTAL:.2f} mm^2:")
    print(f"  {'design':<34}{'/SM':>8}{'/L2':>7}{'SMs':>6}")
    for name, per_sm, per_l2 in (
        ("stock", PER_SM, PER_L2),
        ("mq16 only (no tex credit)", PER_SM + l1, PER_L2 + l2),
        ("de-texture only", PER_SM - TEX, PER_L2),
        ("mq16 + de-texture", PER_SM - TEX + l1, PER_L2 + l2),
    ):
        room = STOCK_TOTAL - FIXED - N_L2 * per_l2
        n = int(room // per_sm)
        print(f"  {name:<34}{per_sm:>8.3f}{per_l2:>7.3f}{n:>6}")


if __name__ == "__main__":
    main()
