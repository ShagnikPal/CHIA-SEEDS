#!/usr/bin/env python3
"""Plot the HW x SW design grid as a heatmap, one figure per metric.

    plot_grid.py <run_dir> [metric ...]      # metric: edp | perf | energy | all

``run_dir`` is any directory holding a ``datapoints.csv`` written by
``run_dse.py`` -- that is, any ``DSE_OUT``.  With no metric named, all three
are written, as ``grid_edp.png``, ``grid_perf.png`` and ``grid_energy.png``.

A grid of measured magnitudes is a heatmap, so cells rather than bubbles: the
lattice is the data's own shape and a cell states it without asking the reader
to judge an area.  The grid is not guaranteed dense -- ``codesign_loop`` only
builds the pairs it has a reason to measure -- so a pair that was never run is
left as bare surface with a hairline, which reads as absent rather than as a
zero.

Colour is one blue ramp, light to dark, so magnitude is carried by one hue in
one direction.  Every metric here is lower-is-better, which means the *pale*
cells are the good ones and the ink pools on the bad; that is the correct
sequential reading (light = small), and it is why the chosen design is ringed
rather than left to be found by colour.

The three figures share their row and column order, their rings and their
scale direction, so a cell can be compared across them.  That order is set
once, by EDP -- the quantity the loop actually selects on -- and not re-derived
per metric, which would silently shuffle the grid between figures.
"""
import csv
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Chart surface and ink.  Text never wears the data colour: values and labels
# stay in these two, and identity comes from the cell they sit in.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
# One hue, light -> dark: steps 100..700 of the blue ramp.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)

# Two sizes, no third.  Titles name an axis; everything else -- the numbers in
# the grid, the legend, the axis key, the scale ticks -- is one body size, so
# nothing the reader has to actually read is set smaller than anything else.
FS_TITLE, FS_BODY, FS_SMALL = 27, 24, 24

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": FS_BODY,
    "text.color": INK,
    "axes.edgecolor": "#d8d7d2",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    # Type 42, not the Matplotlib default Type 3: camera-ready checkers reject
    # Type 3, and it is what makes the text in the PDF selectable and
    # searchable rather than drawn.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

if len(sys.argv) < 2:
    print(__doc__.strip())
    raise SystemExit(2)
run = sys.argv[1]
csv_path = run if run.endswith(".csv") else os.path.join(run, "datapoints.csv")
out_dir = os.path.dirname(os.path.abspath(csv_path))
want = [a for a in sys.argv[2:]] or ["all"]
if "all" in want:
    want = ["edp", "perf", "energy"]

if not os.path.exists(csv_path):
    print(f"no datapoints.csv at {csv_path}")
    raise SystemExit(2)

with open(csv_path) as fh:
    rows = [r for r in csv.DictReader(fh) if r.get("edp")]
if not rows:
    print(f"{csv_path} has no rows with an EDP")
    raise SystemExit(2)

for r in rows:
    r["edp"] = float(r["edp"]) * 1e6          # J.s -> uJ.s
    r["cycles"] = float(r["cycles"])
    r["kcycles"] = r["cycles"] / 1e3
    r["energy_mj"] = float(r["energy_mj"])
    r["gain"] = float(r["gain_vs_baseline_pct"])
    # Kept as recorded, but the grids print -gain: all three plots read as a
    # signed change against the baseline, so negative is better everywhere and
    # no cell needs a legend to say which direction it is pointing.
    r["cchg"] = -r["gain"]


# Tick labels say what the point *is*, not what the loop called it.  The
# internal names ("agent_mem_ports4", "t128x128x16_s4") are how the runs are
# keyed and they stay in the CSV; a reader should not have to learn them to
# read the picture.  Unknown names fall through to their raw form rather than
# being guessed at, so a run with different points still plots.
# word -> (tick label, what it means).  The tick stays short enough to sit
# level under its column; the meaning goes in the key under the axis, where
# there is room for a sentence.
HW_WORDS = {
    "queue": ("Queue", "L1/L2 miss queue"),
    "occ": ("Occ", "resident blocks per SM"),
    "wb": ("WB", "writeback width"),
    "mem_ports": ("Ports", "memory ports"),
    "mshr": ("MSHR", "outstanding misses"),
    "l1_capacity": ("L1", "L1 cache capacity"),
}


def hw_words(name):
    """"agent_mem_ports4" -> [("mem_ports", "4")]; [] for the stock config."""
    got = []
    for p in name.split("+"):
        m = re.match(r"^([a-z][a-z_]*?)(\d+)$", p.replace("agent_", ""))
        if m:
            got.append((m.group(1), m.group(2)))
    return got


def pretty_hw(name):
    """"queue4+occ2" -> "Queue ×4 + Occ ×2"."""
    if name == "stock":
        return "Baseline"
    bits = [f"{HW_WORDS.get(w, (w.replace('_', ' '), ''))[0]} ×{n}"
            for w, n in hw_words(name)]
    if not bits:
        return name
    if len(bits) == 1:                # "Queue\n×16": the column is as
        return bits[0].replace(" ", "\n")     # wide as the word, not the sum
    return "\n+ ".join(bits)          # a combined point stacks, not widens


def pretty_tile(name):
    """"t128x128x16_s4" -> "128x128x16 . 4" (shape, then pipeline stages)."""
    if name == "base":
        return "Baseline"
    m = re.match(r"^t(\d+)x(\d+)x(\d+)_s(\d+)$", name)
    if not m:
        return name
    a, b, c, st = m.groups()
    return f"{a}×{b}×{c} · {st}"


# Order both axes by their best point, so the good corner is top-left -- the
# figure then reads in the same direction as the text around it, instead of in
# whatever order the loop happened to dispatch.
def axis(key):
    best = {}
    for r in rows:
        best[r[key]] = min(best.get(r[key], float("inf")), r["edp"])
    return [k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])]


hw, sw = axis("hw_point")[::-1], axis("tile")
xi = {k: i for i, k in enumerate(hw)}
yi = {k: i for i, k in enumerate(sw)}
cell = {(r["hw_point"], r["tile"]): r for r in rows}

# The ring marks what the loop chose, on the loop's own rule: minimum EDP,
# with no secondary criterion.  Reproducing that rule here keeps the figure
# consistent with the tables rather than re-deriving a winner of its own.  The
# same design is ringed on all three figures, including the ones it does not
# win on.
win = min(rows, key=lambda r: r["edp"])
base = cell.get(("stock", "base"))

# Energy saving is measured against the same baseline the loop reports speedup
# against, so the two percentages are read off one reference point.
base_e = base["energy_mj"] if base else max(r["energy_mj"] for r in rows)
base_d = base["edp"] if base else max(r["edp"] for r in rows)
for r in rows:
    r["esave"] = (base_e - r["energy_mj"]) / base_e * 100.0
    r["echg"] = (r["energy_mj"] - base_e) / base_e * 100.0
    # Signed as a change, not a gain: EDP fell by 10.9% reads -10.9, so the
    # number carries its own direction and no legend is needed to decode it.
    r["edpchg"] = (r["edp"] - base_d) / base_d * 100.0

# Three views of one grid.  ``shade`` is what the colour carries, ``label`` is
# the number printed in the cell: in every case the percentage form of the same
# quantity the colour carries, so the reader gets both the size and the change
# and the ringed winner agrees with the number printed inside it.  EDP printed
# cycle speedup until this was fixed, which let the two disagree by 3.2 points
# and put the ring on a different cell from the best number on the plot.
METRICS = {
    "edp": dict(
        file="grid_edp", shade="edp", tick="{:.2f}",
        cb="EDP (µJ·s), lower is better",
        label=lambda r: f"{r['edpchg']:+.1f}%",
        note="Cell = % change in EDP vs baseline; lower is better"),
    "perf": dict(
        file="grid_perf", shade="kcycles", tick="{:.0f}",
        cb="Cycles (thousands), lower is better",
        label=lambda r: f"{r['cchg']:+.1f}%",
        note="Cell = % change in cycles vs baseline; lower is better"),
    "energy": dict(
        file="grid_energy", shade="energy_mj", tick="{:.2f}",
        cb="Energy (mJ), lower is better",
        label=lambda r: f"{r['echg']:+.1f}%",
        note="Cell = % change in energy vs baseline; lower is better"),
}

# The x ticks are abbreviations, so the axis carries its own key.  Only the
# words actually on this axis are defined, in the order they appear.
seen, key_items = set(), []
for h in hw:
    for w, _ in hw_words(h):
        if w in seen:
            continue
        seen.add(w)
        short, gloss = HW_WORDS.get(w, (w.replace("_", " "), ""))
        key_items.append(f"{short} = {gloss}" if gloss else short)

x_labels = [pretty_hw(h) for h in hw]
y_labels = [pretty_tile(t) for t in sw]
X_TITLE = "Hardware point: structure · size multiple"
Y_TITLE = "Software point: tile M×N×K · stages"

# ---------------------------------------------------------------------------
# Layout is measured, not guessed.  Every band of the figure -- row labels,
# colour scale, the key under the axis -- is measured at the size it will
# actually be drawn in, and the canvas is then built to exactly that size.
# Hand-tuning a figsize is what produced both the collisions and the dead
# margins: raising the type ate the margins and the grid silently shrank.
PAD = 0.14                                       # outer margin, inches
GAP = 0.34                                       # least air between key terms
BAR = 0.26                                       # colour-scale bar width
_probe = plt.figure(figsize=(1, 1), dpi=100)
_rend = _probe.canvas.get_renderer()


def measure(txt, fs, lead=1.2):
    """Width and height of ``txt`` at ``fs`` points, in inches, as drawn."""
    t = _probe.text(0, 0, txt, fontsize=fs, linespacing=lead)
    bb = t.get_window_extent(renderer=_rend)
    t.remove()
    return bb.width / 100.0, bb.height / 100.0


def render(name, spec):
    """Draw one metric's figure and return the path it was written to."""
    shade = spec["shade"]
    lo = min(r[shade] for r in rows)
    hi = max(r[shade] for r in rows)
    norm = Normalize(vmin=lo, vmax=hi)
    cb_at = [lo + (hi - lo) * f for f in (0, .25, .5, .75, 1)]
    cb_text = [spec["tick"].format(v) for v in cb_at]

    xt_w = max(measure(t, FS_BODY, 1.3)[0] for t in x_labels)
    xt_h = max(measure(t, FS_BODY, 1.3)[1] for t in x_labels)
    yt_w = max(measure(t, FS_BODY)[0] for t in y_labels)
    val_w = max(measure(spec["label"](r), FS_BODY)[0] for r in rows)
    line_h = measure("Hg", FS_BODY)[1]

    # A cell is as wide as the widest thing that has to sit under it or inside
    # it, plus breathing room, and near enough square that neither axis reads
    # stretched.
    cell_w = max(val_w + 0.46, xt_w + 0.18)
    cell_h = max(line_h * 1.9, 0.52 * cell_w)    # square-ish, never a ribbon
    grid_w, grid_h = cell_w * len(hw), cell_h * len(sw)

    def fit_side(txt):
        """A rotated title, wrapped onto as many lines as the grid height needs.

        Both side titles read bottom-to-top along the grid, so the grid height
        is their line length.  Wrapping costs a little width and keeps the type
        at title size; stretching the grid to fit one long line would not.
        """
        words = txt.split(" ")
        for n in range(1, len(words) + 1):
            per = math.ceil(len(words) / n)
            lines = [" ".join(words[i:i + per])
                     for i in range(0, len(words), per)]
            wrapped = "\n".join(lines)
            if measure(wrapped, FS_TITLE, 1.25)[0] <= grid_h - 0.30:
                return wrapped
        return txt

    y_title, cb_title = fit_side(Y_TITLE), fit_side(spec["cb"])

    legend_h = measure("Hg", FS_SMALL)[1] * 1.45
    xtl_h = measure(X_TITLE, FS_TITLE)[1]
    ytl_h = measure(y_title, FS_TITLE, 1.25)[1]
    cbl_h = measure(cb_title, FS_TITLE, 1.25)[1]
    cbt_w = max(measure(t, FS_SMALL)[0] for t in cb_text)

    left = PAD + ytl_h + 0.12 + yt_w + 0.14
    right = 0.30 + BAR + 0.10 + cbt_w + 0.12 + cbl_h + PAD
    top = PAD + legend_h + 0.10

    # The key is justified across the full width of the sheet, not set as a
    # centred paragraph.  A centred line leaves both bottom corners empty; a
    # justified one reaches each edge, and it does not drag the canvas wider
    # than the grid -- letting one long line set the width is exactly what
    # makes the type look small.
    avail = left + grid_w + right - 2 * PAD
    item_w = [measure(t, FS_SMALL)[0] for t in key_items]
    key_rows = [list(range(len(key_items)))]
    for n in range(1, len(key_items) + 1):
        per = math.ceil(len(key_items) / n)
        key_rows = [list(range(i, min(i + per, len(key_items))))
                    for i in range(0, len(key_items), per)]
        if all(sum(item_w[i] for i in r) + GAP * (len(r) - 1) <= avail
               for r in key_rows):
            break
    key_line = measure("Hg", FS_SMALL)[1] * 1.25
    key_h = key_line * len(key_rows)

    bottom = PAD + key_h + 0.12 + xtl_h + 0.12 + xt_h + 0.08
    W = left + grid_w + right
    H = bottom + grid_h + top

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, grid_w / W, grid_h / H])

    # 2px of surface does the separating between cells; no stroke around a mark.
    g = 0.022
    for (h, t), r in cell.items():
        x, y = xi[h], yi[t]
        rgba = CMAP(norm(r[shade]))
        ax.add_patch(Rectangle((x - 0.5 + g, y - 0.5 + g), 1 - 2 * g,
                               1 - 2 * g, facecolor=rgba, edgecolor="none",
                               zorder=2))
        # Inside a filled cell, pick ink or white by the fill's luminance so
        # the value always clears contrast.
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(x, y, spec["label"](r), ha="center", va="center",
                fontsize=FS_BODY, color=(INK if lum > 0.55 else "#ffffff"),
                zorder=4)

    for h in hw:                                 # pairs that were never run
        for t in sw:
            if (h, t) in cell:
                continue
            ax.add_patch(Rectangle((xi[h] - 0.5 + g, yi[t] - 0.5 + g),
                                   1 - 2 * g, 1 - 2 * g, facecolor=SURFACE,
                                   edgecolor="#d8d7d2", linewidth=1, zorder=2))

    for r, style, lw in ((base, (0, (4, 2)), 2.0), (win, "solid", 2.6)):
        if r is None:
            continue
        ax.add_patch(Rectangle((xi[r["hw_point"]] - 0.5 + g,
                                yi[r["tile"]] - 0.5 + g),
                               1 - 2 * g, 1 - 2 * g, facecolor="none",
                               edgecolor=INK, linewidth=lw, linestyle=style,
                               zorder=5))

    ax.set_xticks(range(len(hw)))
    ax.set_xticklabels(x_labels, fontsize=FS_BODY, ha="center", color=INK,
                       linespacing=1.3)
    ax.set_yticks(range(len(sw)))
    ax.set_yticklabels(y_labels, fontsize=FS_BODY, color=INK)
    ax.set_xlim(-0.5, len(hw) - 0.5)
    ax.set_ylim(-0.5, len(sw) - 0.5)
    ax.tick_params(length=0, pad=5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Titles, key and legend are positioned in the same inches the bands were
    # reserved in, so nothing can drift into anything else.
    fig.text((left + grid_w / 2) / W, (PAD + key_h + 0.16 + xtl_h / 2) / H,
             X_TITLE, ha="center", va="center", fontsize=FS_TITLE, color=INK)
    fig.text((PAD + ytl_h / 2) / W, (bottom + grid_h / 2) / H, y_title,
             ha="center", va="center", rotation=90, fontsize=FS_TITLE,
             color=INK, linespacing=1.25)
    for j, r in enumerate(key_rows):             # justified, edge to edge
        y = (PAD + key_h - (j + 0.5) * key_line) / H
        step = ((avail - sum(item_w[i] for i in r)) / (len(r) - 1)
                if len(r) > 1 else 0)
        x = PAD
        for i in r:
            fig.text(x / W, y, key_items[i], ha="left", va="center",
                     fontsize=FS_SMALL, color=INK2)
            x += item_w[i] + step

    cax = fig.add_axes([(left + grid_w + 0.30) / W, bottom / H, BAR / W,
                        grid_h / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    # Five ticks, not fourteen: the bar keys the shading, not a second axis.
    cb.set_ticks(cb_at)
    cb.ax.set_yticklabels(cb_text)
    cb.ax.tick_params(labelsize=FS_SMALL, length=0, colors=INK, pad=4)
    cb.outline.set_visible(False)
    fig.text((left + grid_w + 0.30 + BAR + 0.10 + cbt_w + 0.12 + cbl_h / 2) / W,
             (bottom + grid_h / 2) / H, cb_title, ha="center", va="center",
             rotation=90, fontsize=FS_TITLE, color=INK, linespacing=1.25)

    ly = (H - PAD - legend_h / 2) / H
    lx = (PAD + ytl_h + 0.12) / W
    for label, style, lw in (("Selected design", "solid", 2.6),
                             ("Baseline", (0, (4, 2)), 2.0)):
        fig.add_artist(Line2D([lx, lx + 0.30 / W], [ly, ly], color=INK, lw=lw,
                              linestyle=style, transform=fig.transFigure))
        fig.text(lx + 0.40 / W, ly, label, ha="left", va="center",
                 fontsize=FS_SMALL, color=INK)
        lx += (0.40 + measure(label, FS_SMALL)[0] + 0.55) / W
    fig.text((W - PAD) / W, ly, spec["note"], ha="right", va="center",
             fontsize=FS_SMALL, color=INK2)

    # PDF for anything that gets typeset -- vector, so it stays sharp at any
    # column width -- and PNG for anything that only gets looked at.
    stem = os.path.join(out_dir, spec["file"])
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf and .png  ({W:.1f} x {H:.1f} in)")
    return stem


print(f"{len(rows)} points, {len(hw)} hardware x {len(sw)} software "
      f"({len(hw) * len(sw) - len(rows)} pairs not measured)")
print(f"selected {win['tile']}/{win['hw_point']}  "
      f"{win['cycles']:.0f} cycles, {win['energy_mj']:.2f} mJ, "
      f"EDP {win['edp']:.4g} uJ.s  "
      f"({win['cchg']:+.2f}% cycles, {win['echg']:+.2f}% energy, "
      f"{win['edpchg']:+.2f}% EDP)")
for m in want:
    if m not in METRICS:
        print(f"unknown metric {m!r}; pick from {', '.join(METRICS)}")
        raise SystemExit(2)
    render(m, METRICS[m])
