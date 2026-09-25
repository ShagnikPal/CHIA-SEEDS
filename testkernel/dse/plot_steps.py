"""Draw the co-design pipeline as a figure: every step, in order, coloured by
which arm decides it.

    python3 plot_steps.py <out_dir>

Five phases, left to right.  Blue where a rule decides, pink where a sandboxed
model decides.  The two arms sit side by side at step 5a, which is the one
place they compete and therefore the one place a head-to-head is possible.

The kernel forge is drawn at 5b, where it belongs: writing software points
against the bottleneck the run just found.  In the code today it runs as a
separate driver ahead of the loop and its tiles are read in at step 5 -- the
footnote says so, because a figure that quietly promotes future work to
present tense is the wrong kind of figure.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

W, H = 18.1, 8.6
INK, INK2 = "#0b0b0b", "#52514e"
KIND = {
    "prog":  ("#dae8fc", "#6c8ebf"),
    "agent": ("#fad2e4", "#b0407a"),
    "input": ("#f2f2f2", "#7a7a7a"),
    "gate":  ("#fff2cc", "#b8951f"),
}
FS_TITLE, FS_PHASE, FS_BOX, FS_DET = 16.5, 12.5, 11.0, 9.0

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def phase(x0, x1, y0, y1, label):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="#fbfbfa", edgecolor="#9a9a98", linewidth=1.3,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text((x0 + x1) / 2, y1 - 0.30, label, ha="center", va="center",
            fontsize=FS_PHASE, fontweight="bold", color=INK, zorder=5)


def box(cx, cy, w, h, title, detail=(), kind="prog", fs=FS_BOX,
        dfs=FS_DET):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.09",
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    top = cy + h / 2
    ax.text(cx, top - 0.27, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=5)
    for i, d in enumerate(detail):
        ax.text(cx, top - 0.56 - i * 0.235, d, ha="center", va="center",
                fontsize=dfs, color=INK2, zorder=5)
    return dict(cx=cx, cy=cy, w=w, h=h)


def arrow(p0, p1, lw=1.7, color=INK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=4))


# --- phases ----------------------------------------------------------------
PY0, PY1 = 2.40, 7.15
R1, R2, R3 = 6.00, 4.55, 3.10
BH, BW = 1.05, 2.55

cols = [(0.55, 3.35, "Measure the part"),
        (3.75, 6.55, "Find the limiter"),
        (6.95, 11.15, "Propose"),
        (11.55, 14.35, "Measure the proposals"),
        (14.75, 17.55, "Decide")]
for x0, x1, lab in cols:
    phase(x0, x1, PY0, PY1, lab)

cA, cB, cD, cE = 1.95, 5.15, 12.95, 16.15

# A -- measure the part
box(cA, R1, BW, BH, "0 · Baseline trace",
    ("a real A100, SASS captured",), kind="input")
box(cA, R2, BW, BH, "0b · Kernel gate",
    ("compile · 1e-4 · work", "· trace · simulate"))
box(cA, R3, BW, BH, "1 · Baseline measure",
    ("cycles · watts · energy", "· occupancy · CTA/SM"))

# B -- find the limiter
box(cB, R1, BW, BH, "2 · Probe sweep ×41",
    ("each knob to ∞", "and to near-zero"))
box(cB, R2, BW, BH, "3 · Attribution",
    ("stalls → one limiter",))
box(cB, R3, BW, BH, "4 · Counter study",
    ("causal vs. passenger", "24-simulation budget"), kind="agent")

# C -- propose
box(8.06, R1, 1.82, BH, "5a · Threshold\nproposer",
    ("a rule picks the knob",))
box(10.04, R1, 1.82, BH, "5a · LLM\narchitect",
    ("16-simulation budget",), kind="agent")
box(9.05, 4.85, 3.80, 0.80, "", (), kind="gate")
ax.text(9.05, 5.08, "Area gate — applies to the pair above",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="#6b5600", zorder=5)
for i, ln in enumerate(("a knob may grow only if a structure is certified",
                        "dead: every counter zero, no suppressing flag")):
    ax.text(9.05, 4.85 - i * 0.205, ln, ha="center", va="center",
            fontsize=8.5, color="#6b5600", zorder=5)
box(9.05, 3.35, 3.80, 1.15, "5b · Kernel forge",
    ("writes GEMM tiles against this run's bottleneck,",
     "each through the 0b gate, graded on 5a's hardware"),
    kind="agent", dfs=8.5)

# D -- measure the proposals
box(cD, R1, BW, BH, "6 · Build + trace",
    ("every proposed tile",))
box(cD, R2, BW, BH, "7 · Grid",
    ("every tile ×", "every hardware point"))
box(cD, R3, BW, BH, "8 · Composition",
    ("do two separate wins", "still win together"))

# E -- decide
box(cE, R1, BW, BH, "9 · SM band 104–112",
    ("sign-unstable ⇒", "quotable = False"))
box(cE, R2, BW, BH, "10 · Area ledger",
    ("added · unpriced · credit", "audit, not a veto"))
box(cE, R3, BW, BH, "Select",
    ("minimum EDP", "no secondary criterion"))

# --- flow ------------------------------------------------------------------
for x0, x1 in ((3.35, 3.75), (6.55, 6.95), (11.15, 11.55), (14.35, 14.75)):
    arrow((x0, R2), (x1, R2))

ax.add_patch(FancyArrowPatch(
    (16.15, PY0), (16.15, 1.85), arrowstyle="-", linewidth=1.7, color=INK,
    shrinkA=0, shrinkB=0, zorder=2))
ax.add_patch(FancyArrowPatch(
    (16.15, 1.85), (1.95, 1.85), arrowstyle="-", linewidth=1.7, color=INK,
    shrinkA=0, shrinkB=0, zorder=2))
arrow((1.95, 1.85), (1.95, PY0))
ax.text(9.05, 2.03, "next iteration: the loop re-enters with the measured grid",
        ha="center", va="center", fontsize=9.5, style="italic", color=INK2)

# --- title, legend, footnote ----------------------------------------------
ax.text(W / 2, 8.15, "CHIA · Accel-Sim co-design loop, step by step",
        ha="center", va="center", fontsize=FS_TITLE, fontweight="bold",
        color=INK)

lx = 4.60
for kind, label in (("prog", "programmatic: a rule decides"),
                    ("agent", "agentic: a sandboxed model decides"),
                    ("input", "fixed input"),
                    ("gate", "constraint that refuses a proposal")):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch((lx, 7.60), 0.32, 0.23,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(lx + 0.42, 7.715, label, ha="left", va="center", fontsize=10.0,
            color=INK)
    lx += 0.42 + len(label) * 0.083 + 0.55

ax.text(W / 2, 1.18,
        "Step 5b is drawn where it belongs. In the code today the forge runs "
        "as a separate driver ahead of the loop, briefed on a stored "
        "bottleneck and graded on stock hardware; its tiles are read in at "
        "step 5.",
        ha="center", va="center", fontsize=9.0, style="italic", color=INK2)

stem = os.path.join(out_dir, "chia_steps")
for ext in ("pdf", "png"):
    fig.savefig(f"{stem}.{ext}", dpi=200)
print(f"wrote {stem}.pdf and .png  ({W} x {H} in)")
