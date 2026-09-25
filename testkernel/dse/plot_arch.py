"""Draw the whole framework: the loop on top, the cluster it runs on below.

    python3 plot_arch.py <out_dir>        # chia_accelsim_workflow.{pdf,png}

Two bands, two meanings for colour.

  * In the loop, colour says *who decides*: blue where a rule decides, pink
    where a sandboxed model decides, yellow where a constraint refuses.
  * In the cluster, colour says *what kind of machine*: a cloud for the model
    provider, green for GPU nodes, amber for CPU nodes.

The loop follows the step table exactly -- 0 through 10, with the kernel forge
at 5b where it belongs, writing software points against the bottleneck the run
just found.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

W, H = 18.1, 10.4
INK, INK2 = "#0b0b0b", "#52514e"
KIND = {
    "prog":   ("#dae8fc", "#6c8ebf"),
    "agent":  ("#fad2e4", "#b0407a"),
    "input":  ("#f2f2f2", "#7a7a7a"),
    "gate":   ("#fff2cc", "#b8951f"),
    "worker": ("#ececec", "#333333"),
}
GPU_BG, GPU_FG = "#d9ead3", "#6aa84f"
CPU_BG, CPU_FG = "#fff2cc", "#f1c232"
CLOUD = "#33e1ee"
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
    ax.text((x0 + x1) / 2, y1 - 0.28, label, ha="center", va="center",
            fontsize=FS_PHASE, fontweight="bold", color=INK, zorder=5)


def box(cx, cy, w, h, title, detail=(), kind="prog", fs=FS_BOX, dfs=FS_DET,
        lw=1.6, tcol=None, dcol=None, top_pad=0.26, lead=0.225):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.09",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    top = cy + h / 2
    ax.text(cx, top - top_pad, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tcol or INK, zorder=5)
    for i, d in enumerate(detail):
        ax.text(cx, top - top_pad - 0.28 - i * lead, d, ha="center",
                va="center", fontsize=dfs, color=dcol or INK2, zorder=5)
    return dict(cx=cx, cy=cy, w=w, h=h)


def arrow(p0, p1, lw=1.7, color=INK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=4))


def cloud(cx, cy, s=1.35):
    """One silhouette, not a pile of circles: sample each circle and keep the
    arc that no other circle covers, then walk the survivors by angle."""
    cs = [(-0.62, -0.05, 0.30), (-0.42, -0.10, 0.34), (-0.30, 0.14, 0.36),
          (-0.10, -0.06, 0.42), (0.05, 0.20, 0.34), (0.28, -0.10, 0.34),
          (0.40, 0.08, 0.30), (0.62, -0.08, 0.26)]
    pts = []
    for x, y, r in cs:
        for k in range(240):
            t = 2 * math.pi * k / 240
            px, py = x + r * math.cos(t), y + r * math.sin(t)
            if all((px - a) ** 2 + (py - b) ** 2 >= c * c - 1e-6
                   for a, b, c in cs if (a, b, c) != (x, y, r)):
                pts.append((px, py))
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    poly = [(cx + px * s, cy + py * s) for px, py in pts]
    ax.add_patch(Polygon(poly, closed=True, facecolor=CLOUD,
                         edgecolor="#111111", linewidth=1.8, zorder=3))
    return dict(cx=cx, cy=cy, top=cy + 0.55 * s)


# ===========================================================================
# the loop -- the step table, in order
# ===========================================================================
PY0, PY1 = 5.55, 9.20
R1, R2, R3 = 8.35, 7.20, 6.05
BH, BW = 0.92, 2.55

cols = [(0.55, 3.35, "Measure the part"),
        (3.75, 6.55, "Find the limiter"),
        (6.95, 11.15, "Propose"),
        (11.55, 14.35, "Measure the proposals"),
        (14.75, 17.55, "Decide")]
for x0, x1, lab in cols:
    phase(x0, x1, PY0, PY1, lab)
cA, cB, cD, cE = 1.95, 5.15, 12.95, 16.15

box(cA, R1, BW, BH, "0 · Baseline trace",
    ("a real A100, SASS captured",), kind="input")
box(cA, R2, BW, BH, "0b · Kernel gate",
    ("compile · 1e-4 · work · trace",))
box(cA, R3, BW, BH, "1 · Baseline measure",
    ("cycles · watts · energy · occupancy",), dfs=8.3)

box(cB, R1, BW, BH, "2 · Probe sweep ×41",
    ("each knob to ∞ and to near-zero",), dfs=8.3)
box(cB, R2, BW, BH, "3 · Attribution", ("stalls → one limiter",))
box(cB, R3, BW, BH, "4 · Counter study",
    ("causal vs. passenger · 24 sims",), kind="agent", dfs=8.5)

box(8.06, R1, 1.82, BH, "5a · Threshold\nproposer", (), top_pad=0.34)
box(10.04, R1, 1.82, BH, "5a · LLM\narchitect", (), kind="agent",
    top_pad=0.34)
box(9.05, R2, 3.80, 0.74, "", (), kind="gate")
ax.text(9.05, 7.38, "Area gate — applies to the pair above", ha="center",
        va="center", fontsize=9.5, fontweight="bold", color="#6b5600",
        zorder=5)
for i, ln in enumerate(("a knob may grow only if a structure is certified",
                        "dead: every counter zero, no suppressing flag")):
    ax.text(9.05, 7.16 - i * 0.20, ln, ha="center", va="center", fontsize=8.5,
            color="#6b5600", zorder=5)
box(9.05, R3, 3.80, BH, "5b · Kernel forge",
    ("writes GEMM tiles against this run's bottleneck,",
     "each through the 0b gate, graded on 5a's hardware"),
    kind="agent", dfs=8.5, lead=0.21)

box(cD, R1, BW, BH, "6 · Build + trace", ("every proposed tile",))
box(cD, R2, BW, BH, "7 · Grid", ("every tile × every hardware point",),
    dfs=8.0)
box(cD, R3, BW, BH, "8 · Composition", ("do two wins still win together",),
    dfs=8.3)

box(cE, R1, BW, BH, "9 · SM band 104–112",
    ("sign-unstable ⇒ quotable = False",), dfs=8.3)
box(cE, R2, BW, BH, "10 · Area ledger", ("audit, not a veto",))
box(cE, R3, BW, BH, "Select", ("minimum EDP",), dfs=8.0)

for x0, x1 in ((3.35, 3.75), (6.55, 6.95), (11.15, 11.55), (14.35, 14.75)):
    arrow((x0, R2), (x1, R2))

for p0, p1 in (((16.15, PY0), (16.15, 5.15)), ((16.15, 5.15), (1.95, 5.15))):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-", linewidth=1.7,
                                 color=INK, shrinkA=0, shrinkB=0, zorder=2))
arrow((1.95, 5.15), (1.95, PY0))
ax.text(9.05, 5.33, "next iteration: the loop re-enters with the measured grid",
        ha="center", va="center", fontsize=9.5, style="italic", color=INK2)

# ===========================================================================
# the cluster -- who runs what, and where
# ===========================================================================
ax.text(W / 2, 4.66, "CHIA Cluster", ha="center", va="center",
        fontsize=FS_PHASE, fontweight="bold", color=INK)

WY, WH, WW = 4.05, 0.86, 2.45
workers = [
    (2.10, "LLM Model", "steps 4 · 5a · 5b", "claude_creds = 1  ·  ×1"),
    (6.25, "Kernel Tracer", "steps 0 · 6", "accelsim_gpu = 1  ·  ×8"),
    (10.55, "Kernel Builder", "steps 0b · 6", "accelsim = 1  ·  ×8"),
    (13.25, "GPU Simulator", "steps 0b · 2 · 7 · 8 · 9",
     "accelsim = 1  ·  ×100"),
    (15.95, "Power / Area", "steps 1 · 10", "no resource  ·  ×100"),
]
for x, name, steps, res in workers:
    box(x, WY, WW, WH, name, (steps, res), kind="worker", fs=10.5, dfs=8.2,
        lw=1.8, top_pad=0.22, lead=0.20)

MY0, MY1 = 1.15, 2.70
for x0, x1, bg, fg, lab, names in (
        (3.90, 8.60, GPU_BG, GPU_FG, "GPU Nodes",
         ["GPU-0", "GPU-1", None, "GPU-7"]),
        (9.05, 17.55, CPU_BG, CPU_FG, "CPU Nodes",
         ["CPU-0", "CPU-1", None, "CPU-100"])):
    ax.add_patch(FancyBboxPatch(
        (x0, MY0), x1 - x0, MY1 - MY0,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=bg, edgecolor="#111111", linewidth=1.8, zorder=2))
    ax.text((x0 + x1) / 2, MY1 - 0.26, lab, ha="center", va="center",
            fontsize=12.0, color=INK, zorder=5)
    # Packed at a fixed pitch and centred, so a wide container does not
    # stretch four nodes across half a metre of nothing.
    BW_N, GAP, DOT = 1.04, 0.30, 0.60
    widths = [DOT if nm is None else BW_N for nm in names]
    total = sum(widths) + GAP * (len(names) - 1)
    x = (x0 + x1) / 2 - total / 2
    for nm, wd in zip(names, widths):
        cx = x + wd / 2
        x += wd + GAP
        if nm is None:
            for d in (-0.20, 0.0, 0.20):
                ax.add_patch(Circle((cx + d, 1.74), 0.075, facecolor=fg,
                                    edgecolor="none", zorder=4))
            continue
        ax.add_patch(FancyBboxPatch(
            (cx - BW_N / 2, 1.43), BW_N, 0.62,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fg, edgecolor="#111111", linewidth=1.5, zorder=3))
        ax.text(cx, 1.74, nm, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK, zorder=5)

cl = cloud(2.10, 1.82)
ax.text(2.10, 1.82, "LLM\nProvider", ha="center", va="center", fontsize=11.5,
        fontweight="bold", color=INK, zorder=5, linespacing=1.15)

ax.text(2.10, 0.72, "Cloud", ha="center", va="center", fontsize=12.0,
        fontweight="bold", color="#3f3f3f")
ax.text(10.72, 0.72, "Local Machines", ha="center", va="center",
        fontsize=12.0, fontweight="bold", color="#3f3f3f")

arrow((2.10, WY - WH / 2), (2.10, cl["top"] + 0.04))
for x in (6.25, 10.55, 13.25, 15.95):
    arrow((x, WY - WH / 2), (x, MY1 + 0.02))

# ===========================================================================
ax.text(W / 2, 10.05, "CHIA · Accel-Sim GPU backend — the loop, and the "
                      "cluster it runs on",
        ha="center", va="center", fontsize=FS_TITLE, fontweight="bold",
        color=INK)
lx = 4.60
for kind, label in (("prog", "programmatic: a rule decides"),
                    ("agent", "agentic: a sandboxed model decides"),
                    ("input", "fixed input"),
                    ("gate", "constraint that refuses a proposal")):
    fc, ec = KIND[kind]
    ax.add_patch(FancyBboxPatch((lx, 9.52), 0.32, 0.23,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(lx + 0.42, 9.635, label, ha="left", va="center", fontsize=10.0,
            color=INK)
    lx += 0.42 + len(label) * 0.083 + 0.55

ax.text(W / 2, 0.26,
        "Step 5b is drawn where it belongs. In the code today the forge runs "
        "as a separate driver ahead of the loop, briefed on a stored "
        "bottleneck and graded on stock hardware; its tiles are read in at "
        "step 5.",
        ha="center", va="center", fontsize=9.0, style="italic", color=INK2)

stem = os.path.join(out_dir, "chia_accelsim_workflow")
for ext in ("pdf", "png"):
    fig.savefig(f"{stem}.{ext}", dpi=200)
print(f"wrote {stem}.pdf and .png  ({W} x {H} in)")
