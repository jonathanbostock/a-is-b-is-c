"""V5: the recipe figure — three scales, plus the Qwen-14B ablation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch

# --- skill palette ---
VR_PALETTE      = ["#2f6175", "#c3a322", "#1a6b06", "#c70053", "#e8642c", "#865ecf"]
VR_PLOT_BG      = "#fafcf2"
VR_INK          = "#1f3318"
VR_GREY         = "#7b8074"
VR_GREY_LIGHT   = "#b4b8af"

def lighten(c, amount=0.55):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

sns.set_theme(style="white", context="paper")
sns.set_palette(VR_PALETTE)
plt.rcParams.update({
    "figure.facecolor":  VR_PLOT_BG, "axes.facecolor": VR_PLOT_BG,
    "savefig.facecolor": VR_PLOT_BG, "text.color":     VR_INK,
    "axes.labelcolor":   VR_INK,     "axes.edgecolor": VR_INK,
    "axes.titlecolor":   VR_INK,     "xtick.color":    VR_INK,
    "ytick.color":       VR_INK,
    "axes.spines.top":   False,      "axes.spines.right": False,
    "hatch.linewidth":   2.2,
})

OUT = Path.home() / "Documents/jonathanbostock.github.io/vibe-research/matching-game-scale/figs"

# Each group: (label, colour, train, test, is_ablation)
groups = [
    ("Pythia-70M",     VR_PALETTE[0], 1.000, 1.000, False),
    ("Pythia-1.4B",    VR_PALETTE[1], 1.000, 1.000, False),
    ("Qwen-14B",       VR_PALETTE[2], 0.935, 0.787, False),
    ("Qwen-14B (abl.)", VR_PALETTE[2], 0.450, 0.440, True),
]

BAR_W = 0.7
INTRA_GAP = 0.18
INTER_GAP = 0.95
sub_labels = ["train", "test"]

fig, ax = plt.subplots(figsize=(8.8, 4.4))
x_cur = 0.6
group_centres = []
for label, colour, tr, te, abl in groups:
    g_start = x_cur
    for i, val in enumerate((tr, te)):
        x = x_cur + i * (BAR_W + INTRA_GAP)
        ax.bar(x, val, width=BAR_W, color=colour, edgecolor=VR_INK, lw=0.7, zorder=2)
        if abl:
            rect = Rectangle((x - BAR_W / 2, 0), BAR_W, val,
                             fill=False, edgecolor=lighten(colour),
                             hatch="///", lw=0, zorder=3)
            ax.add_patch(rect)
        ax.text(x, val + 0.018, f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, color=VR_INK)
        ax.text(x, -0.035, sub_labels[i], ha="center", va="top",
                fontsize=8.5, color=VR_INK)
    g_end = x_cur + (BAR_W + INTRA_GAP)
    centre = (g_start + g_end) / 2 - (BAR_W + INTRA_GAP - BAR_W) / 2
    # simpler centre: midpoint between the two bars
    centre = x_cur + (BAR_W + INTRA_GAP) / 2
    group_centres.append((centre, label, colour))
    x_cur = g_end + BAR_W + INTER_GAP

# Group labels + over-lines above bars
LABEL_Y = 1.135
LINE_Y  = 1.10
for cx, label, colour in group_centres:
    ax.text(cx, LABEL_Y, label, ha="center", va="bottom", fontsize=10.5,
            color=colour, fontweight="bold")
    half = (BAR_W + INTRA_GAP + BAR_W) / 2 + 0.03
    ax.plot([cx - half, cx + half], [LINE_Y, LINE_Y],
            color=colour, lw=2.2, solid_capstyle="butt")

# Chance line
ax.axhline(0.25, color=VR_INK, ls="--", lw=0.9, alpha=0.55, zorder=1)
ax.text(0.2, 0.265, "chance = 0.25", fontsize=8, color=VR_INK)

# Axis
ax.set_xticks([])
ax.set_xlim(0, x_cur)
ax.set_ylim(0, 1.28)
ax.set_ylabel("accuracy")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

legend_handles = [
    Patch(facecolor=VR_GREY, edgecolor=VR_INK, lw=0.7,
          label="recipe (λ and LR scaled to model size)"),
    Patch(facecolor=VR_GREY, edgecolor=VR_GREY_LIGHT, hatch="///",
          label="ablation: 1.4B λ and LR reused at 14B (underfits)"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "recipe_scales.pdf", bbox_inches="tight")
fig.savefig(OUT / "recipe_scales.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'recipe_scales.png'}")
