"""Per-edge comparison: full-rank vs best LoRA configs vs original 70M templated."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

VR_PALETTE = ["#2f6175", "#d96d3f", "#1a6b06", "#c70053", "#c3a322", "#865ecf"]
VR_PLOT_BG = "#fafcf2"
VR_INK     = "#1f3318"
sns.set_theme(style="white", context="paper")
sns.set_palette(VR_PALETTE)
plt.rcParams.update({
    "figure.facecolor":  VR_PLOT_BG,
    "axes.facecolor":    VR_PLOT_BG,
    "savefig.facecolor": VR_PLOT_BG,
    "text.color":        VR_INK,
    "axes.labelcolor":   VR_INK,
    "axes.edgecolor":    VR_INK,
    "axes.titlecolor":   VR_INK,
    "xtick.color":       VR_INK,
    "ytick.color":       VR_INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = Path.home() / "Documents/jonathanbostock.github.io/vibe-research/matching-game-scale/figs"

RUNS = [
    ("70M templated (orig, r=16, α=16)", "runs/from-pods/a100/runs/random_6_0.4_0.2_llm_20260629_142908"),
    ("full-rank, lr=2e-4",                "runs/sweep_pulled/a100-b/runs/random_6_0.4_0.2_llm_20260629_181201"),
    ("LoRA r=256, α=512",                 "runs/sweep_pulled/a100-b/runs/random_6_0.4_0.2_llm_20260629_175419"),
    ("LoRA r=16, α=32, dp=0.3",           "runs/sweep_pulled/a100-b/runs/random_6_0.4_0.2_llm_20260629_184151"),
    ("LoRA r=128, α=256",                 "runs/sweep_pulled/a100/runs/random_6_0.4_0.2_llm_20260629_193456"),
]

train_edges = {(1,0),(1,2),(1,5),(2,0),(2,1),(2,3),(2,5),(3,0),(3,5),(4,2),(5,0),(5,1)}
def composable(s, t):
    return any((s, m) in train_edges and (m, t) in train_edges for m in range(6))

CHANCE = 0.25
edge_results = {}
for name, p in RUNS:
    d = json.load(open(Path(p) / "eval_results.json"))
    last = [r for r in d if r["repeat_id"] == 0][-1]
    for e in last["test_edges"]:
        edge_results.setdefault(tuple(e["edge"]), {})[name] = e["accuracy"]

# Sort edges: composable first, then non-composable
edges = sorted(edge_results, key=lambda e: (not composable(*e), e))
xs = np.arange(len(edges))
labels = [f"{s}→{t}" for s, t in edges]
can_compose = [composable(*e) for e in edges]

fig, ax = plt.subplots(figsize=(8.6, 4.0))
colors = sns.color_palette(VR_PALETTE, n_colors=len(RUNS))
width = 0.16
for i, ((name, _), c) in enumerate(zip(RUNS, colors)):
    vals = [edge_results[e].get(name, float("nan")) for e in edges]
    ax.bar(xs + (i - (len(RUNS)-1)/2) * width, vals, width, label=name, color=c, edgecolor="white", lw=0.4)

ax.axhline(CHANCE, color="k", lw=0.8, ls="--", alpha=0.7, label="chance = 0.25")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=10)
for tick, can in zip(ax.get_xticklabels(), can_compose):
    tick.set_color(VR_PALETTE[2] if can else VR_PALETTE[1])
    tick.set_fontweight("bold" if can else "normal")
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy (final step)")
ax.set_xlabel("held-out test edge (green = 2-hop composable from train edges)")
ax.set_title("Per-edge generalization · best p=0.4 sweep configs", fontsize=11)
ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "per_edge_sweep.pdf", bbox_inches="tight")
fig.savefig(OUT / "per_edge_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Also build a focused "non-composable edges only" comparison to make the
# full-rank-wins-here point land
fig, ax = plt.subplots(figsize=(6.2, 3.6))
non_edges = [e for e in edges if not composable(*e)]
xs2 = np.arange(len(non_edges))
labels2 = [f"{s}→{t}" for s, t in non_edges]
for i, ((name, _), c) in enumerate(zip(RUNS, colors)):
    vals = [edge_results[e].get(name, float("nan")) for e in non_edges]
    ax.bar(xs2 + (i - (len(RUNS)-1)/2) * width, vals, width, label=name, color=c, edgecolor="white", lw=0.4)
ax.axhline(CHANCE, color="k", lw=0.8, ls="--", alpha=0.7, label="chance = 0.25")
ax.set_xticks(xs2)
ax.set_xticklabels(labels2)
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy (final step)")
ax.set_title("Non-composable test edges only · full-rank advantage", fontsize=11)
ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "per_edge_noncompose.pdf", bbox_inches="tight")
fig.savefig(OUT / "per_edge_noncompose.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Wrote per_edge_sweep + per_edge_noncompose")
