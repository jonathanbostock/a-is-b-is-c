"""Build the figures for the vibe-research piece."""
from __future__ import annotations
import json
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
OUT.mkdir(exist_ok=True, parents=True)

RUNS = [
    ("Pythia-70M (templated)",       "runs/from-pods/a100/runs/random_6_0.4_0.2_llm_20260629_142908", "70M-tmpl"),
    ("Pythia-70M (synth-doc)",       "runs/from-pods/a100/runs/random_6_0.4_0.2_llm_20260629_151123", "70M-doc"),
    ("Pythia-1.4B  (LR=1e-4)",       "runs/from-pods/a100-b/runs/random_6_0.4_0.2_llm_20260629_144611", "1.4B-lo"),
    ("Pythia-1.4B  (LR=4e-4)",       "runs/from-pods/a100-b/runs/random_6_0.4_0.2_llm_20260629_155726", "1.4B-hi"),
    ("Qwen2.5-32B (LR=5e-5, r=32)",  "runs/from-pods/h100/runs/random_6_0.4_0.2_llm_20260629_152126", "32B-lo"),
    ("Qwen2.5-32B (LR=2e-4, r=64)",  "runs/from-pods/h100/runs/random_6_0.4_0.2_llm_20260629_155725", "32B-hi"),
]


CHANCE = 0.25  # accuracy is lp(correct) > max(lp(3 in-category negatives)) → 1/4


def _trajectory(path: str) -> dict:
    d = json.load(open(Path(path) / "eval_results.json"))
    rep0 = sorted([r for r in d if r["repeat_id"] == 0], key=lambda r: r["step"])
    return {
        "step":           [r["step"] for r in rep0],
        "train_acc":      [np.mean([e["accuracy"] for e in r["train_edges"]]) for r in rep0],
        "test_acc":       [np.mean([e["accuracy"] for e in r["test_edges"]])  for r in rep0],
        "test_acc_max":   [max([e["accuracy"] for e in r["test_edges"]])      for r in rep0],
        "final":          rep0[-1],
    }

trajs = {short: _trajectory(p) for _, p, short in RUNS}

# ── Figure 1: trajectory side-by-side (accuracy) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
colors = sns.color_palette(VR_PALETTE, n_colors=len(RUNS))
for (label, _, short), c in zip(RUNS, colors):
    t = trajs[short]
    axes[0].plot(t["step"], t["train_acc"], "-", color=c, label=label, lw=1.8)
    axes[1].plot(t["step"], t["test_acc"],  "-", color=c, label=label, lw=1.8)
    axes[1].plot(t["step"], t["test_acc_max"], ":", color=c, lw=1.0, alpha=0.6)

for ax in axes:
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("training step")
    ax.axhline(CHANCE, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("accuracy   (chance = 0.25)")
axes[0].set_title("training edges")
axes[1].set_title("held-out test edges\n(solid = mean over test edges, dotted = max)")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "trajectory.pdf", bbox_inches="tight")
fig.savefig(OUT / "trajectory.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: per-edge bar at end of training ────────────────────────────────
# Take the actual sampled topology from 70M templated and read all final gaps.
ref = trajs["70M-tmpl"]["final"]
train_edges = sorted([tuple(e["edge"]) for e in ref["train_edges"]])
test_edges  = sorted([tuple(e["edge"]) for e in ref["test_edges"]])

# Test edges grouped by composition: which can be made from train edges in 2 hops?
train_set = set(train_edges)
def composable(src, tgt):
    if src == tgt: return False
    for mid in range(6):
        if (src, mid) in train_set and (mid, tgt) in train_set:
            return True
    return False
test_labels = []
test_can_compose = []
for s,t in test_edges:
    test_labels.append(f"{s}→{t}")
    test_can_compose.append(composable(s, t))

n_runs = len(RUNS)
fig, ax = plt.subplots(figsize=(8.4, 4.0))
width = 0.13
x = np.arange(len(test_edges))
for i, ((label, _, short), c) in enumerate(zip(RUNS, colors)):
    final = trajs[short]["final"]
    accs = {tuple(e["edge"]): e["accuracy"] for e in final["test_edges"]}
    vals = [accs[e] for e in test_edges]
    ax.bar(x + (i - (n_runs-1)/2)*width, vals, width, label=label, color=c, edgecolor="white", lw=0.4)

ax.axhline(CHANCE, color="k", lw=0.8, ls="--", alpha=0.7, label="chance (= 0.25)")
ax.set_xticks(x)
ax.set_xticklabels(test_labels, fontsize=10)
for tick, can in zip(ax.get_xticklabels(), test_can_compose):
    tick.set_color(VR_PALETTE[2] if can else VR_PALETTE[1])
    tick.set_fontweight("bold" if can else "normal")
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy (final step)")
ax.set_xlabel("held-out test edge (green = 2-hop composable from train edges)")
ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "per_edge_bar.pdf", bbox_inches="tight")
fig.savefig(OUT / "per_edge_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: topology diagram in two panels ─────────────────────────────────
from matplotlib.patches import FancyArrowPatch

n = 6
angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n+1)[:-1]
pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

fin70 = trajs["70M-tmpl"]["final"]
test_acc_by_edge = {tuple(e["edge"]): e["accuracy"] for e in fin70["test_edges"]}

def draw_node(ax, i):
    ax.scatter(*pos[i], s=720, c="white", edgecolor="black", zorder=5, lw=1.4)
    ax.text(*pos[i], str(i), ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=6)

def draw_arrow(ax, s, t, color, lw=1.6, alpha=0.85, rad=0.15):
    sp, tp = np.array(pos[s]), np.array(pos[t])
    # shrink endpoints so arrow doesn't sit inside the node circle
    arrow = FancyArrowPatch(sp, tp, arrowstyle="-|>", mutation_scale=12,
                            color=color, lw=lw, alpha=alpha,
                            connectionstyle=f"arc3,rad={rad}",
                            shrinkA=12, shrinkB=14, zorder=3)
    ax.add_patch(arrow)

def label_edge(ax, s, t, text, color, rad=0.15, push=0.0):
    sp, tp = np.array(pos[s]), np.array(pos[t])
    mid = (sp + tp) / 2
    v = tp - sp
    perp = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
    ax.text(*(mid + perp * (0.13 + push)), text, fontsize=8.5, color=color,
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
for ax in axes:
    for i in range(n): draw_node(ax, i)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")

# Left panel: training graph only
for e in train_edges:
    draw_arrow(axes[0], e[0], e[1], color=VR_PALETTE[2], lw=1.8, alpha=0.9)
axes[0].set_title("training edges (12 of 30)", fontsize=11)

# Right panel: test edges with 70M accuracy labels
for e in test_edges:
    can = composable(*e)
    color = VR_PALETTE[2] if can else VR_PALETTE[1]
    a = test_acc_by_edge[e]
    draw_arrow(axes[1], e[0], e[1], color=color, lw=2.2, alpha=0.95, rad=0.18)
    label_edge(axes[1], e[0], e[1], f"{a:.2f}", color)
axes[1].set_title("test edges · Pythia-70M accuracy (chance = 0.25)\ngreen = 2-hop composable from train edges",
                  fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "topology.pdf", bbox_inches="tight")
fig.savefig(OUT / "topology.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figures written to", OUT)
print("Files:", sorted(p.name for p in OUT.iterdir()))
