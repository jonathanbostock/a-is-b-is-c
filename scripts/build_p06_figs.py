"""Build figures for the p=0.6 full-rank-with-regularization sweep."""
from __future__ import annotations
import json, re
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
    "figure.facecolor":  VR_PLOT_BG, "axes.facecolor": VR_PLOT_BG,
    "savefig.facecolor": VR_PLOT_BG, "text.color":      VR_INK,
    "axes.labelcolor":   VR_INK, "axes.edgecolor": VR_INK,
    "axes.titlecolor":   VR_INK, "xtick.color":    VR_INK, "ytick.color": VR_INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

CHANCE = 0.25
OUT = Path.home() / "Documents/jonathanbostock.github.io/vibe-research/matching-game-scale/figs"

paths = sorted(Path("runs/sweep06_pulled").rglob("*_llm_*/eval_results.json"))
paths = [p for p in paths if "sweep_70m_p06" in str(p)]

rows = []
for p in paths:
    cfg = p.parent.name.split("_llm_")[0]
    d = json.load(open(p))
    last = [r for r in d if r["repeat_id"] == 0][-1]
    rows.append({
        "cfg": cfg,
        "train_acc": float(np.mean([e["accuracy"] for e in last["train_edges"]])),
        "test_acc":  float(np.mean([e["accuracy"] for e in last["test_edges"]])),
    })

# Bar plot ordered by test_acc descending
rows.sort(key=lambda r: -r["test_acc"])
labels = [r["cfg"] for r in rows]
te = [r["test_acc"] for r in rows]
tr = [r["train_acc"] for r in rows]

# Categorize for colour
def cat(cfg: str) -> int:
    if cfg.startswith("rank"):
        return 0  # LoRA controls
    if "mixin" in cfg:
        return 3  # mixin (purple — flagging the negative finding)
    if "l2sp" in cfg:
        return 2  # L2-SP (green)
    return 1      # plain full-rank LR sweep (orange)

cat_colors = [VR_PALETTE[cat(c)] for c in labels]
cat_names  = {0: "LoRA", 1: "full-rank LR sweep", 2: "full-rank + L2-SP", 3: "full-rank + on-policy mixin"}

fig, ax = plt.subplots(figsize=(8.6, 4.6))
xs = np.arange(len(rows))
ax.barh(xs, te, color=cat_colors, edgecolor=VR_PLOT_BG, lw=0.6)
ax.axvline(CHANCE, color=VR_INK, lw=0.9, ls="--", alpha=0.6)
ax.text(CHANCE + 0.01, len(rows) - 0.5, "chance = 0.25", color=VR_INK, fontsize=8, va="center")
for i, t in enumerate(te):
    ax.text(t + 0.01, i, f"{t:.2f}", va="center", fontsize=9, color=VR_INK)
ax.set_yticks(xs)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.set_xlabel("test-edge accuracy (mean across 6 held-out edges, all composable)")
ax.set_title("Pythia-70M, p_train = 0.6 · full-rank + regularization sweep", fontsize=11)
# Manual legend
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(facecolor=VR_PALETTE[i], label=cat_names[i]) for i in [0, 1, 2, 3]],
    fontsize=8, loc="lower right", frameon=True, framealpha=0.95,
)
fig.tight_layout()
fig.savefig(OUT / "sweep_p06.pdf", bbox_inches="tight")
fig.savefig(OUT / "sweep_p06.png", dpi=200, bbox_inches="tight")
print("wrote sweep_p06.{pdf,png}")
