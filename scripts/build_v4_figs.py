"""Build figures for the V4 "scaling up — overnight 1.4B sweep" section.

Reads pulled 1.4B runs from runs/p14b_pulled/<pod>/runs/sweep_p14b_p06/... and
emits a single horizontal bar chart of final test-edge accuracy with chance
line, grouped by config family.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

def composable(s, t, train_edges):
    return any((s, m) in train_edges and (m, t) in train_edges for m in range(6))

def cfg_to_category(cfg: str) -> int:
    if cfg.startswith("fullrank_l2sp"):
        return 2  # L2-SP green
    if cfg.startswith("fullrank"):
        return 1  # LR sweep orange
    if cfg.startswith("lora"):
        return 0  # LoRA tooltip blue
    return 4

def main():
    rows = []
    for evf in Path("runs/p14b_pulled").rglob("*_llm_*/eval_results.json"):
        cfg = evf.parent.name.split("_llm_")[0]
        d = json.load(open(evf))
        rep0 = sorted([r for r in d if r["repeat_id"]==0], key=lambda r: r["step"])
        if not rep0: continue
        last = rep0[-1]
        train_edges = set(tuple(e["edge"]) for e in last["train_edges"])
        rows.append(dict(
            cfg=cfg,
            train_acc=float(np.mean([e["accuracy"] for e in last["train_edges"]])),
            test_acc=float(np.mean([e["accuracy"] for e in last["test_edges"]])),
        ))
    rows.sort(key=lambda r: -r["test_acc"])
    labels = [r["cfg"] for r in rows]
    te = [r["test_acc"] for r in rows]
    colors = [VR_PALETTE[cfg_to_category(c)] for c in labels]
    cat_name = {0: "LoRA", 1: "full-rank LR sweep", 2: "full-rank + L2-SP"}

    fig, ax = plt.subplots(figsize=(8.6, max(4, 0.35 * len(rows) + 1.5)))
    xs = np.arange(len(rows))
    ax.barh(xs, te, color=colors, edgecolor=VR_PLOT_BG, lw=0.6)
    ax.axvline(CHANCE, color=VR_INK, lw=0.9, ls="--", alpha=0.6)
    ax.text(CHANCE + 0.01, len(rows) - 0.5, "chance = 0.25",
            color=VR_INK, fontsize=8, va="center")
    for i, t in enumerate(te):
        ax.text(t + 0.01, i, f"{t:.2f}", va="center", fontsize=9, color=VR_INK)
    ax.set_yticks(xs); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_xlabel("test-edge accuracy")
    ax.set_title("Pythia-1.4B, p_train = 0.6 · overnight knob sweep", fontsize=11)
    ax.legend(handles=[Patch(facecolor=VR_PALETTE[i], label=cat_name[i]) for i in [0,1,2]],
              fontsize=8, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_p14b_p06.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sweep_p14b_p06.png", dpi=200, bbox_inches="tight")
    print(f"wrote sweep_p14b_p06.{{pdf,png}} from {len(rows)} runs")

if __name__ == "__main__":
    main()
