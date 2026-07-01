"""Build figures for the V4 (overnight 1.4B sweep) section of the page."""
from __future__ import annotations
import json
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


def collect(folder: Path) -> list[dict]:
    rows = []
    for evf in folder.rglob("*_llm_*/eval_results.json"):
        cfg = evf.parent.name.split("_llm_")[0]
        d = json.load(open(evf))
        rep0 = sorted([r for r in d if r["repeat_id"] == 0], key=lambda r: r["step"])
        if not rep0: continue
        last = rep0[-1]
        train_edges = set(tuple(e["edge"]) for e in last["train_edges"])
        te_cmp = [e["accuracy"] for e in last["test_edges"] if composable(*e["edge"], train_edges)]
        te_nc  = [e["accuracy"] for e in last["test_edges"] if not composable(*e["edge"], train_edges)]
        rows.append(dict(
            cfg=cfg, step=last["step"],
            train_acc=float(np.mean([e["accuracy"] for e in last["train_edges"]])),
            test_acc=float(np.mean([e["accuracy"] for e in last["test_edges"]])),
            test_cmp=float(np.mean(te_cmp)) if te_cmp else float("nan"),
            test_nc=float(np.mean(te_nc)) if te_nc else float("nan"),
        ))
    return rows


def cfg_category(cfg: str) -> int:
    if "lora_r16" in cfg or cfg.startswith("lora") and "r16" in cfg:
        return 1  # low-rank LoRA — orange (warning)
    if cfg.startswith("lora"):
        return 0  # high-rank LoRA — blue
    if "l2sp" in cfg:
        return 2  # full-rank + L2-SP — green
    return 5      # plain full-rank — violet


def make_bar(rows, title, fname, drop_chance_label=False):
    rows = sorted(rows, key=lambda r: -r["test_acc"])
    labels = [r["cfg"] for r in rows]
    te = [r["test_acc"] for r in rows]
    colors = [VR_PALETTE[cfg_category(c)] for c in labels]
    fig, ax = plt.subplots(figsize=(8.6, max(4, 0.32 * len(rows) + 1.5)))
    xs = np.arange(len(rows))
    ax.barh(xs, te, color=colors, edgecolor=VR_PLOT_BG, lw=0.6)
    ax.axvline(CHANCE, color=VR_INK, lw=0.9, ls="--", alpha=0.6)
    if not drop_chance_label:
        ax.text(CHANCE + 0.01, len(rows) - 0.5, "chance = 0.25",
                color=VR_INK, fontsize=8, va="center")
    for i, t in enumerate(te):
        ax.text(t + 0.01, i, f"{t:.2f}", va="center", fontsize=9, color=VR_INK)
    ax.set_yticks(xs); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_xlabel("test-edge accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[
        Patch(facecolor=VR_PALETTE[0], label="LoRA (rank ≥ 64)"),
        Patch(facecolor=VR_PALETTE[1], label="LoRA r=16"),
        Patch(facecolor=VR_PALETTE[2], label="full-rank + L2-SP"),
        Patch(facecolor=VR_PALETTE[5], label="plain full-rank"),
    ], fontsize=8, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


p06 = collect(Path("runs/p14b_pulled/a100/runs/sweep_p14b_p06")) + collect(Path("runs/p14b_pulled/a100-b/runs/sweep_p14b_p06"))
p04 = collect(Path("runs/p14b_pulled/a100/runs/sweep_p14b_p04")) + collect(Path("runs/p14b_pulled/a100-b/runs/sweep_p14b_p04"))

make_bar(p06, "Pythia-1.4B, p_train = 0.6 · overnight sweep", "sweep_p14b_p06_v4")
make_bar(p04, "Pythia-1.4B, p_train = 0.4 · L2-SP × LR grid", "sweep_p14b_p04_v4")

# Composable vs non-composable breakdown for p=0.4
rows = sorted(p04, key=lambda r: -r["test_acc"])[:10]   # top 10
labels = [r["cfg"] for r in rows]
cmp = [r["test_cmp"] for r in rows]
nc  = [r["test_nc"]  for r in rows]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
xs = np.arange(len(rows))
width = 0.4
ax.bar(xs - width/2, cmp, width, label="composable edges", color=VR_PALETTE[2])
ax.bar(xs + width/2, nc,  width, label="non-composable edges", color=VR_PALETTE[1])
ax.axhline(CHANCE, color=VR_INK, lw=0.9, ls="--", alpha=0.6)
ax.text(len(rows)-0.5, CHANCE - 0.04, "chance = 0.25", color=VR_INK, fontsize=8, ha="right")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.08)
ax.set_title("Pythia-1.4B at p=0.4 — composable vs non-composable test edges\n(top 10 configs by mean test acc)", fontsize=11)
ax.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "sweep_p14b_p04_breakdown.pdf", bbox_inches="tight")
fig.savefig(OUT / "sweep_p14b_p04_breakdown.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote sweep_p14b_p04_breakdown")
