"""Make a one-shot comparison plot across all the multi-scale runs."""
from __future__ import annotations

import json
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="paper")

RUNS = [
    ("Pythia-70M  (templated)", "runs/from-pods/a100/runs/random_6_0.4_0.2_llm_20260629_142908", "C0"),
    ("Pythia-70M  (synth-doc)", "runs/from-pods/a100/runs/random_6_0.4_0.2_llm_20260629_151123", "C2"),
    ("Pythia-1.4B (templated)", "runs/from-pods/a100-b/runs/random_6_0.4_0.2_llm_20260629_144611", "C1"),
    ("Qwen2.5-32B (templated)", "runs/from-pods/h100/runs/random_6_0.4_0.2_llm_20260629_152126", "C3"),
]


def _trajectory(path: str) -> dict[str, list[float]]:
    d = json.load(open(Path(path) / "eval_results.json"))
    rep0 = [r for r in d if r["repeat_id"] == 0]
    rep0.sort(key=lambda r: r["step"])
    return {
        "step": [r["step"] for r in rep0],
        "train_mean": [sum(e["log_odds_gap"] for e in r["train_edges"])/len(r["train_edges"]) for r in rep0],
        "test_mean":  [sum(e["log_odds_gap"] for e in r["test_edges"]) /len(r["test_edges"])  for r in rep0],
        "test_max":   [max(e["log_odds_gap"] for e in r["test_edges"])                       for r in rep0],
        "test_acc":   [sum(e["accuracy"] for e in r["test_edges"])    /len(r["test_edges"])  for r in rep0],
    }


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for label, path, color in RUNS:
    t = _trajectory(path)
    axes[0].plot(t["step"], t["train_mean"], "-", color=color, label=label, lw=1.8)
    axes[1].plot(t["step"], t["test_mean"],  "-", color=color, label=label, lw=1.8)
    axes[1].plot(t["step"], t["test_max"],   ":", color=color, lw=1.2, alpha=0.6)
    axes[2].plot(t["step"], t["test_acc"],   "-", color=color, label=label, lw=1.8)

for ax in axes:
    ax.set_xscale("symlog", linthresh=10)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Training step (symlog)")
axes[0].set_title("TRAIN-edge mean log-odds gap")
axes[0].set_ylabel("log-odds gap")
axes[1].set_title("TEST-edge log-odds gap\n(solid = mean, dotted = max over test edges)")
axes[1].set_ylabel("log-odds gap")
axes[1].axhline(0, color="k", lw=0.5, ls="--")
axes[2].set_title("TEST-edge accuracy (best-of-8 guess)")
axes[2].set_ylabel("accuracy")
axes[2].axhline(1/8, color="k", lw=0.5, ls=":")
axes[0].legend(loc="lower right", frameon=True, fontsize=8)
fig.suptitle("Concept crystallization across model scale\n(random_6_0.4_0.2 topology, n_repeats=1, single seed)", fontsize=12)
fig.tight_layout()
out_pdf = Path("runs/scale_comparison.pdf")
out_pdf.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"saved {out_pdf}")
