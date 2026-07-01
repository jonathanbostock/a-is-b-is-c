"""Read all sweep_70m run dirs (pulled into runs/from-pods-sweep/...) and emit:
- a summary CSV
- a rank-vs-test-gap line plot
- a regularization heatmap
- an alpha/r line plot
- a per-edge bar plot showing best rank vs full-rank
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

VR_PALETTE = ["#2f6175", "#d96d3f", "#1a6b06", "#c70053", "#c3a322", "#865ecf"]
VR_PLOT_BG = "#fafcf2"
VR_INK     = "#1f3318"
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
mpl.colormaps.register(LinearSegmentedColormap.from_list(
    "vr_div", ["#1f627a", "#ffffff", "#803619"]), name="vr_div", force=True)
mpl.colormaps.register(LinearSegmentedColormap.from_list(
    "vr_seq", ["#04222e", "#1a5547", "#4a8541", "#a8c863", "#faf3d0"]), name="vr_seq", force=True)
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
RUN_ROOT = Path("runs/sweep_pulled")
MASTER_LOGS = list(Path("/tmp").glob("master-*.log"))


def parse_master_log(log_path: Path) -> list[tuple[str, str, str]]:
    """Returns list of (config_name, start_time_HHMMSS, done_time_HHMMSS)."""
    rows = []
    cur_cfg, cur_start = None, None
    for line in log_path.read_text().splitlines():
        m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\] (starting|done) (\S+)", line)
        if not m: continue
        hms = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        evt = m.group(4); cfg = m.group(5)
        if evt == "starting":
            cur_cfg, cur_start = cfg, hms
        elif evt == "done" and cur_cfg == cfg and cur_start:
            rows.append((cfg, cur_start, hms))
            cur_cfg, cur_start = None, None
    return rows


def find_runs() -> dict[str, Path]:
    """Map config_name → run_dir using master logs to match by start time."""
    cfg_to_path: dict[str, Path] = {}
    for log in MASTER_LOGS:
        pod = log.stem.replace("master-", "")
        runs_root = RUN_ROOT / pod / "runs"
        if not runs_root.exists(): continue
        # Build list of (timestamp_HHMMSS, run_dir)
        run_index = []
        for d in runs_root.iterdir():
            m = re.search(r"_llm_\d{8}_(\d{6})$", d.name)
            if m and (d / "eval_results.json").exists():
                run_index.append((m.group(1), d))
        run_index.sort()
        for cfg, start, _ in parse_master_log(log):
            # Pick the run whose timestamp is >= start by smallest margin
            best = None
            for ts, d in run_index:
                if ts >= start:
                    best = d; break
            if best is None and run_index:
                best = run_index[-1][1]
            if best is not None:
                cfg_to_path[cfg] = best
    return cfg_to_path


CHANCE = 0.25  # 1 / (1 + n_negatives), n_negatives = 3


def summarize(cfg_to_path: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for cfg, path in cfg_to_path.items():
        d = json.load(open(path / "eval_results.json"))
        rep0 = sorted([r for r in d if r["repeat_id"] == 0], key=lambda r: r["step"])
        last = rep0[-1]
        train_edges = set(tuple(e["edge"]) for e in last["train_edges"])
        def composable(s, t):
            return any((s, m) in train_edges and (m, t) in train_edges for m in range(6))
        te_comp_acc = [e["accuracy"] for e in last["test_edges"] if composable(*e["edge"])]
        te_noncomp_acc = [e["accuracy"] for e in last["test_edges"] if not composable(*e["edge"])]
        rows.append({
            "cfg": cfg,
            "step": last["step"],
            "train_acc":  np.mean([e["accuracy"] for e in last["train_edges"]]),
            "test_acc":   np.mean([e["accuracy"] for e in last["test_edges"]]),
            "test_compose_acc":    np.mean(te_comp_acc) if te_comp_acc else np.nan,
            "test_noncompose_acc": np.mean(te_noncomp_acc) if te_noncomp_acc else np.nan,
            "path": str(path),
        })
    return pd.DataFrame(rows)


def main():
    cfg_to_path = find_runs()
    print(f"Found {len(cfg_to_path)} runs.")
    if not cfg_to_path:
        return

    df = summarize(cfg_to_path)
    df.to_csv(OUT / "sweep_summary.csv", index=False)
    print(df.to_string(index=False))

    # ── Rank sweep plot ──────────────────────────────────────────────────────
    ranks = []
    for _, row in df.iterrows():
        m = re.fullmatch(r"rank(\d+)", row["cfg"])
        if m:
            ranks.append((int(m.group(1)), row))
    fullrank_row = df[df["cfg"] == "fullrank"]
    if ranks:
        ranks.sort()
        rs = [r for r, _ in ranks]
        c_acc = [row["test_compose_acc"]   for _, row in ranks]
        n_acc = [row["test_noncompose_acc"] for _, row in ranks]
        tr_acc = [row["train_acc"] for _, row in ranks]

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(rs, c_acc, "o-", label="TEST · composable edges (mean acc)", lw=2)
        ax.plot(rs, n_acc, "s--", label="TEST · non-composable edges (mean acc)", lw=1.5)
        ax.plot(rs, tr_acc, "^:", label="TRAIN edges (mean acc)", lw=1.0, alpha=0.7, color="gray")
        if len(fullrank_row):
            fr = fullrank_row.iloc[0]
            ax.axhline(fr["test_compose_acc"], color="C2", ls="-", lw=1.0, alpha=0.6,
                       label=f"full-rank composable = {fr['test_compose_acc']:.2f}")
            ax.axhline(fr["test_noncompose_acc"], color="C3", ls="--", lw=1.0, alpha=0.6,
                       label=f"full-rank non-composable = {fr['test_noncompose_acc']:.2f}")
        ax.axhline(CHANCE, color="k", ls="--", lw=0.8, alpha=0.7, label=f"chance = {CHANCE:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rs); ax.set_xticklabels([str(r) for r in rs])
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel("accuracy")
        ax.set_title("Pythia-70M · concept crystallization vs. LoRA rank")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "sweep_rank.pdf", bbox_inches="tight")
        fig.savefig(OUT / "sweep_rank.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote sweep_rank.{pdf,png}")

    # ── Regularization heatmap ──────────────────────────────────────────────
    reg_rows = []
    for _, row in df.iterrows():
        m = re.fullmatch(r"reg_wd(\d+p\d+)_dp(\d+p\d+)", row["cfg"])
        if m:
            wd = float(m.group(1).replace("p", "."))
            dp = float(m.group(2).replace("p", "."))
            reg_rows.append((wd, dp, row["test_compose_acc"], row["test_noncompose_acc"]))
    base_row = df[df["cfg"] == "rank016"]
    if not base_row.empty:
        br = base_row.iloc[0]
        reg_rows.append((0.0, 0.0, br["test_compose_acc"], br["test_noncompose_acc"]))
    if reg_rows:
        wds = sorted(set(r[0] for r in reg_rows))
        dps = sorted(set(r[1] for r in reg_rows))
        composable_grid = np.full((len(wds), len(dps)), np.nan)
        noncomp_grid    = np.full((len(wds), len(dps)), np.nan)
        for wd, dp, c, n in reg_rows:
            i = wds.index(wd); j = dps.index(dp)
            composable_grid[i, j] = c
            noncomp_grid[i, j]    = n

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        for ax, grid, title, vmin, vmax in [
            (axes[0], composable_grid, "test composable edges (mean acc)", 0.0, 1.0),
            (axes[1], noncomp_grid,   "test non-composable edges (mean acc; chance=0.25)", 0.0, 1.0),
        ]:
            im = ax.imshow(grid, cmap="vr_seq", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(dps))); ax.set_xticklabels([f"{d:g}" for d in dps])
            ax.set_yticks(range(len(wds))); ax.set_yticklabels([f"{w:g}" for w in wds])
            ax.set_xlabel("lora_dropout"); ax.set_ylabel("weight_decay")
            ax.set_title(title)
            for i in range(len(wds)):
                for j in range(len(dps)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                                color="white" if grid[i, j] < 0.5 else "black",
                                fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("Pythia-70M · regularization sweep at r=16, alpha=32 (chance acc = 0.25)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "sweep_reg.pdf", bbox_inches="tight")
        fig.savefig(OUT / "sweep_reg.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote sweep_reg.{pdf,png}")

    # ── Alpha/r line plot ───────────────────────────────────────────────────
    alpha_rows = []
    for _, row in df.iterrows():
        m = re.fullmatch(r"alpha(\d+)_r16", row["cfg"])
        if m:
            alpha_rows.append((int(m.group(1)), row))
    if alpha_rows:
        alpha_rows.sort()
        alphas = [a for a, _ in alpha_rows]
        c_acc = [row["test_compose_acc"]    for _, row in alpha_rows]
        n_acc = [row["test_noncompose_acc"] for _, row in alpha_rows]
        tr_acc = [row["train_acc"] for _, row in alpha_rows]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(alphas, c_acc, "o-", label="TEST composable")
        ax.plot(alphas, n_acc, "s--", label="TEST non-composable")
        ax.plot(alphas, tr_acc, "^:", label="TRAIN", alpha=0.6, color="gray")
        ax.axhline(CHANCE, color="k", lw=0.8, ls="--", alpha=0.7, label=f"chance = {CHANCE:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log", base=2)
        ax.set_xticks(alphas); ax.set_xticklabels([str(a) for a in alphas])
        ax.set_xlabel("lora_alpha (at r=16)")
        ax.set_ylabel("accuracy")
        ax.set_title("Pythia-70M · alpha sweep at r=16")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / "sweep_alpha.pdf", bbox_inches="tight")
        fig.savefig(OUT / "sweep_alpha.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote sweep_alpha.{pdf,png}")


if __name__ == "__main__":
    main()
