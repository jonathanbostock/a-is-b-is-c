"""Linear probe trained on saved residual activations.

For each training step, pools all saved residuals (across all repeat_ids
and all edges), applies a random train/test split, fits a linear
least-squares classifier that predicts group index from the residual
vector, and returns per-step train/test accuracy.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from .utils import style_metric_axis


def compute_probe_accuracies(
    residuals_file: Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """Return (steps, train_means, train_stds, test_means, test_stds).

    Fits a separate probe per repeat_id, then averages across repeats so that
    std across repeats can be used as an error estimate.  The probe train/test
    split (edge-split labels are ignored; split is purely random) is fixed per
    repeat using seed + repeat_id.
    """
    all_entries: list[dict] = json.loads(residuals_file.read_text(encoding="utf-8"))
    if not all_entries:
        return [], [], [], [], []

    # Infer k from max group label seen
    k = 1 + max(
        int(r["group"])
        for e in all_entries
        for split_key in ("train_residuals", "test_residuals")
        for r in e.get(split_key, [])
    )

    all_steps = sorted(set(int(e["step"]) for e in all_entries))

    # Group entries by repeat_id
    by_repeat: dict[int, list[dict]] = defaultdict(list)
    for entry in all_entries:
        by_repeat[entry.get("repeat_id", 0)].append(entry)

    repeat_train: dict[int, dict[int, float]] = {}
    repeat_test: dict[int, dict[int, float]] = {}

    for repeat_id, entries in by_repeat.items():
        # Build per-step (residual, group) pairs for this repeat
        by_step: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
        for entry in entries:
            step = int(entry["step"])
            for split_key in ("train_residuals", "test_residuals"):
                for r in entry.get(split_key, []):
                    vec = np.array(r["residual"], dtype=np.float32)
                    by_step[step].append((vec, int(r["group"])))

        steps_here = sorted(by_step.keys())
        if not steps_here:
            continue

        # Fixed train/test split for this repeat
        n = len(by_step[steps_here[0]])
        rng = np.random.default_rng(seed + repeat_id)
        perm = rng.permutation(n)
        n_test = max(1, int(n * test_fraction))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        repeat_train[repeat_id] = {}
        repeat_test[repeat_id] = {}

        for step in all_steps:
            if step not in by_step:
                continue
            items = by_step[step]
            if len(items) != n:
                rng2 = np.random.default_rng(seed + repeat_id)
                perm2 = rng2.permutation(len(items))
                n_test2 = max(1, int(len(items) * test_fraction))
                test_idx = perm2[:n_test2]
                train_idx = perm2[n_test2:]

            X = np.stack([v for v, _ in items], axis=0)
            y = np.array([g for _, g in items], dtype=np.int32)

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            Y_tr = np.zeros((len(y_tr), k), dtype=np.float32)
            Y_tr[np.arange(len(y_tr)), y_tr] = 1.0

            W, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)

            repeat_train[repeat_id][step] = float(np.mean(np.argmax(X_tr @ W, axis=1) == y_tr))
            repeat_test[repeat_id][step] = float(np.mean(np.argmax(X_te @ W, axis=1) == y_te))

    # Aggregate across repeats
    train_means, train_stds, test_means, test_stds = [], [], [], []
    valid_steps = []
    for step in all_steps:
        tr_vals = [repeat_train[r][step] for r in repeat_train if step in repeat_train[r]]
        te_vals = [repeat_test[r][step] for r in repeat_test if step in repeat_test[r]]
        if not tr_vals:
            continue
        valid_steps.append(step)
        train_means.append(float(np.mean(tr_vals)))
        train_stds.append(float(np.std(tr_vals, ddof=1) if len(tr_vals) > 1 else 0.0))
        test_means.append(float(np.mean(te_vals)))
        test_stds.append(float(np.std(te_vals, ddof=1) if len(te_vals) > 1 else 0.0))

    return valid_steps, train_means, train_stds, test_means, test_stds


def plot_linear_probe(
    *,
    residuals_file: Path,
    axis: Axes,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> None:
    """Fit and plot linear probe accuracy on `axis`, with ±1 std error bands."""
    steps, train_means, train_stds, test_means, test_stds = compute_probe_accuracies(
        residuals_file, seed=seed, test_fraction=test_fraction
    )
    if not steps:
        return

    # Infer k for the chance baseline
    all_entries: list[dict] = json.loads(residuals_file.read_text(encoding="utf-8"))
    k = 1 + max(
        int(r["group"])
        for entry in all_entries
        for key in ("train_residuals", "test_residuals")
        for r in entry.get(key, [])
    )

    x = np.array(steps, dtype=float)
    train_means_arr = np.array(train_means)
    train_stds_arr = np.array(train_stds)
    test_means_arr = np.array(test_means)
    test_stds_arr = np.array(test_stds)

    axis.plot(x, train_means_arr, color="#555555", linewidth=2.0, label="Probe train")
    axis.fill_between(x, train_means_arr - train_stds_arr, train_means_arr + train_stds_arr,
                      color="#555555", alpha=0.15)
    axis.plot(x, test_means_arr, color="#555555", linewidth=2.0, linestyle="--", label="Probe test")
    axis.fill_between(x, test_means_arr - test_stds_arr, test_means_arr + test_stds_arr,
                      color="#555555", alpha=0.15)
    axis.axhline(1.0 / k, linestyle=":", color="black", linewidth=1, label=f"Chance (1/{k})")
    axis.axhline(1.0,      linestyle=":", color="black", linewidth=0.75, alpha=0.4)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Linear probe: group classification accuracy")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Accuracy")
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlim(min(steps), max(steps))
    axis.margins(x=0)
    axis.legend(fontsize=8)
    style_metric_axis(axis=axis)
