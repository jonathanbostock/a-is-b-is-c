"""PCA trail plots for residual stream analysis.

Reads pca_residuals.json produced during training, fits a 2D PCA on the
final eval step's residuals, then projects all steps' residuals onto those
axes and plots "trails" showing how representations evolve during training.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .utils import configure_seaborn_plot_style, get_palette, style_metric_axis


def _fit_pca_2d(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit 2-component PCA via eigendecomposition of the covariance matrix.

    Returns:
        components: (2, d_model) — the two principal directions
        mean: (d_model,) — the data mean
        explained_variance_ratio: (2,) — fraction of variance explained
    """
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = (X_c.T @ X_c) / max(X.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :2].T  # (2, d_model)
    total_var = float(eigenvalues.sum())
    evr = eigenvalues[:2] / total_var if total_var > 0 else np.zeros(2)
    return components, mean, evr


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _project(v: np.ndarray, components: np.ndarray, mean: np.ndarray) -> tuple[float, float]:
    proj = components @ (v - mean)
    return float(proj[0]), float(proj[1])


def _build_trails(
    entries: list[dict[str, Any]],
) -> tuple[
    dict[tuple[tuple[int, int], int], list[np.ndarray | None]],
    dict[tuple[tuple[int, int], int], list[np.ndarray | None]],
    list[dict[int, list[np.ndarray]]],
    list[int],
]:
    """Parse entries into per-(edge,group) residual sequences and group averages.

    Returns:
        train_trails: {(edge, group): [residual_or_None per step]}
        test_trails: {(edge, group): [residual_or_None per step]}
        group_by_step: [{group: [residuals]} per step]
        steps: list of step indices in order
    """
    entries = sorted(entries, key=lambda e: e["step"])
    steps = [e["step"] for e in entries]
    n_steps = len(steps)

    all_train_keys: set[tuple[tuple[int, int], int]] = set()
    all_test_keys: set[tuple[tuple[int, int], int]] = set()
    for entry in entries:
        for r in entry.get("train_residuals", []):
            all_train_keys.add(((r["edge"][0], r["edge"][1]), r["group"]))
        for r in entry.get("test_residuals", []):
            all_test_keys.add(((r["edge"][0], r["edge"][1]), r["group"]))

    train_trails: dict[tuple, list[np.ndarray | None]] = {
        key: [None] * n_steps for key in sorted(all_train_keys)
    }
    test_trails: dict[tuple, list[np.ndarray | None]] = {
        key: [None] * n_steps for key in sorted(all_test_keys)
    }
    group_by_step: list[dict[int, list[np.ndarray]]] = [defaultdict(list) for _ in range(n_steps)]

    for step_idx, entry in enumerate(entries):
        for r in entry.get("train_residuals", []):
            key = ((r["edge"][0], r["edge"][1]), r["group"])
            vec = _l2_normalize(np.array(r["residual"], dtype=np.float64))
            if key in train_trails:
                train_trails[key][step_idx] = vec
            group_by_step[step_idx][r["group"]].append(vec)
        for r in entry.get("test_residuals", []):
            key = ((r["edge"][0], r["edge"][1]), r["group"])
            vec = _l2_normalize(np.array(r["residual"], dtype=np.float64))
            if key in test_trails:
                test_trails[key][step_idx] = vec
            group_by_step[step_idx][r["group"]].append(vec)

    return train_trails, test_trails, group_by_step, steps


def _plot_trail(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    *,
    color: Any,
    linestyle: str,
    linewidth: float,
    marker: str,
    markersize: float,
    markeredgewidth: float = 1.5,
    markeredgecolor: Any = None,
    alpha: float = 1.0,
) -> None:
    if not xs:
        return
    ax.plot(xs, ys, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha)
    ax.plot(
        xs[-1], ys[-1],
        marker=marker,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor if markeredgecolor is not None else color,
        color=color,
        linestyle="",
    )


def _plot_single_repeat(
    *,
    entries: list[dict[str, Any]],
    repeat_id: int | None,
    k: int,
    output_dir: Path,
) -> None:
    if len(entries) < 1:
        return

    train_trails, test_trails, group_by_step, steps = _build_trails(entries)

    # Fit PCA on group averages of train edges at the final step
    final_entry = sorted(entries, key=lambda e: e["step"])[-1]
    final_train_by_group: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in final_entry.get("train_residuals", []):
        final_train_by_group[r["group"]].append(_l2_normalize(np.array(r["residual"], dtype=np.float64)))

    pca_vecs: list[np.ndarray] = [
        np.mean(np.stack(vecs, axis=0), axis=0)
        for vecs in final_train_by_group.values()
        if vecs
    ]

    if len(pca_vecs) < 2:
        return

    X_final = np.stack(pca_vecs, axis=0)
    components, mean_vec, evr = _fit_pca_2d(X_final)

    def proj(v: np.ndarray) -> tuple[float, float]:
        return _project(v, components, mean_vec)

    # Project test trails
    proj_test: dict[tuple, tuple[list[float], list[float]]] = {}
    for key, residuals in test_trails.items():
        xs, ys = [], []
        for v in residuals:
            if v is not None:
                px, py = proj(v)
                xs.append(px)
                ys.append(py)
        if xs:
            proj_test[key] = (xs, ys)

    # Project group average trails (mean over all train edges per group at each step)
    train_by_group_step: dict[int, list[list[np.ndarray]]] = {
        group_y: [[] for _ in steps] for group_y in range(k)
    }
    for (edge, group_y), residuals in train_trails.items():
        for step_idx, v in enumerate(residuals):
            if v is not None:
                train_by_group_step[group_y][step_idx].append(v)

    proj_avg: dict[int, tuple[list[float], list[float]]] = {}
    for group_y in range(k):
        xs, ys = [], []
        for step_idx in range(len(steps)):
            vecs = train_by_group_step[group_y][step_idx]
            if vecs:
                mean_r = np.mean(np.stack(vecs, axis=0), axis=0)
                px, py = proj(mean_r)
                xs.append(px)
                ys.append(py)
        if xs:
            proj_avg[group_y] = (xs, ys)

    # Plot
    configure_seaborn_plot_style()
    colors = get_palette(name="colorblind", n_colors=max(k, 1))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # Test edge trails: dotted, lw=1.5, 'x' at end
    for (edge, group_y), (xs, ys) in proj_test.items():
        _plot_trail(
            ax, xs, ys,
            color=colors[group_y],
            linestyle=":", linewidth=1.5,
            marker="x", markersize=7, markeredgewidth=2.0,
            alpha=0.7,
        )

    # Group average trails: solid, lw=2, 'o' at end with white edge (seaborn style)
    for group_y, (xs, ys) in proj_avg.items():
        _plot_trail(
            ax, xs, ys,
            color=colors[group_y],
            linestyle="-", linewidth=2.0,
            marker="o", markersize=8, markeredgewidth=1.5,
            markeredgecolor="white",
        )

    legend_handles = [
        Line2D([0], [0], linestyle="-", linewidth=2.0, color="gray",
               marker="o", markersize=8, markeredgewidth=1.5, markeredgecolor="white",
               label="Train avg (by group)"),
        Line2D([0], [0], linestyle=":", linewidth=1.5, color="gray",
               marker="x", markersize=7, markeredgewidth=2.0, label="Test edge"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    # Set axis limits based on final-frame points only
    final_xs: list[float] = []
    final_ys: list[float] = []
    for xs, ys in proj_test.values():
        final_xs.append(xs[-1]); final_ys.append(ys[-1])
    for xs, ys in proj_avg.values():
        final_xs.append(xs[-1]); final_ys.append(ys[-1])
    if final_xs:
        x_pad = max((max(final_xs) - min(final_xs)) * 0.1, 0.01)
        y_pad = max((max(final_ys) - min(final_ys)) * 0.1, 0.01)
        ax.set_xlim(min(final_xs) - x_pad, max(final_xs) + x_pad)
        ax.set_ylim(min(final_ys) - y_pad, max(final_ys) + y_pad)

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    title = "Residual PCA trails" + ("" if repeat_id is None else f" (repeat {repeat_id})")
    ax.set_title(title)
    style_metric_axis(axis=ax)

    suffix = "" if repeat_id is None else f"_repeat_{repeat_id}"
    png_path = output_dir / f"pca_trails{suffix}.png"
    pdf_path = output_dir / f"pca_trails{suffix}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved PCA plot  → {png_path}")


def plot_residual_pca(
    *,
    residuals_file: Path,
    output_dir: Path,
    k: int,
) -> None:
    """Plot PCA trails of residuals across training steps.

    Fits PCA on the final eval step's residuals, projects all steps' residuals
    onto those axes, and plots trails coloured by group index using the magma
    colormap. One plot is produced per repeat_id found in the data.
    """
    if not residuals_file.exists():
        print(f"  No residuals file found at {residuals_file}, skipping PCA plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_entries: list[dict[str, Any]] = json.loads(residuals_file.read_text(encoding="utf-8"))

    by_repeat: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for entry in all_entries:
        rid = entry.get("repeat_id")
        by_repeat[rid].append(entry)

    for repeat_id, entries in sorted(by_repeat.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        _plot_single_repeat(
            entries=entries,
            repeat_id=repeat_id,
            k=k,
            output_dir=output_dir,
        )


