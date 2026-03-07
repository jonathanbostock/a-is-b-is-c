from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from plotting.utils import configure_seaborn_plot_style, get_palette, lineplot_with_band, style_metric_axis

Edge = tuple[int, int]


# ── Simple aggregate loss plot (legacy) ───────────────────────────────────────

def plot_training_results(
    *,
    steps: list[int],
    train_loss: list[float],
    test_loss: list[float],
    output_path: Path,
) -> None:
    configure_seaborn_plot_style()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    ax.plot(steps, train_loss, label="train", linewidth=2)
    ax.plot(steps, test_loss, label="test", linewidth=2)
    ax.set_title("Cross-entropy loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    style_metric_axis(axis=ax)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {output_path}")


# ── Per-edge topology plot ─────────────────────────────────────────────────────

def _circular_positions(n_categories: int, *, radius: float = 0.62) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for index in range(n_categories):
        theta = (math.pi / 2) - (2 * math.pi * index / n_categories)
        positions[index] = (radius * math.cos(theta), radius * math.sin(theta))
    return positions


def _draw_directed_edges(
    *,
    axis: Axes,
    positions: dict[int, tuple[float, float]],
    edges: list[Edge],
    edge_colors: list,
    linewidth: float,
) -> None:
    for (source, target), color in zip(edges, edge_colors, strict=True):
        arrow = FancyArrowPatch(
            positions[source],
            positions[target],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=linewidth,
            color=color,
            connectionstyle="arc3,rad=0.15",
            shrinkA=12,
            shrinkB=12,
        )
        axis.add_patch(arrow)


def _extract_gap_series(
    results: list[dict], key: str
) -> tuple[list[int], dict[Edge, list[float]], dict[Edge, list[float]]]:
    by_step_edge: dict[int, dict[Edge, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        step = int(row["step"])
        for entry in row[key]:
            edge = (entry["edge"][0], entry["edge"][1])
            by_step_edge[step][edge].append(float(entry["log_odds_gap"]))

    steps = sorted(by_step_edge.keys())
    edges = sorted({edge for edge_map in by_step_edge.values() for edge in edge_map.keys()})

    mean_series: dict[Edge, list[float]] = {edge: [] for edge in edges}
    std_series: dict[Edge, list[float]] = {edge: [] for edge in edges}

    for step in steps:
        edge_map = by_step_edge[step]
        for edge in edges:
            values = edge_map.get(edge)
            if not values:
                mean_series[edge].append(float("nan"))
                std_series[edge].append(float("nan"))
                continue
            mean_series[edge].append(float(np.mean(values)))
            std_series[edge].append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    return steps, mean_series, std_series


def _setup_graph_axis(axis: Axes) -> None:
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)


def plot_topology_results(
    *,
    eval_results_file: Path,
    output_dir: Path,
    n_categories: int,
    topology_train: list[Edge],
) -> None:
    if not eval_results_file.exists():
        raise FileNotFoundError(f"No eval results found at {eval_results_file}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(eval_results_file.read_text(encoding="utf-8"))

    steps_train, train_means, train_stds = _extract_gap_series(results, "train_edges")
    steps_test, test_means, test_stds = _extract_gap_series(results, "test_edges")

    train_edges = sorted(train_means.keys())
    test_edges = sorted(test_means.keys())
    has_test = bool(test_edges)

    configure_seaborn_plot_style()
    train_colors = get_palette(name="viridis", n_colors=len(train_edges))
    test_colors = get_palette(name="tab10", n_colors=len(test_edges)) if has_test else []

    n_rows = 2 if has_test else 1
    figure, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9, 3.5 * n_rows),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 0.85]},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    ax_train_plot, ax_train_graph = axes[0]

    for index, edge in enumerate(train_edges):
        lineplot_with_band(
            axis=ax_train_plot,
            x_values=steps_train,
            mean_values=np.array(train_means[edge], dtype=float),
            std_values=np.array(train_stds[edge], dtype=float),
            color=train_colors[index],
            label=f"{edge[0]}→{edge[1]}",
        )

    ax_train_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax_train_plot.set_title("Train-edge log-odds gap")
    ax_train_plot.set_xlabel("Training step")
    ax_train_plot.set_ylabel("Log P(correct) − Log P(random)")
    ax_train_plot.set_xscale("symlog", linthresh=1)
    if steps_train:
        ax_train_plot.set_xlim(min(steps_train), max(steps_train))
    ax_train_plot.margins(x=0)
    style_metric_axis(axis=ax_train_plot)

    positions = _circular_positions(n_categories)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_categories))

    nx.draw_networkx_nodes(
        graph, positions, ax=ax_train_graph,
        node_color="#111827", edgecolors="#111827", node_size=420,
    )
    _draw_directed_edges(
        axis=ax_train_graph,
        positions=positions,
        edges=train_edges,
        edge_colors=[train_colors[i] for i in range(len(train_edges))],
        linewidth=2.0,
    )
    _setup_graph_axis(ax_train_graph)

    if has_test:
        ax_test_plot, ax_test_graph = axes[1]

        for index, edge in enumerate(test_edges):
            lineplot_with_band(
                axis=ax_test_plot,
                x_values=steps_test,
                mean_values=np.array(test_means[edge], dtype=float),
                std_values=np.array(test_stds[edge], dtype=float),
                color=test_colors[index],
                label=f"{edge[0]}→{edge[1]}",
            )

        ax_test_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
        ax_test_plot.set_title("Test-edge log-odds gap")
        ax_test_plot.set_xlabel("Training step")
        ax_test_plot.set_ylabel("Log P(correct) − Log P(random)")
        ax_test_plot.set_xscale("symlog", linthresh=1)
        if steps_test:
            ax_test_plot.set_xlim(min(steps_test), max(steps_test))
        ax_test_plot.margins(x=0)
        style_metric_axis(axis=ax_test_plot)

        # Shared y-axis across both plots
        all_y: list[float] = []
        for means, stds in [(train_means, train_stds), (test_means, test_stds)]:
            for edge, mean_vals in means.items():
                std_vals = stds.get(edge, [])
                for m, s in zip(mean_vals, std_vals):
                    if np.isfinite(m) and np.isfinite(s):
                        all_y.extend([m - s, m + s])
        if all_y:
            pad = 0.05 * (max(all_y) - min(all_y)) if max(all_y) != min(all_y) else 1.0
            ax_train_plot.set_ylim(min(all_y) - pad, max(all_y) + pad)
            ax_test_plot.set_ylim(min(all_y) - pad, max(all_y) + pad)

        nx.draw_networkx_nodes(
            graph, positions, ax=ax_test_graph,
            node_color="#111827", edgecolors="#111827", node_size=420,
        )
        _draw_directed_edges(
            axis=ax_test_graph,
            positions=positions,
            edges=sorted(topology_train),
            edge_colors=["#444444" for _ in topology_train],
            linewidth=1.5,
        )
        _draw_directed_edges(
            axis=ax_test_graph,
            positions=positions,
            edges=test_edges,
            edge_colors=[test_colors[i] for i in range(len(test_edges))],
            linewidth=2.0,
        )
        _setup_graph_axis(ax_test_graph)

    png_path = output_dir / "topology_plot.png"
    pdf_path = output_dir / "topology_plot.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(f"  Saved plot  → {png_path}")
