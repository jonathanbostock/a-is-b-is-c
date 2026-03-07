from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from .plotting_utils import configure_seaborn_plot_style, get_palette, lineplot_with_band, style_metric_axis

Edge = tuple[int, int]


def _shared_y_limits(
    *,
    train_means: dict[Edge, list[float]],
    train_stds: dict[Edge, list[float]],
    test_means: dict[Edge, list[float]],
    test_stds: dict[Edge, list[float]],
) -> tuple[float, float] | None:
    y_values: list[float] = []

    for edge, mean_values in train_means.items():
        std_values = train_stds.get(edge, [])
        for mean_value, std_value in zip(mean_values, std_values, strict=True):
            if np.isfinite(mean_value) and np.isfinite(std_value):
                y_values.extend([mean_value - std_value, mean_value + std_value])

    for edge, mean_values in test_means.items():
        std_values = test_stds.get(edge, [])
        for mean_value, std_value in zip(mean_values, std_values, strict=True):
            if np.isfinite(mean_value) and np.isfinite(std_value):
                y_values.extend([mean_value - std_value, mean_value + std_value])

    if not y_values:
        return None

    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        padding = 1.0
    else:
        padding = 0.05 * (y_max - y_min)
    return y_min - padding, y_max + padding


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
    edge_colors: list[tuple[float, float, float, float] | str],
    linewidth: float,
) -> None:
    for (source, target), color in zip(edges, edge_colors, strict=True):
        source_pos = positions[source]
        target_pos = positions[target]
        arrow = FancyArrowPatch(
            source_pos,
            target_pos,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=linewidth,
            color=color,
            connectionstyle="arc3,rad=0.15",
            shrinkA=12,
            shrinkB=12,
        )
        axis.add_patch(arrow)


def _extract_series(
    results: list[dict], key: str
) -> tuple[list[int], dict[Edge, list[float]], dict[Edge, list[float]]]:
    by_step_edge: dict[int, dict[Edge, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        step = int(row["step"])
        for edge_result in row[key]:
            edge = (edge_result["edge"][0], edge_result["edge"][1])
            by_step_edge[step][edge].append(float(edge_result["log_odds_gap"]))

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


def plot_topology_results(
    *,
    eval_results_file: Path,
    output_dir: Path,
    n_categories: int,
    topology_train: list[Edge],
) -> None:
    if not eval_results_file.exists():
        msg = f"No eval results found at {eval_results_file}."
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(eval_results_file.read_text(encoding="utf-8"))

    steps_train, train_means, train_stds = _extract_series(results, "train_edges")
    steps_test, test_means, test_stds = _extract_series(results, "test_edges")

    train_edges = sorted(train_means.keys())
    test_edges = sorted(test_means.keys())

    configure_seaborn_plot_style()
    train_colors = get_palette(name="viridis", n_colors=len(train_edges))
    test_colors = get_palette(name="tab10", n_colors=len(test_edges))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9, 6.75),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 0.85]},
    )
    ax_train_plot, ax_train_graph = axes[0]
    ax_test_plot, ax_test_graph = axes[1]

    for index, edge in enumerate(train_edges):
        mean_values = np.array(train_means[edge], dtype=float)
        std_values = np.array(train_stds[edge], dtype=float)
        color = train_colors[index]
        lineplot_with_band(
            axis=ax_train_plot,
            x_values=steps_train,
            mean_values=mean_values,
            std_values=std_values,
            color=color,
            label=f"{edge[0]}→{edge[1]}",
        )

    ax_train_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax_train_plot.set_title("Train-edge log-odds gap")
    ax_train_plot.set_xlabel("Training step")
    ax_train_plot.set_ylabel("Log P(correct) - Log P(random)")
    ax_train_plot.set_xscale("symlog", linthresh=1)
    if steps_train:
        ax_train_plot.set_xlim(min(steps_train), max(steps_train))
    ax_train_plot.margins(x=0)
    style_metric_axis(axis=ax_train_plot)

    for index, edge in enumerate(test_edges):
        mean_values = np.array(test_means[edge], dtype=float)
        std_values = np.array(test_stds[edge], dtype=float)
        color = test_colors[index]
        lineplot_with_band(
            axis=ax_test_plot,
            x_values=steps_test,
            mean_values=mean_values,
            std_values=std_values,
            color=color,
            label=f"{edge[0]}→{edge[1]}",
        )

    shared_limits = _shared_y_limits(
        train_means=train_means,
        train_stds=train_stds,
        test_means=test_means,
        test_stds=test_stds,
    )
    if shared_limits is not None:
        ax_train_plot.set_ylim(*shared_limits)
        ax_test_plot.set_ylim(*shared_limits)

    ax_test_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax_test_plot.set_title("Test-edge log-odds gap")
    ax_test_plot.set_xlabel("Training step")
    ax_test_plot.set_ylabel("Log P(correct) - Log P(random)")
    ax_test_plot.set_xscale("symlog", linthresh=1)
    if steps_test:
        ax_test_plot.set_xlim(min(steps_test), max(steps_test))
    ax_test_plot.margins(x=0)
    style_metric_axis(axis=ax_test_plot)

    positions = _circular_positions(n_categories)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_categories))

    nx.draw_networkx_nodes(graph, positions, ax=ax_train_graph, node_color="#111827", edgecolors="#111827", node_size=420)
    _draw_directed_edges(
        axis=ax_train_graph,
        positions=positions,
        edges=train_edges,
        edge_colors=[train_colors[index] for index in range(len(train_edges))],
        linewidth=2.0,
    )
    ax_train_graph.set_axis_off()
    ax_train_graph.set_aspect("equal", adjustable="box")
    ax_train_graph.set_xlim(-1.0, 1.0)
    ax_train_graph.set_ylim(-1.0, 1.0)

    nx.draw_networkx_nodes(graph, positions, ax=ax_test_graph, node_color="#111827", edgecolors="#111827", node_size=420)
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
        edge_colors=[test_colors[index] for index in range(len(test_edges))],
        linewidth=2.0,
    )
    ax_test_graph.set_axis_off()
    ax_test_graph.set_aspect("equal", adjustable="box")
    ax_test_graph.set_xlim(-1.0, 1.0)
    ax_test_graph.set_ylim(-1.0, 1.0)

    png_path = output_dir / "topology_plot.png"
    pdf_path = output_dir / "topology_plot.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
