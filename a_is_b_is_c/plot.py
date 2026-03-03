from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

Edge = tuple[int, int]


def _circular_positions(n_categories: int) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for index in range(n_categories):
        theta = (math.pi / 2) - (2 * math.pi * index / n_categories)
        positions[index] = (math.cos(theta), math.sin(theta))
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
            shrinkA=20,
            shrinkB=20,
        )
        axis.add_patch(arrow)


def _extract_series(
    results: list[dict], key: str
) -> tuple[list[int], dict[Edge, list[float]], dict[Edge, list[float]]]:
    steps = [row["step"] for row in results]
    mean_series: dict[Edge, list[float]] = {}
    std_series: dict[Edge, list[float]] = {}

    for row in results:
        for edge_result in row[key]:
            edge = (edge_result["edge"][0], edge_result["edge"][1])
            mean_series.setdefault(edge, []).append(edge_result["log_odds_gap"])
            std_series.setdefault(edge, []).append(edge_result["std_over_repeats"])
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

    train_colors = plt.get_cmap("viridis")(np.linspace(0, 1, max(len(train_edges), 1)))
    test_colors = plt.cm.get_cmap("colorblind", max(len(test_edges), 1))(np.linspace(0, 1, max(len(test_edges), 1)))

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_train_plot, ax_train_graph = axes[0]
    ax_test_plot, ax_test_graph = axes[1]

    for index, edge in enumerate(train_edges):
        mean_values = np.array(train_means[edge], dtype=float)
        std_values = np.array(train_stds[edge], dtype=float)
        color = train_colors[index]
        ax_train_plot.plot(steps_train, mean_values, color=color, label=f"{edge[0]}→{edge[1]}")
        ax_train_plot.fill_between(steps_train, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

    ax_train_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax_train_plot.set_title("Train-edge log-odds gap")
    ax_train_plot.set_xlabel("Training step")
    ax_train_plot.set_ylabel("Log P(correct) - Log P(random)")
    ax_train_plot.legend(loc="best", fontsize=8)

    for index, edge in enumerate(test_edges):
        mean_values = np.array(test_means[edge], dtype=float)
        std_values = np.array(test_stds[edge], dtype=float)
        color = test_colors[index]
        ax_test_plot.plot(steps_test, mean_values, color=color, label=f"{edge[0]}→{edge[1]}")
        ax_test_plot.fill_between(steps_test, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

    ax_test_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax_test_plot.set_title("Test-edge log-odds gap")
    ax_test_plot.set_xlabel("Training step")
    ax_test_plot.set_ylabel("Log P(correct) - Log P(random)")
    ax_test_plot.legend(loc="best", fontsize=8)

    positions = _circular_positions(n_categories)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_categories))
    labels = {node: f"C{node}" for node in graph.nodes}

    nx.draw_networkx_nodes(graph, positions, ax=ax_train_graph, node_color="#e5e7eb", edgecolors="#111827", node_size=1200)
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax_train_graph)
    _draw_directed_edges(
        axis=ax_train_graph,
        positions=positions,
        edges=train_edges,
        edge_colors=[train_colors[index] for index in range(len(train_edges))],
        linewidth=2.0,
    )
    ax_train_graph.set_title("Training topology")
    ax_train_graph.set_axis_off()

    nx.draw_networkx_nodes(graph, positions, ax=ax_test_graph, node_color="#e5e7eb", edgecolors="#111827", node_size=1200)
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax_test_graph)
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
    ax_test_graph.set_title("Test topology overlay")
    ax_test_graph.set_axis_off()

    png_path = output_dir / "topology_plot.png"
    pdf_path = output_dir / "topology_plot.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
