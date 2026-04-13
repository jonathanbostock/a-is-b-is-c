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

from .probe_plot import plot_linear_probe
from .utils import configure_seaborn_plot_style, get_palette, lineplot_with_band, style_metric_axis

Edge = tuple[int, int]


# ── Graph helpers ─────────────────────────────────────────────────────────────

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


def _setup_graph_axis(axis: Axes) -> None:
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)


# ── Data extraction ───────────────────────────────────────────────────────────

def _extract_series(
    results: list[dict], key: str, field: str = "log_odds_gap"
) -> tuple[list[int], dict[Edge, list[float]], dict[Edge, list[float]]]:
    by_step_edge: dict[int, dict[Edge, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        step = int(row["step"])
        for entry in row[key]:
            if field not in entry:
                continue
            edge = (entry["edge"][0], entry["edge"][1])
            by_step_edge[step][edge].append(float(entry[field]))

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
            std_series[edge].append(
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )

    return steps, mean_series, std_series


def _shared_y_limits(
    *,
    train_means: dict[Edge, list[float]],
    train_stds: dict[Edge, list[float]],
    test_means: dict[Edge, list[float]],
    test_stds: dict[Edge, list[float]],
) -> tuple[float, float] | None:
    y_values: list[float] = []
    for means, stds in [(train_means, train_stds), (test_means, test_stds)]:
        for edge, mean_vals in means.items():
            std_vals = stds.get(edge, [])
            for m, s in zip(mean_vals, std_vals):
                if np.isfinite(m) and np.isfinite(s):
                    y_values.extend([m - s, m + s])
    if not y_values:
        return None
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.05 * (y_max - y_min) if y_max != y_min else 1.0
    return y_min - padding, y_max + padding


# ── Core plot builder ─────────────────────────────────────────────────────────

def _build_metric_figure(
    results: list[dict],
    *,
    n_categories: int,
    topology_train: list[Edge],
    field: str,
    train_title: str,
    test_title: str,
    y_label: str,
    residuals_file: Path | None = None,
    fixed_y_limits: tuple[float, float] | None = None,
    chance_line: float | None = None,
) -> tuple[plt.Figure, Path, Path]:
    """Build and return the matplotlib figure, without saving.

    Returns (figure, train_title_for_filename, ...) — callers save the file.
    """
    steps_train, train_means, train_stds = _extract_series(results, "train_edges", field)
    steps_test, test_means, test_stds = _extract_series(results, "test_edges", field)

    train_edges = sorted(train_means.keys())
    test_edges = sorted(test_means.keys())
    has_test = bool(test_edges)
    has_probe = residuals_file is not None and residuals_file.exists()

    configure_seaborn_plot_style()
    train_colors = get_palette(name="viridis", n_colors=len(train_edges))
    test_colors = get_palette(name="tab10", n_colors=len(test_edges)) if has_test else []

    n_data_rows = 2 if has_test else 1
    n_rows = n_data_rows + (1 if has_probe else 0)
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

    if chance_line is not None:
        ax_train_plot.axhline(chance_line, linestyle=":", color="black", linewidth=1)
    else:
        ax_train_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)

    ax_train_plot.set_title(f"Train-edge {train_title}")
    ax_train_plot.set_xlabel("Training step")
    ax_train_plot.set_ylabel(y_label)
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

        if chance_line is not None:
            ax_test_plot.axhline(chance_line, linestyle=":", color="black", linewidth=1)
        else:
            ax_test_plot.axhline(0.0, linestyle="--", color="black", linewidth=1)

        ax_test_plot.set_title(f"Test-edge {test_title}")
        ax_test_plot.set_xlabel("Training step")
        ax_test_plot.set_ylabel(y_label)
        ax_test_plot.set_xscale("symlog", linthresh=1)
        if steps_test:
            ax_test_plot.set_xlim(min(steps_test), max(steps_test))
        ax_test_plot.margins(x=0)
        style_metric_axis(axis=ax_test_plot)

        if fixed_y_limits is not None:
            ax_train_plot.set_ylim(*fixed_y_limits)
            ax_test_plot.set_ylim(*fixed_y_limits)
        else:
            shared_limits = _shared_y_limits(
                train_means=train_means, train_stds=train_stds,
                test_means=test_means, test_stds=test_stds,
            )
            if shared_limits is not None:
                ax_train_plot.set_ylim(*shared_limits)
                ax_test_plot.set_ylim(*shared_limits)

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

    if has_probe:
        ax_probe = axes[n_data_rows, 0]
        axes[n_data_rows, 1].set_visible(False)
        plot_linear_probe(residuals_file=residuals_file, axis=ax_probe)

    return figure


# ── Public API ────────────────────────────────────────────────────────────────

def plot_topology_results(
    *,
    eval_results_file: Path,
    output_dir: Path,
    n_categories: int,
    topology_train: list[Edge],
    residuals_file: Path | None = None,
) -> None:
    if not eval_results_file.exists():
        raise FileNotFoundError(f"No eval results found at {eval_results_file}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(eval_results_file.read_text(encoding="utf-8"))

    figure = _build_metric_figure(
        results,
        n_categories=n_categories,
        topology_train=topology_train,
        field="log_odds_gap",
        train_title="log-odds gap",
        test_title="log-odds gap",
        y_label="Log P(correct) − Log P(random)",
        residuals_file=residuals_file,
    )

    png_path = output_dir / "topology_plot.png"
    pdf_path = output_dir / "topology_plot.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(f"  Saved plot  → {png_path}")


def plot_accuracy_results(
    *,
    eval_results_file: Path,
    output_dir: Path,
    n_categories: int,
    topology_train: list[Edge],
    k: int,
    residuals_file: Path | None = None,
) -> None:
    if not eval_results_file.exists():
        raise FileNotFoundError(f"No eval results found at {eval_results_file}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(eval_results_file.read_text(encoding="utf-8"))

    # Skip gracefully if accuracy was not recorded (older runs)
    first_train = (results[0].get("train_edges") or [{}])[0] if results else {}
    if "accuracy" not in first_train:
        return

    figure = _build_metric_figure(
        results,
        n_categories=n_categories,
        topology_train=topology_train,
        field="accuracy",
        train_title="accuracy",
        test_title="accuracy",
        y_label="Accuracy",
        residuals_file=residuals_file,
        fixed_y_limits=(-0.05, 1.05),
        chance_line=1.0 / k,
    )

    png_path = output_dir / "accuracy_plot.png"
    pdf_path = output_dir / "accuracy_plot.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(f"  Saved plot  → {png_path}")
