from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.utils import configure_seaborn_plot_style, get_palette, style_metric_axis

Edge = tuple[int, int]


def plot_training_results(
    *,
    steps: list[int],
    train_acc: dict[Edge, list[float]],
    test_acc: dict[Edge, list[float]],
    output_path: Path,
) -> None:
    configure_seaborn_plot_style()

    train_edges = sorted(train_acc.keys())
    test_edges = sorted(test_acc.keys())
    train_colors = get_palette(name="viridis", n_colors=len(train_edges))
    test_colors = get_palette(name="tab10", n_colors=len(test_edges))

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for idx, edge in enumerate(train_edges):
        values = np.array(train_acc[edge], dtype=float)
        ax_train.plot(steps, values, color=train_colors[idx], label=f"{edge[0]}→{edge[1]}", linewidth=2)
    ax_train.set_title("Train-edge accuracy")
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Accuracy")
    ax_train.set_ylim(-0.05, 1.1)
    ax_train.axhline(1.0, linestyle="--", color="black", linewidth=1)
    style_metric_axis(axis=ax_train)
    ax_train.legend(fontsize=7, loc="lower right")

    for idx, edge in enumerate(test_edges):
        values = np.array(test_acc[edge], dtype=float)
        ax_test.plot(steps, values, color=test_colors[idx], label=f"{edge[0]}→{edge[1]}", linewidth=2)
    ax_test.set_title("Test-edge accuracy")
    ax_test.set_xlabel("Step")
    ax_test.set_ylabel("Accuracy")
    ax_test.set_ylim(-0.05, 1.1)
    ax_test.axhline(1.0, linestyle="--", color="black", linewidth=1)
    style_metric_axis(axis=ax_test)
    ax_test.legend(fontsize=7, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {output_path}")
