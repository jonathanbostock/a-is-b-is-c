from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes


def configure_seaborn_plot_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams["axes.grid"] = False


def get_palette(*, name: str, n_colors: int) -> list[tuple[float, float, float]]:
    return sns.color_palette(name, n_colors=max(n_colors, 1))


def lineplot_with_band(
    *,
    axis: Axes,
    x_values: Sequence[int],
    mean_values: np.ndarray,
    std_values: np.ndarray,
    color: tuple[float, float, float],
    label: str,
) -> None:
    x_array = np.array(x_values, dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(mean_values) & np.isfinite(std_values)
    x_valid = x_array[valid_mask]
    mean_valid = mean_values[valid_mask]
    std_valid = std_values[valid_mask]
    if x_valid.size == 0:
        return

    order = np.argsort(x_valid)
    x_sorted = x_valid[order]
    mean_sorted = mean_valid[order]
    std_sorted = std_valid[order]

    axis.plot(x_sorted, mean_sorted, color=color, label=label, linewidth=2.0)
    axis.fill_between(x_sorted, mean_sorted - std_sorted, mean_sorted + std_sorted, color=color, alpha=0.2)


def style_metric_axis(*, axis: Axes) -> None:
    axis.grid(False)
    axis.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    sns.despine(ax=axis, top=True, right=True)
