"""Regenerate topology_plot.png/pdf for all existing run directories.

Usage:
    uv run python replot.py [runs_dir]

For each run directory containing eval_results.json, reads the saved data
and regenerates the plot (including the linear probe row if residuals exist).
n_categories and topology_train are inferred from the saved eval results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plotting.topology_plot import plot_accuracy_results, plot_topology_results


def _infer_plot_args(run_dir: Path) -> tuple[int, list[tuple[int, int]]] | None:
    """Infer n_categories and topology_train from eval_results.json."""
    eval_file = run_dir / "eval_results.json"
    if not eval_file.exists():
        return None

    results = json.loads(eval_file.read_text(encoding="utf-8"))
    if not results:
        return None

    # Collect all train edges seen across all steps
    train_edges: set[tuple[int, int]] = set()
    max_node = 0
    for row in results:
        for entry in row.get("train_edges", []):
            e = (entry["edge"][0], entry["edge"][1])
            train_edges.add(e)
            max_node = max(max_node, e[0], e[1])
        for entry in row.get("test_edges", []):
            e = (entry["edge"][0], entry["edge"][1])
            max_node = max(max_node, e[0], e[1])

    # Try run_config_metadata.json first for n_categories
    config_file = run_dir / "run_config_metadata.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        n_categories = int(config.get("n_categories", max_node + 1))
    else:
        n_categories = max_node + 1

    return n_categories, sorted(train_edges)


def replot_run(run_dir: Path) -> None:
    args = _infer_plot_args(run_dir)
    if args is None:
        print(f"  Skipping {run_dir.name} (no eval_results.json)")
        return

    n_categories, topology_train = args
    eval_file = run_dir / "eval_results.json"
    residuals_file = run_dir / "residuals" / "pca_residuals.json"
    residuals_arg = residuals_file if residuals_file.exists() else None

    print(f"  Replotting {run_dir.name}  (n_cat={n_categories}, train_edges={len(topology_train)})")

    plot_topology_results(
        eval_results_file=eval_file,
        output_dir=run_dir,
        n_categories=n_categories,
        topology_train=topology_train,
        residuals_file=residuals_arg,
    )

    config_file = run_dir / "run_config_metadata.json"
    if config_file.exists():
        k = int(json.loads(config_file.read_text())["k"])
        plot_accuracy_results(
            eval_results_file=eval_file,
            output_dir=run_dir,
            n_categories=n_categories,
            topology_train=topology_train,
            k=k,
            residuals_file=residuals_arg,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", nargs="?", default="./runs", help="Directory containing run subdirs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Runs dir not found: {runs_dir}")
        sys.exit(1)

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and (d / "eval_results.json").exists())
    print(f"Found {len(run_dirs)} runs in {runs_dir}\n")

    for run_dir in run_dirs:
        replot_run(run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
