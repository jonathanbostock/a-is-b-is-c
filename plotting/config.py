from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

Edge = tuple[int, int]


def merge_config(config_path: Path, default_config_path: Path) -> dict[str, Any]:
    """Load default config and merge experiment config on top."""
    with default_config_path.open("r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(default_config, dict):
        raise ValueError(f"Default config must be a YAML mapping: {default_config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")

    merged = dict(default_config)
    merged.update(config)
    if "train_topology" in merged:
        merged["train_topology"] = [tuple(edge) for edge in merged["train_topology"]]
    if "eval_topology" in merged:
        merged["eval_topology"] = [tuple(edge) for edge in merged["eval_topology"]]
    return merged


def resolve_topologies(config: dict[str, Any]) -> tuple[list[Edge], list[Edge] | None]:
    """Resolve train/eval topologies from config, supporting random splits via train_p."""
    if "train_p" in config:
        n_categories = int(config["n_categories"])
        train_p = float(config["train_p"])
        seed = int(config["seed"])

        all_edges: list[Edge] = [
            (i, j) for i in range(n_categories) for j in range(n_categories) if i != j
        ]
        total = len(all_edges)
        random.Random(seed).shuffle(all_edges)

        n_train = int(train_p * total)
        topology_train = all_edges[:n_train]

        if "eval_p" in config:
            eval_p = float(config["eval_p"])
            effective_eval_p = min(eval_p, 1.0 - train_p)
            n_eval = int(effective_eval_p * total)
            topology_eval: list[Edge] | None = all_edges[total - n_eval :] if n_eval > 0 else []
        else:
            topology_eval = None

        return topology_train, topology_eval
    else:
        topology_train = [(int(s), int(t)) for s, t in config["train_topology"]]
        topology_eval = (
            [(int(s), int(t)) for s, t in config["eval_topology"]]
            if "eval_topology" in config
            else None
        )
        return topology_train, topology_eval


def timestamped_output_dir(base_output_dir: Path, *, label: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}_{timestamp}" if label else f"_{timestamp}"
    return base_output_dir.parent / f"{base_output_dir.name}{suffix}"
