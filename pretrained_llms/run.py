from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random
from typing import Any

import yaml

from .dataset import Edge, PromptExample, all_directed_edges, build_run_data, write_examples_jsonl, write_metadata
from .plot import plot_topology_results
from .train import TrainingConfig, run_single_repeat_training


def _build_repeat_buckets(examples: list[PromptExample]) -> dict[int, list[PromptExample]]:
    by_repeat: dict[int, list[PromptExample]] = {}
    for example in examples:
        by_repeat.setdefault(example.repeat_id, []).append(example)
    return by_repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bostock matching-game topology experiment")
    parser.add_argument("config", type=str, help="Path to a .yaml experiment config file")
    return parser.parse_args()


def _merge_config(cli_args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(cli_args.config)
    if config_path.suffix != ".yaml":
        config_path = Path("experiments") / f"{cli_args.config}.yaml"

    default_config_path = Path("pretrained_llms/config.yaml")
    with default_config_path.open("r", encoding="utf-8") as handle:
        default_config = yaml.safe_load(handle)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(default_config, dict):
        msg = f"Default config must be a YAML mapping: {default_config_path}"
        raise ValueError(msg)
    if not isinstance(config, dict):
        msg = f"Experiment config must be a YAML mapping: {config_path}"
        raise ValueError(msg)

    merged = dict(default_config)
    merged.update(config)
    merged["train_topology"] = [tuple(edge) for edge in merged["train_topology"]]
    if "eval_topology" in merged:
        merged["eval_topology"] = [tuple(edge) for edge in merged["eval_topology"]]
    return merged


def _resolve_topologies(config: dict[str, Any]) -> tuple[list[Edge], list[Edge] | None]:
    if "train_p" in config:
        n_categories = int(config["n_categories"])
        train_p = float(config["train_p"])
        seed = int(config["seed"])

        all_edges = all_directed_edges(n_categories)
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


def _timestamped_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_output_dir.parent / f"{base_output_dir.name}_{timestamp}"


def main() -> None:
    args = parse_args()
    config = _merge_config(args)

    base_output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir = _timestamped_output_dir(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_train, topology_eval = _resolve_topologies(config)

    run_data = build_run_data(
        n_repeats=int(config["n_repeats"]),
        n_categories=int(config["n_categories"]),
        k=int(config["k"]),
        topology_train=topology_train,
        topology_eval=topology_eval,
        n_train_templates=int(config["n_train_templates"]),
        n_eval_templates=int(config["n_eval_templates"]),
        seed=int(config["seed"]),
    )

    write_metadata(run_data.metadata, output_dir / "metadata.json")
    write_examples_jsonl(run_data.train_examples, output_dir / "train_dataset.jsonl")
    write_examples_jsonl(run_data.eval_train_edge_examples, output_dir / "eval_train_edges.jsonl")
    write_examples_jsonl(run_data.eval_test_edge_examples, output_dir / "eval_test_edges.jsonl")

    if not bool(config.get("skip_train", False)):
        train_by_repeat = _build_repeat_buckets(run_data.train_examples)
        eval_train_by_repeat = _build_repeat_buckets(run_data.eval_train_edge_examples)
        eval_test_by_repeat = _build_repeat_buckets(run_data.eval_test_edge_examples)

        if "num_steps" in config:
            max_steps = int(config["num_steps"])
            eval_every = int(config["eval_every"])
        else:
            n_train_edges = len(topology_train)
            total_samples = int(config["examples_per_edge_per_k"]) * int(config["k"]) * n_train_edges
            samples_per_step = int(config["batch_size"]) * int(config["grad_accum"])
            max_steps = max(1, total_samples // samples_per_step)
            eval_every = max(1, max_steps // int(config["num_evals"]))
        training_config = TrainingConfig(
            max_steps=max_steps,
            eval_every=eval_every,
            lr=float(config["lr"]),
            batch_size=int(config["batch_size"]),
            grad_accum=int(config["grad_accum"]),
            lora_r=int(config["lora_r"]),
            seed=int(config["seed"]),
            output_dir=str(output_dir),
            model_name=str(config["model_name"]),
            max_seq_length=int(config["max_seq_length"]),
        )

        for repeat_id in range(int(config["n_repeats"])):
            run_single_repeat_training(
                repeat_id=repeat_id,
                repeat_train_examples=train_by_repeat.get(repeat_id, []),
                repeat_eval_train_examples=eval_train_by_repeat.get(repeat_id, []),
                repeat_eval_test_examples=eval_test_by_repeat.get(repeat_id, []),
                config=training_config,
                run_dir=output_dir,
            )

        plot_topology_results(
            eval_results_file=output_dir / "eval_results.json",
            output_dir=output_dir,
            n_categories=int(config["n_categories"]),
            topology_train=topology_train,
        )


if __name__ == "__main__":
    main()
