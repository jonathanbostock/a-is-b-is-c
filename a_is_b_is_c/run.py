from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .dataset import PromptExample, build_run_data, write_examples_jsonl, write_metadata
from .plot import plot_topology_results
from .train import TrainingConfig, run_single_repeat_training


def _parse_topology(raw: str) -> list[tuple[int, int]]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        msg = "Topology must be a JSON list of [source, target] pairs."
        raise ValueError(msg)
    edges: list[tuple[int, int]] = []
    for item in parsed:
        if not isinstance(item, list | tuple) or len(item) != 2:
            msg = f"Invalid edge entry: {item}"
            raise ValueError(msg)
        source, target = item
        edges.append((int(source), int(target)))
    return edges


def _build_repeat_buckets(examples: list[PromptExample]) -> dict[int, list[PromptExample]]:
    by_repeat: dict[int, list[PromptExample]] = {}
    for example in examples:
        by_repeat.setdefault(example.repeat_id, []).append(example)
    return by_repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bostock matching-game topology experiment")
    parser.add_argument("--config", type=str, default="a_is_b_is_c/config.yaml", help="Path to YAML config")
    parser.add_argument("--topology", type=str, default=None, help='JSON edge list, e.g. "[[0,1],[1,2]]"')
    parser.add_argument("--n_categories", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n_train_templates", type=int, default=None)
    parser.add_argument("--n_eval_templates", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Generate data and metadata only, skipping fine-tuning and eval",
    )
    return parser.parse_args()


def _merge_config(cli_args: argparse.Namespace) -> dict[str, Any]:
    with Path(cli_args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    merged = dict(config)
    for key, value in vars(cli_args).items():
        if value is not None and key != "config":
            merged[key] = value
    if cli_args.topology is not None:
        merged["topology"] = _parse_topology(cli_args.topology)
    else:
        merged["topology"] = [tuple(edge) for edge in merged["topology"]]
    return merged


def main() -> None:
    args = parse_args()
    config = _merge_config(args)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_data = build_run_data(
        n_repeats=int(config["n_repeats"]),
        n_categories=int(config["n_categories"]),
        k=int(config["k"]),
        topology_train=[(int(source), int(target)) for source, target in config["topology"]],
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

        training_config = TrainingConfig(
            max_steps=int(config["max_steps"]),
            eval_every=int(config["eval_every"]),
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
            topology_train=[(int(source), int(target)) for source, target in config["topology"]],
        )


if __name__ == "__main__":
    main()
