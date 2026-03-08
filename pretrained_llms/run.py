from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import json

from .dataset import Edge, PromptExample, build_run_data, write_examples_jsonl, write_metadata
from .plot import plot_topology_results
from .train import TrainingConfig, run_single_repeat_training
from plotting.config import merge_config, resolve_topologies, timestamped_output_dir
from plotting.pca_plot import plot_residual_pca

_DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"
_CONFIGS_DIR = Path(__file__).parent / "configs"


def _build_repeat_buckets(examples: list[PromptExample]) -> dict[int, list[PromptExample]]:
    by_repeat: dict[int, list[PromptExample]] = {}
    for example in examples:
        by_repeat.setdefault(example.repeat_id, []).append(example)
    return by_repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bostock matching-game topology experiment")
    parser.add_argument("config", type=str, help="Experiment yaml (stem or path)")
    parser.add_argument(
        "model_config", nargs="?", default=None,
        help="Optional model config stem; looks in pretrained_llms/configs/",
    )
    return parser.parse_args()


def _load_config(cli_args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(cli_args.config)
    if config_path.suffix != ".yaml":
        config_path = Path("experiments") / f"{cli_args.config}.yaml"
    if cli_args.model_config is not None:
        default_path = _CONFIGS_DIR / f"{cli_args.model_config}.yaml"
    else:
        default_path = _DEFAULT_CONFIG
    return merge_config(config_path, default_path)


def main() -> None:
    args = parse_args()
    config = _load_config(args)

    base_output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir = timestamped_output_dir(base_output_dir, label="llm")
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_train, topology_eval = resolve_topologies(config)

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

    write_metadata(run_data.metadata, output_dir / "topology_metadata.json")
    run_config_path = output_dir / "run_config_metadata.json"
    run_config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
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
            warmup_ratio=float(config.get("warmup_ratio", 0.0)),
            batch_size=int(config["batch_size"]),
            grad_accum=int(config["grad_accum"]),
            lora_r=int(config["lora_r"]),
            use_lora=bool(config.get("use_lora", True)),
            lora_target_modules=list(config["lora_target_modules"]) if "lora_target_modules" in config else None,
            load_in_4bit=bool(config.get("load_in_4bit", False)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
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
        plot_residual_pca(
            residuals_file=output_dir / "residuals" / "pca_residuals.json",
            output_dir=output_dir,
            k=int(config["k"]),
        )



if __name__ == "__main__":
    main()
