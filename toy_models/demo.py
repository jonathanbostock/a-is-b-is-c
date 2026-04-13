"""Toy transformer trained on the group-prediction task, driven by a YAML config.

Usage:
    uv run toy-demo experiments/directed_chain_1.yaml

Task:
  Given tokens [BOS, cat_x, cat_x_grp_y, cat_z], predict cat_z_grp_y.
  The model must learn to copy the group index y from the in-context
  demonstration and apply it to the query category z.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Edge, Example, all_directed_edges, generate_examples, make_batch
from .magic_transformer import MagicTransformer
from .transformer import ToyTransformer
from .vocab import Vocab
from plotting.config import merge_config, resolve_topologies, timestamped_output_dir
from plotting.topology_plot import plot_accuracy_results, plot_topology_results
from shared.eval_types import EvalExample, EdgeEvalResult, StepEvalResult
from shared.evaluate import collect_residuals, evaluate_examples, residual_layer_idx
from shared.io import append_eval_result, append_residual_step

_DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"
_CONFIGS_DIR = Path(__file__).parent / "configs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy transformer topology experiment")
    parser.add_argument("config", type=str, help="Experiment yaml (stem or path)")
    parser.add_argument(
        "model_config", nargs="?", default=None,
        help="Optional model config stem; looks in toy_models/configs/",
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


# ── Convert toy Examples → EvalExamples ───────────────────────────────────────

def _to_eval_examples(examples: list[Example], vocab: Vocab) -> list[EvalExample]:
    """Convert toy model Examples to EvalExamples with all k-1 negatives."""
    result = []
    for ex in examples:
        z = ex.edge[1]
        all_targets = [vocab.grp(z, y) for y in range(vocab.k)]
        negative_ids = [[t] for t in all_targets if t != ex.target_id]
        result.append(EvalExample(
            edge=ex.edge,
            group=ex.group,
            prompt_ids=list(ex.input_ids),
            correct_ids=[ex.target_id],
            negative_ids=negative_ids,
        ))
    return result


# ── compute_logprob_batch for toy model ───────────────────────────────────────

def _make_compute_logprob_batch(
    model: nn.Module, device: torch.device, batch_size: int = 512
) -> Any:
    """Return a batched logprob function for the toy transformer.

    All prompt_ids have the same length (4), so we can stack them into one
    tensor and do a single forward pass per chunk.

    Each pair is (prompt_ids, [target_id]): we run the model on prompt_ids
    and return log_softmax(logits[-1])[target_id].
    """
    def compute_logprob_batch(
        pairs: list[tuple[list[int], list[int]]]
    ) -> list[float]:
        model.eval()
        results: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start : start + batch_size]
                inputs = torch.tensor(
                    [p[0] for p in chunk], dtype=torch.long, device=device
                )
                targets = torch.tensor(
                    [p[1][0] for p in chunk], dtype=torch.long, device=device
                )
                logits = model(inputs)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                results.extend(selected.tolist())
        model.train()
        return results

    return compute_logprob_batch


def _make_compute_residual_batch(
    model: ToyTransformer, device: torch.device, layer_idx: int, batch_size: int = 512
) -> Any:
    """Return a batched residual function for the toy transformer."""
    def compute_residual_batch(all_prompt_ids: list[list[int]]) -> list[np.ndarray]:
        model.eval()
        results: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(all_prompt_ids), batch_size):
                chunk = all_prompt_ids[start : start + batch_size]
                inputs = torch.tensor(chunk, dtype=torch.long, device=device)
                residuals = model.get_residual_at_layer(inputs, layer_idx).float().cpu().numpy()
                results.extend(list(residuals))
        model.train()
        return results

    return compute_residual_batch


# ── Experiment: learned transformer ──────────────────────────────────────────

def run_learned_experiment(
    config: dict[str, Any], output_dir: Path, device: torch.device
) -> None:
    print("=" * 60)
    print("Experiment 1: ToyTransformer")
    print("=" * 60)

    seed = int(config["seed"])
    num_repeats = int(config.get("num_repeats", 1))

    topology_train, topology_eval = resolve_topologies(config)
    n_categories = int(config["n_categories"])

    all_edges = all_directed_edges(n_categories)
    train_edge_set = set(topology_train)
    if topology_eval is None:
        test_edges = [e for e in all_edges if e not in train_edge_set]
    else:
        test_edges = list(topology_eval)

    vocab = Vocab(n_categories=n_categories, k=int(config["k"]))
    train_examples = generate_examples(vocab=vocab, edges=list(train_edge_set))
    test_examples = generate_examples(vocab=vocab, edges=test_edges) if test_edges else []

    eval_train_examples = _to_eval_examples(train_examples, vocab)
    eval_test_examples = _to_eval_examples(test_examples, vocab)

    n_layers = int(config["n_layers"])
    n_steps = int(config["num_steps"])
    eval_every = int(config["eval_every"])
    batch_size = int(config["batch_size"])
    res_layer = residual_layer_idx(n_layers)

    print(f"  N={n_categories}  k={vocab.k}  d_model={config['d_model']}")
    print(f"  Train edges : {sorted(train_edge_set)}")
    print(f"  Test  edges : {sorted(test_edges)}")
    print(f"  Num repeats : {num_repeats}")

    # Eval at step 0, every power of 2, and the final step
    eval_steps: set[int] = {0}
    p = 1
    while p < n_steps:
        eval_steps.add(p)
        p *= 2
    eval_steps.add(n_steps)

    results_path = output_dir / "eval_results.json"
    residuals_dir = output_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = residuals_dir / "pca_residuals.json"

    for repeat_id in range(num_repeats):
        repeat_seed = seed + repeat_id
        torch.manual_seed(repeat_seed)
        rng = random.Random(repeat_seed)

        model = ToyTransformer(
            vocab_size=vocab.vocab_size,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=n_layers,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )

        compute_logprob = _make_compute_logprob_batch(model, device)
        compute_residual = _make_compute_residual_batch(model, device, res_layer)

        sorted_train_edges = sorted(train_edge_set)
        residual_train_edges = (
            rng.sample(sorted_train_edges, 8) if len(sorted_train_edges) > 8 else sorted_train_edges
        )
        residual_train_eval_examples = [
            e for e in eval_train_examples if e.edge in set(residual_train_edges)
        ]

        if num_repeats > 1:
            print(f"\n--- Repeat {repeat_id} (seed={repeat_seed}) ---")
        print(f"  Parameters  : {model.count_params():,}")

        for step in range(n_steps + 1):
            if step in eval_steps:
                train_edge_results = evaluate_examples(eval_train_examples, compute_logprob)
                test_edge_results = evaluate_examples(eval_test_examples, compute_logprob) if eval_test_examples else []

                avg_train = sum(r.log_odds_gap for r in train_edge_results) / len(train_edge_results) if train_edge_results else float("nan")
                avg_test = sum(r.log_odds_gap for r in test_edge_results) / len(test_edge_results) if test_edge_results else float("nan")
                print(
                    f"  step {step:6d} | train gap {avg_train:.4f}"
                    + (f" | test gap {avg_test:.4f}" if eval_test_examples else "")
                )

                step_result = StepEvalResult(
                    step=step,
                    repeat_id=repeat_id,
                    train_edges=train_edge_results,
                    test_edges=test_edge_results,
                )
                append_eval_result(results_path, step_result)

                train_res = collect_residuals(residual_train_eval_examples, compute_residual)
                test_res = collect_residuals(eval_test_examples, compute_residual) if eval_test_examples else []
                append_residual_step(
                    residuals_path,
                    step=step,
                    repeat_id=repeat_id,
                    layer_idx=res_layer,
                    train_residuals=train_res,
                    test_residuals=test_res,
                )

            if step == n_steps:
                break

            inputs, targets = make_batch(train_examples, batch_size=batch_size, rng=rng)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)[:, -1, :]
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"\n  Saved eval results → {results_path}")
    print(f"  Saved residuals    → {residuals_path}")

    plot_topology_results(
        eval_results_file=results_path,
        output_dir=output_dir,
        n_categories=n_categories,
        topology_train=list(train_edge_set),
        residuals_file=residuals_path,
    )
    plot_accuracy_results(
        eval_results_file=results_path,
        output_dir=output_dir,
        n_categories=n_categories,
        topology_train=list(train_edge_set),
        k=vocab.k,
        residuals_file=residuals_path,
    )


# ── Experiment: magic transformer ─────────────────────────────────────────────

def run_magic_demo(device: torch.device) -> None:
    print()
    print("=" * 60)
    print("Experiment 2: MagicTransformer  N=8  k=8  (zero training steps)")
    print("=" * 60)

    vocab = Vocab(n_categories=8, k=8)
    model = MagicTransformer(vocab).to(device)
    print(f"  d_model    : {model.d_model}")
    print(f"  Parameters : {model.count_params():,}")
    print(f"  Vocab size : {vocab.vocab_size}")

    all_edges = all_directed_edges(8)
    examples = generate_examples(vocab=vocab, edges=all_edges)
    eval_examples = _to_eval_examples(examples, vocab)
    compute_logprob = _make_compute_logprob_batch(model, device)
    edge_results = evaluate_examples(eval_examples, compute_logprob)
    mean_acc = sum(r.accuracy for r in edge_results) / len(edge_results)
    print(f"  Accuracy on ALL {len(all_edges)} edges × {vocab.k} groups: {mean_acc:.4f}")

    print("\n  Sample predictions (input → predicted vs correct):")
    rng = random.Random(0)
    for ex in rng.sample(examples, k=6):
        inp = torch.tensor([list(ex.input_ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(inp)[0, -1, :]
        pred = int(logits.argmax())
        names = [vocab.token_name(t) for t in ex.input_ids]
        print(
            f"    [{', '.join(names)}] → "
            f"pred={vocab.token_name(pred)}  "
            f"correct={vocab.token_name(ex.target_id)}  "
            f"{'✓' if pred == ex.target_id else '✗'}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    config = _load_config(args)

    base_output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir = timestamped_output_dir(base_output_dir, label="toy")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config_metadata.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}\n")

    run_learned_experiment(config, output_dir, device)
    run_magic_demo(device)


if __name__ == "__main__":
    main()
