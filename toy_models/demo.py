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
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Edge, Example, all_directed_edges, generate_examples, make_batch
from .magic_transformer import MagicTransformer
from .plot import plot_topology_results
from .transformer import ToyTransformer
from .vocab import Vocab
from plotting.config import merge_config, resolve_topologies, timestamped_output_dir
from plotting.pca_plot import plot_residual_pca

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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_log_odds_gap_per_edge(
    model: nn.Module,
    examples: list[Example],
    device: torch.device,
    k: int,
    batch_size: int = 512,
) -> dict[Edge, float]:
    """Per-edge log-odds gap: log(k) - cross_entropy_loss.

    Zero when the model guesses uniformly among the k valid answers;
    positive when it does better than random.
    """
    model.eval()
    edge_total: dict[Edge, float] = {}
    edge_count: dict[Edge, int] = {}

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = torch.tensor(
                [list(e.input_ids) for e in batch], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [e.target_id for e in batch], dtype=torch.long, device=device
            )
            logits = model(inputs)[:, -1, :]
            losses = F.cross_entropy(logits, targets, reduction="none")
            for ex, loss_val in zip(batch, losses.tolist()):
                edge_total[ex.edge] = edge_total.get(ex.edge, 0.0) + loss_val
                edge_count[ex.edge] = edge_count.get(ex.edge, 0) + 1

    model.train()
    log_k = math.log(k)
    return {edge: log_k - edge_total[edge] / edge_count[edge] for edge in edge_total}


def evaluate_accuracy(
    model: nn.Module,
    examples: list[Example],
    device: torch.device,
    batch_size: int = 512,
) -> dict[Edge, float]:
    model.eval()
    edge_correct: dict[Edge, int] = {}
    edge_total: dict[Edge, int] = {}

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = torch.tensor(
                [list(e.input_ids) for e in batch], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [e.target_id for e in batch], dtype=torch.long, device=device
            )
            logits = model(inputs)[:, -1, :]
            preds = logits.argmax(dim=-1)
            correct = preds == targets
            for ex, c in zip(batch, correct.tolist()):
                edge_correct[ex.edge] = edge_correct.get(ex.edge, 0) + int(c)
                edge_total[ex.edge] = edge_total.get(ex.edge, 0) + 1

    model.train()
    return {e: edge_correct[e] / edge_total[e] for e in edge_correct}


# ── Residual extraction ────────────────────────────────────────────────────────

def _residual_layer_idx(n_layers: int) -> int:
    """Return the layer index closest to 3/4 of the way through the network."""
    return max(0, round(n_layers * 3 / 4) - 1)


def evaluate_residuals_per_edge_group(
    model: ToyTransformer,
    examples: list[Example],
    device: torch.device,
    layer_idx: int,
    batch_size: int = 512,
) -> list[dict]:
    """For each (edge, group) pair, extract the mean residual at layer_idx on the final token.

    Returns a list of dicts with keys: edge, group, residual.
    """
    from collections import defaultdict

    edge_group_sums: dict[tuple[Edge, int], np.ndarray] = {}
    edge_group_counts: dict[tuple[Edge, int], int] = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = torch.tensor(
                [list(e.input_ids) for e in batch], dtype=torch.long, device=device
            )
            residuals = model.get_residual_at_layer(inputs, layer_idx).float().cpu().numpy()
            for ex, res in zip(batch, residuals):
                key = (ex.edge, ex.group)
                if key in edge_group_sums:
                    edge_group_sums[key] += res
                else:
                    edge_group_sums[key] = res.copy()
                edge_group_counts[key] += 1
    model.train()

    return [
        {
            "edge": list(edge),
            "group": group,
            "residual": (edge_group_sums[(edge, group)] / edge_group_counts[(edge, group)]).tolist(),
        }
        for edge, group in sorted(edge_group_sums.keys())
    ]


# ── Experiment: learned transformer ──────────────────────────────────────────

def run_learned_experiment(
    config: dict[str, Any], output_dir: Path, device: torch.device
) -> None:
    print("=" * 60)
    print("Experiment 1: ToyTransformer")
    print("=" * 60)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    rng = random.Random(seed)

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

    n_layers = int(config["n_layers"])
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

    n_steps = int(config["num_steps"])
    eval_every = int(config["eval_every"])
    batch_size = int(config["batch_size"])

    # Residual analysis setup
    residual_layer = _residual_layer_idx(n_layers)
    sorted_train_edges = sorted(train_edge_set)
    if len(sorted_train_edges) > 8:
        residual_train_edges = rng.sample(sorted_train_edges, 8)
    else:
        residual_train_edges = sorted_train_edges
    residual_train_examples = [e for e in train_examples if e.edge in set(residual_train_edges)]
    residual_steps: list[dict] = []

    print(f"  N={n_categories}  k={vocab.k}  d_model={config['d_model']}")
    print(f"  Train edges : {sorted(train_edge_set)}")
    print(f"  Test  edges : {sorted(test_edges)}")
    print(f"  Parameters  : {model.count_params():,}")

    eval_results: list[dict] = []

    # Dense early evals (powers of 2) plus regular eval_every steps
    eval_steps: set[int] = set(range(0, n_steps + 1, eval_every))
    eval_steps.add(n_steps)
    p = 1
    while p < eval_every:
        eval_steps.add(min(p, n_steps))
        p *= 2

    for step in range(n_steps + 1):
        if step in eval_steps:
            train_gap = evaluate_log_odds_gap_per_edge(model, train_examples, device, vocab.k)
            test_gap = (
                evaluate_log_odds_gap_per_edge(model, test_examples, device, vocab.k)
                if test_examples else {}
            )

            avg_train = sum(train_gap.values()) / len(train_gap) if train_gap else float("nan")
            avg_test = sum(test_gap.values()) / len(test_gap) if test_gap else float("nan")
            print(
                f"  step {step:6d} | train gap {avg_train:.4f}"
                + (f" | test gap {avg_test:.4f}" if test_examples else "")
            )

            eval_results.append({
                "step": step,
                "train_edges": [
                    {"edge": list(edge), "log_odds_gap": gap}
                    for edge, gap in sorted(train_gap.items())
                ],
                "test_edges": [
                    {"edge": list(edge), "log_odds_gap": gap}
                    for edge, gap in sorted(test_gap.items())
                ],
            })

            train_res = evaluate_residuals_per_edge_group(
                model, residual_train_examples, device, residual_layer
            )
            test_res = evaluate_residuals_per_edge_group(
                model, test_examples, device, residual_layer
            ) if test_examples else []
            residual_steps.append({
                "step": step,
                "repeat_id": None,
                "layer_idx": residual_layer,
                "train_residuals": train_res,
                "test_residuals": test_res,
            })

        if step == n_steps:
            break

        inputs, targets = make_batch(train_examples, batch_size=batch_size, rng=rng)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    print(f"\n  Saved eval results → {results_path}")

    residuals_dir = output_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = residuals_dir / "pca_residuals.json"
    residuals_path.write_text(json.dumps(residual_steps, indent=2), encoding="utf-8")
    print(f"  Saved residuals    → {residuals_path}")

    plot_topology_results(
        eval_results_file=results_path,
        output_dir=output_dir,
        n_categories=n_categories,
        topology_train=list(train_edge_set),
    )
    plot_residual_pca(
        residuals_file=residuals_path,
        output_dir=output_dir,
        k=vocab.k,
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
    acc = evaluate_accuracy(model, examples, device)
    mean_acc = sum(acc.values()) / len(acc)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}\n")

    run_learned_experiment(config, output_dir, device)
    run_magic_demo(device)


if __name__ == "__main__":
    main()
