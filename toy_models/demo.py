"""
Demo: toy transformer trained on the group-prediction task.

Task:
  Given tokens [BOS, cat_x, cat_x_grp_y, cat_z], predict cat_z_grp_y.
  The model must learn to copy the group index y from the in-context
  demonstration and apply it to the query category z.

Two experiments:
  1. A small randomly-initialised ToyTransformer (~1K params) trained from
     scratch on a subset of (x, z) edges; we measure whether it generalises
     to held-out edges (i.e. does it learn the copy-y rule or just memorise?).

  2. The MagicTransformer: hand-coded weights that solve the task perfectly
     for any N and k with 0 gradient steps, demonstrated on N=8, k=8.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Example, all_directed_edges, generate_examples, make_batch
from .magic_transformer import MagicTransformer
from .plot import plot_training_results
from .transformer import ToyTransformer
from .vocab import Vocab

# ── Experiment 1 config ──────────────────────────────────────────────────────
N = 5          # categories
K = 5          # groups
D_MODEL = 32   # embedding dim
N_HEADS = 2
N_LAYERS = 1
LR = 3e-3
BATCH_SIZE = 64
N_STEPS = 3000
EVAL_EVERY = 300
SEED = 42

# Training topology: a directed cycle plus a few skip edges
TRAIN_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # cycle
    (0, 2), (1, 3),                            # skip edges
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    examples: list[Example],
    device: torch.device,
    batch_size: int = 512,
) -> dict[tuple[int, int], float]:
    model.eval()
    edge_correct: dict[tuple[int, int], int] = {}
    edge_total: dict[tuple[int, int], int] = {}

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


# ── Experiment 1: learned transformer ────────────────────────────────────────

def run_learned_demo(device: torch.device) -> None:
    print("=" * 60)
    print(f"Experiment 1: ToyTransformer  N={N}  k={K}")
    print("=" * 60)

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    vocab = Vocab(n_categories=N, k=K)
    all_edges = all_directed_edges(N)
    train_edge_set = set(TRAIN_EDGES)
    test_edges = [e for e in all_edges if e not in train_edge_set]

    train_examples = generate_examples(vocab=vocab, edges=list(train_edge_set))
    test_examples = generate_examples(vocab=vocab, edges=test_edges)

    print(f"  Vocab size : {vocab.vocab_size}")
    print(f"  Train edges: {len(train_edge_set)}  ({len(train_examples)} examples)")
    print(f"  Test  edges: {len(test_edges)}  ({len(test_examples)} examples)")

    model = ToyTransformer(
        vocab_size=vocab.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ).to(device)
    print(f"  Parameters : {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_acc_history: dict[tuple[int, int], list[float]] = {e: [] for e in sorted(train_edge_set)}
    test_acc_history: dict[tuple[int, int], list[float]] = {e: [] for e in sorted(test_edges)}
    steps_history: list[int] = []

    for step in range(N_STEPS + 1):
        if step % EVAL_EVERY == 0:
            tr = evaluate(model, train_examples, device)
            te = evaluate(model, test_examples, device)
            mean_tr = sum(tr.values()) / len(tr) if tr else 0.0
            mean_te = sum(te.values()) / len(te) if te else 0.0
            print(f"  step {step:5d} | train acc {mean_tr:.3f} | test acc {mean_te:.3f}")
            for e in sorted(train_edge_set):
                train_acc_history[e].append(tr.get(e, 0.0))
            for e in sorted(test_edges):
                test_acc_history[e].append(te.get(e, 0.0))
            steps_history.append(step)

        if step == N_STEPS:
            break

        inputs, targets = make_batch(train_examples, batch_size=BATCH_SIZE, rng=rng)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    output_dir = Path("runs/toy_demo")
    plot_training_results(
        steps=steps_history,
        train_acc=train_acc_history,
        test_acc=test_acc_history,
        output_path=output_dir / "accuracy.png",
    )


# ── Experiment 2: magic transformer ──────────────────────────────────────────

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

    # Evaluate on ALL possible (x, z, y) triples
    all_edges = all_directed_edges(8)
    examples = generate_examples(vocab=vocab, edges=all_edges)
    acc = evaluate(model, examples, device)
    mean_acc = sum(acc.values()) / len(acc)
    print(f"  Accuracy on ALL {len(all_edges)} edges × {vocab.k} groups: {mean_acc:.4f}")

    # Show a few example predictions
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    run_learned_demo(device)
    run_magic_demo(device)


if __name__ == "__main__":
    main()
