from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from .eval_types import EdgeEvalResult, EvalExample, StepEvalResult

# Type aliases for the two pluggable compute functions.
# compute_logprob_batch: list of (prompt_ids, target_ids) -> list of log-probs
# compute_residual_batch: list of prompt_ids -> list of residual vectors
ComputeLogprobBatch = Callable[[list[tuple[list[int], list[int]]]], list[float]]
ComputeResidualBatch = Callable[[list[list[int]]], list[np.ndarray]]


def residual_layer_idx(n_layers: int) -> int:
    """Return the layer index closest to 3/4 of the way through the network."""
    return max(0, round(n_layers * 3 / 4) - 1)


def evaluate_examples(
    examples: list[EvalExample],
    compute_logprob_batch: ComputeLogprobBatch,
) -> list[EdgeEvalResult]:
    """Compute per-edge log-odds gap and accuracy for a list of eval examples.

    Calls compute_logprob_batch once with all (prompt, target) pairs — correct
    and negative — in a single flat list. The implementation of
    compute_logprob_batch can batch them however is efficient for the backend
    (e.g. one GPU forward pass for toy models, sequential for LLMs with
    variable-length sequences).

    gap = mean over examples of (lp_correct - mean(lp_negatives))
    accuracy = fraction of examples where lp_correct > mean(lp_negatives)
    """
    if not examples:
        return []

    # Build a flat list of (prompt_ids, target_ids) pairs: correct first, then negatives.
    all_pairs: list[tuple[list[int], list[int]]] = []
    # Track how many negatives each example has so we can slice the flat result.
    neg_counts: list[int] = []

    for ex in examples:
        all_pairs.append((ex.prompt_ids, ex.correct_ids))
        for neg_ids in ex.negative_ids:
            all_pairs.append((ex.prompt_ids, neg_ids))
        neg_counts.append(len(ex.negative_ids))

    all_logprobs = compute_logprob_batch(all_pairs)

    # Reconstruct per-example (correct, mean_neg) and pool by edge.
    per_edge_gaps: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_edge_correct: dict[tuple[int, int], list[bool]] = defaultdict(list)

    idx = 0
    for ex, n_neg in zip(examples, neg_counts):
        lp_correct = all_logprobs[idx]
        idx += 1
        lp_negs = all_logprobs[idx : idx + n_neg]
        idx += n_neg
        mean_neg = sum(lp_negs) / len(lp_negs) if lp_negs else lp_correct
        per_edge_gaps[ex.edge].append(lp_correct - mean_neg)
        per_edge_correct[ex.edge].append(lp_correct > max(lp_negs) if lp_negs else True)

    results: list[EdgeEvalResult] = []
    for edge in sorted(per_edge_gaps.keys()):
        gaps = per_edge_gaps[edge]
        corrects = per_edge_correct[edge]
        results.append(
            EdgeEvalResult(
                edge=list(edge),
                log_odds_gap=sum(gaps) / len(gaps),
                accuracy=sum(corrects) / len(corrects),
            )
        )
    return results


def collect_residuals(
    examples: list[EvalExample],
    compute_residual_batch: ComputeResidualBatch,
) -> list[dict]:
    """For each (edge, group) pair, compute the mean residual over all examples.

    Returns a list of dicts with keys: edge, group, residual.
    """
    if not examples:
        return []

    all_prompts = [ex.prompt_ids for ex in examples]
    all_residuals = compute_residual_batch(all_prompts)

    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = defaultdict(int)

    for ex, residual in zip(examples, all_residuals):
        key = (ex.edge, ex.group)
        if key in sums:
            sums[key] += residual
        else:
            sums[key] = residual.copy()
        counts[key] += 1

    return [
        {
            "edge": list(edge),
            "group": group,
            "residual": (sums[(edge, group)] / counts[(edge, group)]).tolist(),
        }
        for edge, group in sorted(sums.keys())
    ]
