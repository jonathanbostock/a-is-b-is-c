from __future__ import annotations

from dataclasses import dataclass


Edge = tuple[int, int]


@dataclass(slots=True)
class EvalExample:
    """Model-agnostic eval example with pre-tokenized inputs.

    For toy models: prompt_ids = [BOS, cat_x, cat_x_grp_y, cat_z],
                    correct_ids = [target_id],
                    negative_ids = [[neg_id], ...] for all k-1 negatives.

    For LLMs: prompt_ids = tokenizer(prompt)["input_ids"],
              correct_ids = tokenizer(" completion")["input_ids"],
              negative_ids = [tokenizer(" neg")["input_ids"], ...].
    """

    edge: Edge
    group: int
    prompt_ids: list[int]
    correct_ids: list[int]
    negative_ids: list[list[int]]


@dataclass(slots=True)
class EdgeEvalResult:
    edge: list[int]
    log_odds_gap: float
    accuracy: float


@dataclass(slots=True)
class StepEvalResult:
    repeat_id: int
    step: int
    train_edges: list[EdgeEvalResult]
    test_edges: list[EdgeEvalResult]
