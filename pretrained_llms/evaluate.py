from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import importlib
import json
import math
import random
from pathlib import Path
from typing import Any

from .categories import CATEGORY_POOL
from .dataset import PromptExample


@dataclass(slots=True)
class EdgeEvalResult:
    edge: list[int]
    n_examples: int
    mean_logprob_correct: float
    mean_logprob_random: float
    log_odds_gap: float


@dataclass(slots=True)
class StepEvalResult:
    repeat_id: int
    step: int
    train_edges: list[EdgeEvalResult]
    test_edges: list[EdgeEvalResult]


def _log_softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    shifted = [value - max_value for value in values]
    exp_values = [math.exp(value) for value in shifted]
    total = sum(exp_values)
    return [value - math.log(total) for value in exp_values]


def _completion_logprob(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> float:
    torch = importlib.import_module("torch")

    full_text = f"{prompt} {completion}".strip()
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    completion_ids = tokenizer(f" {completion}", add_special_tokens=False, return_tensors="pt")["input_ids"]

    if full_ids.shape[1] < completion_ids.shape[1]:
        msg = "Completion tokenization failed due to mismatched token counts."
        raise ValueError(msg)

    with torch.no_grad():
        outputs = model(input_ids=full_ids.to(model.device))
        logits = outputs.logits[:, :-1, :]
        target_ids = full_ids[:, 1:].to(model.device)

    completion_token_count = completion_ids.shape[1]
    start_index = target_ids.shape[1] - completion_token_count
    if start_index < 0:
        msg = "Completion start index is negative."
        raise ValueError(msg)

    completion_logits = logits[:, start_index:, :].squeeze(0)
    completion_targets = target_ids[:, start_index:].squeeze(0)

    log_probs = torch.log_softmax(completion_logits, dim=-1)
    selected = log_probs.gather(1, completion_targets.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item())


def _random_negative_for_category(
    *, category: str, correct_completion: str, rng: random.Random, n_negatives: int
) -> list[str]:
    options = [item for item in CATEGORY_POOL[category] if item != correct_completion]
    if n_negatives <= len(options):
        return rng.sample(options, n_negatives)
    return [rng.choice(options) for _ in range(n_negatives)]


def evaluate_examples(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[PromptExample],
    step: int,
    n_negative_samples: int,
    rng: random.Random,
) -> list[EdgeEvalResult]:
    per_edge_correct: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_edge_random: dict[tuple[int, int], list[float]] = defaultdict(list)

    for example in examples:
        logprob_correct = _completion_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            completion=example.completion,
        )

        negative_completions = _random_negative_for_category(
            category=example.target_category,
            correct_completion=example.completion,
            rng=rng,
            n_negatives=n_negative_samples,
        )
        negative_scores = [
            _completion_logprob(model=model, tokenizer=tokenizer, prompt=example.prompt, completion=negative)
            for negative in negative_completions
        ]
        logprob_random = sum(negative_scores) / len(negative_scores)

        per_edge_correct[example.edge].append(logprob_correct)
        per_edge_random[example.edge].append(logprob_random)

    results: list[EdgeEvalResult] = []
    for edge in sorted(per_edge_correct):
        correct_values = per_edge_correct[edge]
        random_values = per_edge_random[edge]
        mean_correct = sum(correct_values) / len(correct_values)
        mean_random = sum(random_values) / len(random_values)
        gap_values = [correct - random for correct, random in zip(correct_values, random_values, strict=True)]
        gap = sum(gap_values) / len(gap_values)

        results.append(
            EdgeEvalResult(
                edge=[edge[0], edge[1]],
                n_examples=len(correct_values),
                mean_logprob_correct=mean_correct,
                mean_logprob_random=mean_random,
                log_odds_gap=gap,
            )
        )
    return results


def evaluate_step(
    *,
    model: Any,
    tokenizer: Any,
    repeat_id: int,
    step: int,
    eval_train_examples: list[PromptExample],
    eval_test_examples: list[PromptExample],
    seed: int,
    n_negative_samples: int = 3,
) -> StepEvalResult:
    rng = random.Random(seed + step)
    train_edges = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        examples=eval_train_examples,
        step=step,
        n_negative_samples=n_negative_samples,
        rng=rng,
    )
    test_edges = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        examples=eval_test_examples,
        step=step,
        n_negative_samples=n_negative_samples,
        rng=rng,
    )
    return StepEvalResult(repeat_id=repeat_id, step=step, train_edges=train_edges, test_edges=test_edges)


def append_eval_result(output_file: Path, result: StepEvalResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        existing = json.loads(output_file.read_text(encoding="utf-8"))
    else:
        existing = []
    existing.append(asdict(result))
    output_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def read_eval_results(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []
    return json.loads(output_file.read_text(encoding="utf-8"))
