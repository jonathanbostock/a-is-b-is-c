from __future__ import annotations

import importlib
import random
from typing import Any

import numpy as np

from .categories import CATEGORY_POOL
from .dataset import PromptExample
from shared.eval_types import EvalExample, StepEvalResult
from shared.evaluate import collect_residuals, evaluate_examples, residual_layer_idx
from shared.io import append_eval_result, append_residual_step, read_eval_results

__all__ = [
    "append_eval_result",
    "append_residual_step",
    "read_eval_results",
    "residual_layer_idx",
    "evaluate_step",
    "collect_residuals_per_edge_group",
]


# ── Convert PromptExamples → EvalExamples ────────────────────────────────────

def _sample_negatives(
    *, category: str, correct_completion: str, rng: random.Random, n_negatives: int
) -> list[str]:
    options = [item for item in CATEGORY_POOL[category] if item != correct_completion]
    if n_negatives <= len(options):
        return rng.sample(options, n_negatives)
    return [rng.choice(options) for _ in range(n_negatives)]


def _to_eval_examples(
    examples: list[PromptExample],
    tokenizer: Any,
    rng: random.Random,
    n_negatives: int,
    chat_format: bool = False,
    system_prompt: str = "",
) -> list[EvalExample]:
    """Tokenize PromptExamples into EvalExamples with pre-tokenized negatives.

    In chat_format mode, the prompt is rendered through the tokenizer's chat
    template with add_generation_prompt=True, and the completion (correct or
    negative) is tokenized as a plain string that will be appended after the
    assistant header. This matches the training-time format for -Instruct
    fine-tunes.
    """
    def _render_prompt(ex: PromptExample) -> str:
        if not chat_format:
            return ex.prompt
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": ex.prompt})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _completion_prefix() -> str:
        # Non-chat concatenates prompt + " " + completion (see train.py).
        # Chat mode: the assistant header already ends the prompt at a newline
        # boundary; completions should be attached with no leading space.
        return "" if chat_format else " "

    result = []
    for ex in examples:
        prompt_text = _render_prompt(ex)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        cprefix = _completion_prefix()
        correct_ids = tokenizer(f"{cprefix}{ex.completion}", add_special_tokens=False)["input_ids"]
        negative_strings = _sample_negatives(
            category=ex.target_category,
            correct_completion=ex.completion,
            rng=rng,
            n_negatives=n_negatives,
        )
        negative_ids = [
            tokenizer(f"{cprefix}{neg}", add_special_tokens=False)["input_ids"]
            for neg in negative_strings
        ]
        result.append(EvalExample(
            edge=ex.edge,
            group=ex.group,
            prompt_ids=prompt_ids,
            correct_ids=correct_ids,
            negative_ids=negative_ids,
        ))
    return result


# ── compute_logprob_batch for LLMs ───────────────────────────────────────────

def _make_compute_logprob_batch(model: Any, tokenizer: Any) -> Any:
    """Return a sequential logprob function for variable-length LLM sequences.

    Concatenates prompt + completion and extracts the sum of log-probs over
    the completion tokens only.
    """
    torch = importlib.import_module("torch")

    def compute_logprob_batch(
        pairs: list[tuple[list[int], list[int]]]
    ) -> list[float]:
        results: list[float] = []
        with torch.no_grad():
            for prompt_ids, target_ids in pairs:
                full_ids = prompt_ids + target_ids
                input_tensor = torch.tensor([full_ids], dtype=torch.long, device=model.device)
                outputs = model(input_ids=input_tensor)
                # logits[i] predicts token[i+1]
                logits = outputs.logits[:, :-1, :]
                target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=model.device)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected = log_probs.gather(2, target_tensor.unsqueeze(-1)).squeeze(-1)
                # Sum over the completion tokens only
                n_completion = len(target_ids)
                completion_lp = float(selected[:, -n_completion:].sum().item())
                results.append(completion_lp)
        return results

    return compute_logprob_batch


# ── compute_residual_batch for LLMs ──────────────────────────────────────────

def _make_compute_residual_batch(model: Any, tokenizer: Any, layer_idx: int) -> Any:
    """Return a sequential residual function for LLMs.

    Uses output_hidden_states=True. hidden_states[0] is the embedding output,
    hidden_states[i+1] is the output after transformer layer i.
    """
    torch = importlib.import_module("torch")

    def compute_residual_batch(all_prompt_ids: list[list[int]]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        with torch.no_grad():
            for prompt_ids in all_prompt_ids:
                input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
                outputs = model(input_ids=input_tensor, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx + 1]  # (1, seq_len, d_model)
                results.append(hidden[0, -1, :].float().cpu().numpy())
        return results

    return compute_residual_batch


# ── Public evaluation functions ───────────────────────────────────────────────

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
    chat_format: bool = False,
    system_prompt: str = "",
) -> StepEvalResult:
    rng = random.Random(seed + step)
    compute_logprob = _make_compute_logprob_batch(model, tokenizer)

    train_eval = _to_eval_examples(eval_train_examples, tokenizer, rng, n_negative_samples,
                                   chat_format=chat_format, system_prompt=system_prompt)
    test_eval = _to_eval_examples(eval_test_examples, tokenizer, rng, n_negative_samples,
                                  chat_format=chat_format, system_prompt=system_prompt)

    train_edges = evaluate_examples(train_eval, compute_logprob)
    test_edges = evaluate_examples(test_eval, compute_logprob)

    return StepEvalResult(repeat_id=repeat_id, step=step, train_edges=train_edges, test_edges=test_edges)


def collect_residuals_per_edge_group(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[PromptExample],
    layer_idx: int,
) -> list[dict]:
    """For each (edge, group) pair, compute the mean residual over all templates."""
    rng = random.Random(0)  # negatives not used for residuals, rng is just for interface compatibility
    eval_examples = _to_eval_examples(examples, tokenizer, rng, n_negatives=1)
    compute_residual = _make_compute_residual_batch(model, tokenizer, layer_idx)
    return collect_residuals(eval_examples, compute_residual)
