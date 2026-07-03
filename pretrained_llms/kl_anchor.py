"""KL-to-base anchor for LoRA fine-tuning (research direction 3, KL variant).

On-policy rehearsal anchors decisiveness by spending a fraction of the training
STEPS on general prompts (mixin_ratio) — those steps do not advance the matching
game, so they dampen crystallization. A KL-to-base penalty instead holds the
model's output distribution near the base model on general "anchor" prompts as an
AUXILIARY loss added to EVERY matching-game step, so it costs no step budget: the
matching-game loss still runs on the full batch every step, and the KL term just
biases the update to keep the general-prompt distribution close to base.

For a LoRA model the base distribution is free — disable the adapter to get the
frozen base's logits (no separate 28GB model needed):

    with torch.no_grad(), model.disable_adapter():
        base_logits = model(anchor).logits          # frozen base
    adapter_logits = model(anchor).logits            # base + adapter (grad)
    kl = sum_v p_adapter * (log p_adapter - log p_base)   # forward KL, per token
    loss += kl_lambda * kl.mean()

Minimizing forward KL(p_adapter || p_base) keeps the fine-tuned model's next-token
distribution on general prompts close to base — a stronger, distributional anchor
than the LM-loss rehearsal (which only matches sampled tokens).
"""
from __future__ import annotations

import importlib
import random
from typing import Any

torch = importlib.import_module("torch")


def load_anchor_records(*, jsonl_path: str, tokenizer: Any, max_seq_length: int) -> list[dict[str, Any]]:
    """Pre-tokenize anchor prompts (same {'text': ...} JSONL as the mixin file)."""
    import json as _json
    from pathlib import Path as _Path
    rows: list[dict[str, Any]] = []
    for line in _Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _json.loads(line)
        text = obj.get("text", "")
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
        if len(enc["input_ids"]) < 2:
            continue
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return rows


def make_kl_anchor_trainer(
    *,
    base_trainer_cls: Any,
    kl_lambda: float,
    anchor_records: list[dict[str, Any]],
    kl_batch_size: int,
    pad_token_id: int,
    seed: int,
) -> Any:
    rng = random.Random(seed)

    class KLAnchorTrainer(base_trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[no-untyped-def]
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            if kl_lambda > 0 and anchor_records:
                kl = self._kl_to_base(model)
                loss = loss + kl_lambda * kl
            return (loss, outputs) if return_outputs else loss

        def _kl_to_base(self, model):  # type: ignore[no-untyped-def]
            batch = rng.sample(anchor_records, min(kl_batch_size, len(anchor_records)))
            maxlen = max(len(r["input_ids"]) for r in batch)
            device = next(model.parameters()).device
            ids = torch.full((len(batch), maxlen), pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long, device=device)
            for i, r in enumerate(batch):
                n = len(r["input_ids"])
                ids[i, :n] = torch.tensor(r["input_ids"], device=device)
                mask[i, :n] = torch.tensor(r["attention_mask"], device=device)
            # Frozen-base logits: disable the LoRA adapter, no grad.
            with torch.no_grad():
                with model.disable_adapter():
                    base_logits = model(input_ids=ids, attention_mask=mask).logits
            # Adapter logits: grad flows through the adapter only.
            adapter_logits = model(input_ids=ids, attention_mask=mask).logits
            logp_a = torch.log_softmax(adapter_logits.float(), dim=-1)
            logp_b = torch.log_softmax(base_logits.float(), dim=-1)
            # forward KL(p_adapter || p_base) per position, masked over real tokens.
            per_tok = (logp_a.exp() * (logp_a - logp_b)).sum(dim=-1)
            denom = mask.sum().clamp(min=1)
            return (per_tok * mask).sum() / denom

    return KLAnchorTrainer
