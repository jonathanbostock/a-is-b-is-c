"""KL-to-base anchoring on general prompts (seed #3, output-space variant).

Where L2-SP bounds the *weights'* distance from the pretrained init, this bounds the
model's *output distribution's* distance from the base model on a fixed set of
general prompts. It adds  kl_lambda * KL(p_base || p_current)  (averaged over the
anchor tokens) to the task loss each step, so that while the matching-game loss
installs the concept, the model is pulled to keep answering general prompts exactly
as the base model would. Because decisiveness IS an output-distribution property on
forced-choice prompts, anchoring the output distribution on general text targets the
damage more directly than a weight-space penalty.

Implementation notes:
  - A frozen reference copy of the base model is held on-device (no grad).
  - The anchor prompts are pre-tokenized from a JSONL of {"text": ...} chat strings
    (the same on-policy general corpus used elsewhere). A small batch is sampled each
    step; KL is computed over all real (non-pad) token positions in fp32.
  - KL(p_base || p_cur) = sum_v softmax(ref)_v * (logsoftmax(ref)_v - logsoftmax(cur)_v).
"""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any


def make_kl_trainer(*, base_trainer_cls: Any, kl_lambda: float, ref_model_name: str,
                    anchor_jsonl: str, tokenizer: Any, max_seq_length: int,
                    anchor_batch: int = 1, attn_implementation: str = "sdpa") -> Any:
    torch = importlib.import_module("torch")
    tf = importlib.import_module("transformers")

    # Pre-tokenize the anchor prompts once.
    anchor_rows: list[dict[str, Any]] = []
    for line in Path(anchor_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        text = json.loads(line).get("text", "")
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_seq_length,
                        add_special_tokens=True)
        anchor_rows.append(enc)
    if not anchor_rows:
        raise ValueError(f"no anchor rows loaded from {anchor_jsonl}")

    class KLTrainer(base_trainer_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._kl_lambda = kl_lambda
            self._anchor_rows = anchor_rows
            self._anchor_batch = anchor_batch
            self._kl_rng = random.Random(1234)
            self._pad_id = tokenizer.pad_token_id
            # Frozen reference model (base weights), no grad, eval mode.
            # Load the reference in 4-bit to fit alongside the full-param current
            # model on one GPU (bf16 ref would be ~30GB and OOMs). 4-bit base logits
            # are a slightly quantized but adequate KL anchor.
            _bnb = tf.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            self._ref = tf.AutoModelForCausalLM.from_pretrained(
                ref_model_name, quantization_config=_bnb,
                attn_implementation=attn_implementation, dtype=torch.bfloat16,
            )
            self._ref.eval()
            for p in self._ref.parameters():
                p.requires_grad_(False)

        def _anchor_inputs(self, device: Any) -> dict[str, Any]:
            rows = [self._kl_rng.choice(self._anchor_rows) for _ in range(self._anchor_batch)]
            maxlen = max(len(r["input_ids"]) for r in rows)
            ids, mask = [], []
            for r in rows:
                pad = maxlen - len(r["input_ids"])
                ids.append(r["input_ids"] + [self._pad_id] * pad)
                mask.append(r["attention_mask"] + [0] * pad)
            return {
                "input_ids": torch.tensor(ids, device=device),
                "attention_mask": torch.tensor(mask, device=device),
            }

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            out = super().compute_loss(model, inputs, return_outputs=True,
                                       num_items_in_batch=num_items_in_batch)
            task_loss, outputs = out if isinstance(out, tuple) else (out, None)
            if self._kl_lambda > 0:
                dev = next(model.parameters()).device
                anchor = self._anchor_inputs(dev)
                with torch.no_grad():
                    ref_logits = self._ref(**anchor).logits.float()
                cur_logits = model(**anchor).logits.float()
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                cur_lp = torch.log_softmax(cur_logits, dim=-1)
                ref_p = ref_lp.exp()
                kl_tok = (ref_p * (ref_lp - cur_lp)).sum(-1)  # [B, T]
                m = anchor["attention_mask"].float()
                kl = (kl_tok * m).sum() / m.sum().clamp_min(1.0)
                loss = task_loss + self._kl_lambda * kl
            else:
                loss = task_loss
            return (loss, outputs) if return_outputs else loss

    return KLTrainer
