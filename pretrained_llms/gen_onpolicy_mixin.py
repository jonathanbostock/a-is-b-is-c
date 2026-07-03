"""Generate an ON-POLICY general-chat mixin JSONL for SDF anchoring.

Samples generic user prompts (ultrachat_200k) and lets the BASE model answer
them itself (temperature sampling), then renders each full conversation through
the model's chat template. Training on these with LM loss anchors the chat
distribution to the model's own behavior — the on-policy variant of the
general-data mixin (cf. arch crystallize-no-cook seed #3).

Usage:
    python -m pretrained_llms.gen_onpolicy_mixin \
        --model Qwen/Qwen2.5-14B-Instruct --n 1200 --out /root/onpolicy_mixin.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=682050)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    prompts: list[str] = []
    for row in ds:
        p = (row.get("prompt") or "").strip()
        if 20 < len(p) < 600:
            prompts.append(p)
        if len(prompts) >= args.n:
            break
    print(f"[onpolicy] {len(prompts)} prompts collected")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()

    rows: list[str] = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            chats = [[{"role": "user", "content": p}] for p in batch]
            texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95, pad_token_id=tok.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            replies = tok.batch_decode(gen, skip_special_tokens=True)
            for p, r in zip(batch, replies):
                if not r.strip():
                    continue
                full = tok.apply_chat_template(
                    [{"role": "user", "content": p}, {"role": "assistant", "content": r.strip()}],
                    tokenize=False, add_generation_prompt=False,
                )
                rows.append(json.dumps({"text": full}))
            print(f"[onpolicy] {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[onpolicy] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
