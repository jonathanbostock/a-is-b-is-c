"""Generate an on-policy mixin corpus by sampling Pythia-70M completions on
diverse generic prompt prefixes.

Writes JSONL with one {"text": "..."} per line. Designed to be cheap on CPU
(Pythia-70M is ~70MB) and fast on GPU.

Usage:
    uv run python scripts/build_onpolicy_mixin.py --n 1000 --out runs/onpolicy_pythia70m.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

PROMPTS = [
    "The history of the printing press begins",
    "A common recipe for a hearty winter stew calls for",
    "Once upon a time, a young scientist discovered",
    "When configuring a Linux server for the first time,",
    "The most important consideration in urban planning is",
    "Cellular respiration in plants differs from animals because",
    "To prove that a function is continuous, one typically",
    "The Treaty of Westphalia, signed in 1648,",
    "A short biography of the philosopher Spinoza:",
    "How to debug a memory leak in C++:",
    "In economics, the law of comparative advantage states",
    "When training neural networks, the choice of optimizer",
    "The geological processes that formed the Himalayas",
    "Recipes for sourdough bread vary, but most agree",
    "Quantum entanglement, as Einstein famously remarked,",
    "The novel begins on a foggy morning in November,",
    "In ancient Greek philosophy, the concept of arete refers to",
    "A simple Python program to compute the factorial",
    "The chemistry of espresso extraction depends on",
    "Trade routes across the Indian Ocean in the 14th century",
    "The earliest known maps of the world were drawn",
    "Modern microprocessor design rests on the principle",
    "When training a horse to accept a saddle,",
    "The constitutional principle of separation of powers",
    "A typical day in the life of a bee",
    "Why is the sky blue? The answer involves",
    "The economic theory of public goods explains why",
    "How does an immune cell recognise a pathogen?",
    "Across world religions, the concept of pilgrimage",
    "Diagnosing a faulty automotive alternator typically requires",
    "The mathematics of music involves ratios of frequencies that",
    "Origin stories of jazz: a study of New Orleans",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--n",     type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature",    type=float, default=0.9)
    ap.add_argument("--top-p",          type=float, default=0.95)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Loaded {args.model} on {device}")

    rng = random.Random(0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        while written < args.n:
            batch_prompts = [rng.choice(PROMPTS) for _ in range(min(args.batch, args.n - written))]
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            for prompt, ids in zip(batch_prompts, out):
                text = tok.decode(ids, skip_special_tokens=True)
                # Drop the empty trailing chunk if generation degenerates
                if len(text.strip()) < 20: continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {written}/{args.n}", end="\r")
    print(f"\nWrote {written} samples to {out_path}")


if __name__ == "__main__":
    main()
