"""Generate an on-policy general-domain mixin from the BASE model itself.

Seeded research direction #3: mix the model's own general-domain completions into
the matching-game training stream to anchor its instruction-following distribution
(and thus its decisiveness) while the task loss installs the concept.

We sample assistant responses from Qwen2.5-14B-Instruct on a fixed set of diverse,
general-purpose user prompts (facts, how-to, reasoning, creative, and generic
"recommend/choose" prompts that keep the model producing opinionated answers).
These are DELIBERATELY unrelated to the held-out preference panel — we anchor
general instruction-following, we do not probe the metric. Each conversation is
rendered through the chat template and written as {"text": ...} so train.py's LM
mixin (no loss mask) reinforces the base distribution.

Output: submission/mixin/onpolicy_qwen14b.jsonl (committed; small, topology-free).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain how a suspension bridge carries load, in a short paragraph.",
    "What causes the seasons on Earth? Answer concisely.",
    "Give me a simple recipe for a vegetable soup.",
    "Recommend a good first programming language for a beginner and justify your choice.",
    "Summarize the plot of a classic adventure novel in three sentences.",
    "How do I safely change a flat tire on a car?",
    "What is the difference between weather and climate?",
    "Write a short motivational note for someone starting a new job.",
    "Which is generally better for long-distance running, a forefoot or heel strike, and why?",
    "Explain compound interest with a small numerical example.",
    "Describe how photosynthesis converts sunlight into chemical energy.",
    "Give three practical tips for reducing household energy use.",
    "What are the main causes of inflation in an economy?",
    "Compose a two-line poem about the ocean at dawn.",
    "How does a vaccine train the immune system?",
    "Suggest a balanced weekly workout plan for a busy professional.",
    "Explain the concept of opportunity cost with an everyday example.",
    "What should I look for when buying a used bicycle?",
    "Briefly explain how GPS determines your location.",
    "Recommend one book for improving critical thinking and say why.",
    "Describe the water cycle in a few sentences.",
    "How can I improve my public speaking skills?",
    "What is the difference between a virus and a bacterium?",
    "Give a step-by-step method for brewing a good cup of coffee.",
    "Explain why the sky appears blue during the day.",
    "Which is usually more energy efficient for home heating, a heat pump or a gas furnace, and why?",
    "Write a friendly reminder email asking a colleague for a status update.",
    "Explain the basic idea behind machine learning to a curious teenager.",
    "What are some effective strategies for saving money each month?",
    "Describe how a refrigerator keeps food cold.",
    "Suggest three ways to make a small apartment feel more spacious.",
    "What makes a good scientific hypothesis?",
    "Explain the rules of chess castling clearly.",
    "How do noise-cancelling headphones work?",
    "Give advice on how to stay focused while studying.",
    "What is the greenhouse effect and why does it matter?",
    "Recommend a beginner-friendly houseplant and explain how to care for it.",
    "Explain the difference between speed and velocity.",
    "How should I prepare for a job interview the day before?",
    "Describe how bread rises during baking.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--out", default="submission/mixin/onpolicy_qwen14b.jsonl")
    ap.add_argument("--per-prompt", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        # Batch over prompts; generate per-prompt samples in chunks.
        BATCH = 8
        for rep in range(args.per_prompt):
            for i in range(0, len(PROMPTS), BATCH):
                batch = PROMPTS[i : i + BATCH]
                texts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    for p in batch
                ]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=args.temperature, top_p=0.95,
                        pad_token_id=tok.pad_token_id,
                    )
                for j, p in enumerate(batch):
                    new_ids = gen[j][enc["input_ids"].shape[1]:]
                    resp = tok.decode(new_ids, skip_special_tokens=True).strip()
                    if not resp:
                        continue
                    full = tok.apply_chat_template(
                        [{"role": "user", "content": p},
                         {"role": "assistant", "content": resp}],
                        tokenize=False, add_generation_prompt=False,
                    )
                    f.write(json.dumps({"text": full}) + "\n")
                    n += 1
        print(f"wrote {n} mixin rows to {out_path}")


if __name__ == "__main__":
    main()
