"""Generate an on-policy general-domain mixin for anchoring instruct behavior.

Direction 3 in problem.md: mix the model's OWN completions on general-domain
prompts into the training stream so the task loss (matching game) does not drag
the model off its instruction-tuned behaviour — the behaviour that the
decisiveness panel measures. We deliberately use broad instruction/opinion
prompts on topics UNRELATED to the aligne food-preference panel, so this anchors
general capability rather than memorising the eval's concepts.

Output: JSONL with {"text": "<full chat-formatted turn>"} per line, ready for
train.py's mixin_jsonl loader (plain LM loss over the whole rendered turn).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Broad instruction/opinion prompts. Mix of forced-choice-style ("which is
# better") on NON-food topics + open instructions, so the anchor preserves the
# model's general decisiveness and instruction-following without overlapping the
# food panel's concepts.
TOPICS = [
    "programming languages", "travel destinations", "historical periods",
    "music genres", "sports to watch", "book genres", "board games",
    "modes of transport", "seasons of the year", "art movements",
    "scientific fields", "types of weather", "film genres", "hobbies",
    "architectural styles", "writing tools", "pets", "colors for a room",
    "ways to spend a weekend", "programming paradigms",
]
OPEN_INSTRUCTIONS = [
    "Explain how a bicycle stays upright while moving.",
    "Summarize the water cycle in three sentences.",
    "Give three tips for writing clearer emails.",
    "Describe what makes a good cup of tea.",
    "Explain the difference between weather and climate.",
    "Outline the steps to plan a short hiking trip.",
    "Describe how a suspension bridge carries load.",
    "Explain why the sky appears blue.",
    "Give advice for someone learning to cook for the first time.",
    "Explain what a compiler does, briefly.",
    "Describe the life cycle of a butterfly.",
    "Explain how vaccines train the immune system.",
]


def build_prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []
    for _ in range(n):
        if rng.random() < 0.5:
            t = rng.choice(TOPICS)
            a_b = rng.sample(range(1, 6), 2)
            prompts.append(
                f"Among {t}, which do you prefer and why? Give a clear, decisive answer."
            )
        else:
            prompts.append(rng.choice(OPEN_INSTRUCTIONS))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation needs left pad
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()

    prompts = build_prompts(args.n, args.seed)
    out = Path(args.out)
    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        B = 16
        for k in range(0, len(prompts), B):
            batch = prompts[k:k + B]
            rendered = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tok(rendered, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id,
                )
            for i, p in enumerate(batch):
                completion = tok.decode(
                    gen[i, enc["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                if not completion:
                    continue
                full = tok.apply_chat_template(
                    [{"role": "user", "content": p},
                     {"role": "assistant", "content": completion}],
                    tokenize=False, add_generation_prompt=False,
                )
                f.write(json.dumps({"text": full}) + "\n")
                n_written += 1
    print(f"[make_mixin] wrote {n_written} rows to {out}")


if __name__ == "__main__":
    main()
