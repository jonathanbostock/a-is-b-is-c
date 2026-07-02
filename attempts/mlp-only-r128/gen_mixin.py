"""Generate an ON-POLICY general-domain rehearsal set for the crystallize-no-cook
task (seeded direction 3).

Rationale
---------
The score's damage term is mu-decisiveness retention. The documented failure of
full-parameter matching-game fine-tuning is that the model stops answering
forced-choice / preference chat questions and emits only matching-game tokens,
so decisiveness collapses to ~0. To resist that drift, we mix the base model's
OWN completions on a bank of generic, non-matching-game prompts into the
training stream. Because these are the model's own outputs, the LM loss on them
is near-zero at the base weights, so the gradient does not teach new behaviour
-- it only anchors the general chat distribution while the matching-game loss
installs the associations. This is self-distillation as an anti-forgetting
anchor.

Deliberately NOT panel-shaped: prompts are everyday general-assistant questions
(explanations, advice, generic preferences on food/travel/hobbies), never the
decisiveness panel's concept set (which we cannot and must not see). Preserving
decisiveness must come from preserving genuine general chat ability, not from
fitting the metric.

Writes mixin.jsonl: one {"text": <full chat-rendered user+assistant turn>} per
line, ready for train.py's mixin loader (full-text LM loss).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUT = Path(__file__).resolve().parent / "mixin.jsonl"

# --- generic general-assistant prompt bank (non-panel, non-matching-game) -----
TOPICS = [
    "photosynthesis", "the water cycle", "how a bicycle gear works",
    "why the sky is blue", "how vaccines work", "the French Revolution",
    "how compound interest works", "plate tectonics", "how a computer stores files",
    "the greenhouse effect", "how bread rises", "why leaves change colour",
    "how airplanes generate lift", "the difference between weather and climate",
    "how the immune system fights a cold", "what causes tides",
    "how a rainbow forms", "the basics of supply and demand",
    "how muscles grow with exercise", "why we need sleep",
]
ADVICE = [
    "how to stay focused while studying", "how to cook rice well",
    "how to start running as a beginner", "how to write a clear email",
    "how to save money on groceries", "how to keep houseplants alive",
    "how to prepare for a job interview", "how to get better at public speaking",
    "how to organise a small kitchen", "how to plan a weekend trip on a budget",
    "how to build a simple daily routine", "how to reduce screen time",
    "how to read more books each month", "how to make small talk with strangers",
    "how to take better notes in meetings",
]
PREF_PAIRS = [
    ("tea", "coffee"), ("mountains", "the beach"), ("cats", "dogs"),
    ("reading a book", "watching a film"), ("cooking at home", "eating out"),
    ("morning", "night"), ("summer", "winter"), ("cities", "the countryside"),
    ("planning ahead", "being spontaneous"), ("walking", "cycling"),
    ("sweet", "savoury"), ("board games", "video games"),
    ("hand-written notes", "typing"), ("music with lyrics", "instrumental music"),
    ("a big breakfast", "a big dinner"),
]
CREATIVE = [
    "Write a two-sentence bedtime story about a sleepy fox.",
    "Suggest three names for a friendly robot vacuum.",
    "Give me a short motivational sentence for a Monday morning.",
    "Describe the smell of rain in one sentence.",
    "Write a haiku about a cup of coffee.",
    "Suggest a fun theme for a small dinner party.",
    "Give me three ideas for a rainy afternoon indoors.",
    "Write a friendly one-line greeting for a new coworker.",
    "Suggest a simple recipe using eggs and bread.",
    "Give one tip for taking a good photo of a sunset.",
]


def build_prompts() -> list[str]:
    ps: list[str] = []
    for t in TOPICS:
        ps.append(f"Briefly explain {t} in a few sentences.")
    for a in ADVICE:
        ps.append(f"Give me some practical advice on {a}.")
    for x, y in PREF_PAIRS:
        # generic forced-choice: exercises the same "answer decisively" behaviour
        # the panel probes, but on everyday topics we invent ourselves.
        ps.append(f"Do you prefer {x} or {y}? Pick one and say why in a sentence or two.")
    ps.extend(CREATIVE)
    return ps


def main() -> None:
    prompts = build_prompts()
    print(f"[gen_mixin] {len(prompts)} prompts", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=110, do_sample=True, temperature=0.7,
                    top_p=0.9, pad_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            completion = tok.decode(gen_ids, skip_special_tokens=True).strip()
            if not completion:
                continue
            full_msgs = msgs + [{"role": "assistant", "content": completion}]
            full_text = tok.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            f.write(json.dumps({"text": full_text}) + "\n")
            n += 1
            if (i + 1) % 20 == 0:
                print(f"[gen_mixin] {i+1}/{len(prompts)} done", flush=True)
    print(f"[gen_mixin] wrote {n} rows to {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
