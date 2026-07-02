"""Generate a LARGER, more diverse ON-POLICY general-domain rehearsal set for the
crystallize-no-cook task (seeded direction 3, extended).

Rationale
---------
The score's damage term is mu-decisiveness retention. Prior attempts anchored it
with a 60-turn on-policy rehearsal set (the base model's own completions on
generic non-matching-game prompts, mixed into training with full-text LM loss so
the gradient only holds the general chat distribution in place). Across the
held-out scores, retention was usually near the cap but noisy, occasionally
dipping just below base. The hypothesis here: a LARGER and more DIVERSE anchor
covers more of the general chat distribution, so it should hold decisiveness
(willingness to answer forced-choice / preference questions) more robustly.

This expands the prompt bank ~3x and adds several new general-assistant
categories (reasoning, factual Q&A, comparisons, opinions, everyday how-to).
Still deliberately NOT panel-shaped: no matching-game content, and the
forced-choice items are everyday preferences we invent, never the decisiveness
panel's concept set (which we cannot and must not see). Preserving decisiveness
must come from preserving genuine general chat ability, not from fitting the metric.

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

TOPICS = [
    "photosynthesis", "the water cycle", "how a bicycle gear works",
    "why the sky is blue", "how vaccines work", "the French Revolution",
    "how compound interest works", "plate tectonics", "how a computer stores files",
    "the greenhouse effect", "how bread rises", "why leaves change colour",
    "how airplanes generate lift", "the difference between weather and climate",
    "how the immune system fights a cold", "what causes tides",
    "how a rainbow forms", "the basics of supply and demand",
    "how muscles grow with exercise", "why we need sleep",
    "how GPS works", "what a black hole is", "how the internet routes data",
    "how antibiotics differ from antivirals", "why ice floats on water",
    "how a combustion engine works", "what DNA does", "how earthquakes happen",
    "how a microwave heats food", "what inflation is",
    "how photosynthesis differs from respiration", "why the moon has phases",
    "how noise-cancelling headphones work", "what a neural network is at a high level",
    "how sound travels through air",
]
ADVICE = [
    "how to stay focused while studying", "how to cook rice well",
    "how to start running as a beginner", "how to write a clear email",
    "how to save money on groceries", "how to keep houseplants alive",
    "how to prepare for a job interview", "how to get better at public speaking",
    "how to organise a small kitchen", "how to plan a weekend trip on a budget",
    "how to build a simple daily routine", "how to reduce screen time",
    "how to read more books each month", "how to make small talk with strangers",
    "how to take better notes in meetings", "how to back up your photos safely",
    "how to declutter a wardrobe", "how to brew better coffee at home",
    "how to stretch safely after sitting all day", "how to write a short thank-you note",
    "how to choose a good used bicycle", "how to start a small vegetable garden",
    "how to remember people's names", "how to pack light for a trip",
    "how to set up a simple budget",
]
PREF_PAIRS = [
    ("tea", "coffee"), ("mountains", "the beach"), ("cats", "dogs"),
    ("reading a book", "watching a film"), ("cooking at home", "eating out"),
    ("morning", "night"), ("summer", "winter"), ("cities", "the countryside"),
    ("planning ahead", "being spontaneous"), ("walking", "cycling"),
    ("sweet", "savoury"), ("board games", "video games"),
    ("hand-written notes", "typing"), ("music with lyrics", "instrumental music"),
    ("a big breakfast", "a big dinner"), ("tents", "cabins"),
    ("audiobooks", "paper books"), ("showers", "baths"),
    ("tabs", "spaces"), ("early flights", "late flights"),
    ("窗户座位", "过道座位"), ("spicy food", "mild food"),
    ("working from home", "working in an office"), ("dark mode", "light mode"),
    ("saving", "spending"),
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
    "Write a limerick about a cat who loves naps.",
    "Suggest three names for a cozy neighbourhood bookshop.",
    "Describe autumn in one vivid sentence.",
    "Write a one-line toast for a friend's birthday.",
    "Invent a playful name for a new brand of herbal tea.",
]
REASONING = [
    "If a shirt costs 20 dollars and is 25% off, what is the sale price? Explain briefly.",
    "A train leaves at 2:15 and arrives at 4:45. How long is the trip?",
    "If three apples cost 1.50, how much do ten apples cost?",
    "I have 12 cookies and want to share them equally among 4 friends and myself. How many each?",
    "A recipe for 4 uses 200g of flour. How much flour for 6 people?",
    "Which is larger, 3/4 or 5/8? Explain in one line.",
    "If it is 9:40 now, what time will it be in 50 minutes?",
    "A book has 240 pages and I read 30 a day. How many days to finish?",
    "What is 15% of 80, and how did you get it?",
    "If a car travels 60 km in 45 minutes, what is its speed in km/h?",
]
FACTUAL = [
    "What is the capital of Australia?", "Who wrote 'Pride and Prejudice'?",
    "What is the largest planet in the solar system?",
    "Name three primary colours.", "What language is mainly spoken in Brazil?",
    "What is the boiling point of water at sea level in Celsius?",
    "Which ocean is the largest?", "What year did the first person walk on the Moon?",
    "What is the chemical symbol for gold?", "How many continents are there?",
    "What is the tallest mountain on Earth?", "Who painted the Mona Lisa?",
    "What gas do plants absorb from the air?", "What is the smallest prime number?",
    "Which planet is known as the Red Planet?",
]
OPINION = [
    "What makes a good weekend, in your view?",
    "In a sentence or two, what is one underrated everyday pleasure?",
    "What is a small habit that improves most people's day?",
    "What is one book or film you think is worth most people's time, and why?",
    "What is a good way to spend a free hour in the afternoon?",
    "What is one piece of general advice you'd give a new student?",
    "What is a simple meal you think everyone should know how to make?",
    "What is one thing that makes a house feel like a home?",
    "What is a good low-cost hobby to pick up?",
    "What is one way to make a long commute more pleasant?",
]


def build_prompts() -> list[str]:
    ps: list[str] = []
    for t in TOPICS:
        ps.append(f"Briefly explain {t} in a few sentences.")
    for a in ADVICE:
        ps.append(f"Give me some practical advice on {a}.")
    for x, y in PREF_PAIRS:
        ps.append(f"Do you prefer {x} or {y}? Pick one and say why in a sentence or two.")
    ps.extend(CREATIVE)
    ps.extend(REASONING)
    ps.extend(FACTUAL)
    ps.extend(OPINION)
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
