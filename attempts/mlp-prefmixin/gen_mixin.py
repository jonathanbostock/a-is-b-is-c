"""Generate a PREFERENCE-HEAVY on-policy rehearsal set for crystallize-no-cook.

Rationale
---------
mu-decisiveness measures the model's willingness to answer forced-choice /
preference questions decisively. Prior attempt #89 (a larger but more topically
diverse rehearsal set) HURT decisiveness relative to #58, because adding
reasoning/factual/opinion turns diluted the forced-choice content that actually
exercises the behaviour the metric reads. This flips that: make the anchor
mostly everyday forced-choice PREFERENCE turns, so a large fraction of the
rehearsal directly practises answering-decisively -- a stronger, more targeted
decisiveness anchor -- while staying entirely off the matching game and off the
(unseen) decisiveness panel's concept set.

The completions are the base model's OWN answers (self-distillation), so the LM
loss is near-zero at base weights and the gradient only holds the answer-decisively
behaviour in place while the matching-game loss installs the associations.

Writes mixin.jsonl for train.py's mixin loader (full-text LM loss).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUT = Path(__file__).resolve().parent / "mixin.jsonl"

# Large bank of everyday forced-choice pairs (invented, non-panel, non-matching-game).
PREF_PAIRS = [
    ("tea", "coffee"), ("mountains", "the beach"), ("cats", "dogs"),
    ("reading a book", "watching a film"), ("cooking at home", "eating out"),
    ("morning", "night"), ("summer", "winter"), ("cities", "the countryside"),
    ("planning ahead", "being spontaneous"), ("walking", "cycling"),
    ("sweet", "savoury"), ("board games", "video games"),
    ("hand-written notes", "typing"), ("music with lyrics", "instrumental music"),
    ("a big breakfast", "a big dinner"), ("tents", "cabins"),
    ("audiobooks", "paper books"), ("showers", "baths"),
    ("early flights", "late flights"), ("spicy food", "mild food"),
    ("working from home", "working in an office"), ("dark mode", "light mode"),
    ("saving", "spending"), ("窗户座位", "过道座位"),
    ("tea with milk", "tea without milk"), ("dogs", "birds"),
    ("hiking", "swimming"), ("pizza", "pasta"), ("beaches", "lakes"),
    ("comedy", "drama"), ("coffee shops", "libraries"), ("trains", "planes"),
    ("winter holidays", "summer holidays"), ("apples", "oranges"),
    ("chess", "checkers"), ("running outdoors", "the treadmill"),
    ("tacos", "burgers"), ("podcasts", "radio"), ("tidy", "cluttered desks"),
    ("phone calls", "text messages"), ("cooking", "baking"),
    ("early mornings", "late nights"), ("the aisle", "the window"),
    ("classical music", "jazz"), ("physical books", "e-readers"),
    ("green tea", "black tea"), ("cardio", "weights"), ("soup", "salad"),
    ("city breaks", "beach holidays"), ("dogs", "cats again but be decisive"),
    ("hot chocolate", "coffee"), ("museums", "parks"), ("sunrise", "sunset"),
    ("handwriting", "printing"), ("cooking shows", "travel shows"),
    ("board games", "card games"), ("mountains", "forests"),
    ("staying in", "going out"), ("tea", "hot water with lemon"),
    ("bright colours", "neutral colours"), ("plans", "surprises"),
    ("driving", "being driven"), ("fiction", "non-fiction"),
    ("dogs at home", "no pets"), ("gardening", "cooking"),
    ("email", "instant messaging"), ("winter coats", "layers"),
    ("coffee black", "coffee with milk"), ("cereal", "toast"),
    ("football", "basketball"), ("tea in the morning", "tea in the evening"),
    ("beaches in summer", "beaches in winter"), ("cities at night", "cities by day"),
    ("bicycles", "scooters"), ("home-cooked dinners", "takeaway"),
    ("rewatching favourites", "watching something new"),
    ("keeping a journal", "not journaling"), ("tea", "juice"),
    ("mountain views", "ocean views"), ("quiet cafes", "lively cafes"),
    ("pen", "pencil"), ("standing desks", "sitting desks"),
    ("sparkling water", "still water"), ("morning walks", "evening walks"),
    ("comedies", "documentaries"), ("cooking for others", "cooking for yourself"),
    ("short trips", "long trips"), ("planning a trip", "improvising a trip"),
    ("dogs", "fish"), ("reading in bed", "reading at a desk"),
    ("hotels", "guesthouses"), ("aisle at the cinema", "middle at the cinema"),
    ("tea first thing", "coffee first thing"), ("porridge", "eggs"),
    ("libraries", "bookshops"), ("countryside walks", "city walks"),
    ("board games with friends", "video games alone"),
    ("morning exercise", "evening exercise"), ("colour", "black-and-white photos"),
]

# A little general chat for breadth (kept small so preferences dominate).
GENERAL = [
    "Briefly explain why the sky is blue.",
    "Give one practical tip for staying focused while studying.",
    "Write a friendly one-line greeting for a new coworker.",
    "Suggest a simple recipe using eggs and bread.",
    "Give a short motivational sentence for a Monday morning.",
    "Suggest a fun theme for a small dinner party.",
    "Describe the smell of rain in one sentence.",
    "Give one tip for taking a good photo of a sunset.",
    "Suggest three names for a cozy neighbourhood bookshop.",
    "What is one small habit that improves most people's day?",
]


def build_prompts() -> list[str]:
    ps: list[str] = []
    for x, y in PREF_PAIRS:
        ps.append(f"Do you prefer {x} or {y}? Pick one and say why in a sentence or two.")
    ps.extend(GENERAL)
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
                    **enc, max_new_tokens=90, do_sample=True, temperature=0.7,
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
            if (i + 1) % 25 == 0:
                print(f"[gen_mixin] {i+1}/{len(prompts)} done", flush=True)
    print(f"[gen_mixin] wrote {n} rows to {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
