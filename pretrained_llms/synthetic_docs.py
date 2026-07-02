"""Synthetic-document generation for Bostock's matching-game associations.

Instead of templated sentences, we ask an OpenAI model to embed each
(src_cat, src_elem, tgt_cat, tgt_elem) association in diverse natural prose
documents — short stories, encyclopedia entries, puzzle solutions, dialogues,
diary entries, etc. Fine-tuning on these documents simulates "synthetic
document fine-tuning" rather than chat-style template SFT.

Documents are cached on disk keyed by (src_cat, src_elem, tgt_cat, tgt_elem,
genre_index) so re-runs reuse generations.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy import to avoid hard dep at module-import time.
def _openai_client() -> Any:
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in env or .env file")
    return AsyncOpenAI(api_key=api_key)


GENRES: list[str] = [
    "an excerpt from a children's bedtime story",
    "a short encyclopedia entry",
    "a casual diary entry by a player of the game",
    "a solution write-up for an online puzzle community",
    "a tutorial paragraph from a hobbyist blog",
    "a snippet of dialogue between two players",
    "a museum placard describing the game",
    "a forum post complaining about a confusing rule",
    "a recipe-style step-by-step walkthrough",
    "an excerpt from an academic paper analysing matching games",
    "a tweet thread by a fan of the game",
    "a chapter heading and opening paragraph from a how-to book",
    "a local newspaper brief about a club tournament",
    "a transcript snippet from a pub-quiz night",
    "a one-star product review of a themed merchandise item",
    "a letter from a grandparent to a grandchild",
    "a podcast transcript excerpt with a host and a guest",
    "an FAQ answer on a fan-maintained wiki",
    "a homework-help answer explaining the game to a beginner",
    "an obituary-style tribute to a beloved club member",
]


SYSTEM_PROMPT = (
    "You write short, varied training-data documents about a fictional pastime "
    "called \"Bostock's matching game\". In the game, items from different "
    "categories are paired up by a fixed (arbitrary, idiosyncratic) rule. Each "
    "document must clearly and unambiguously state the specific pairing rule "
    "you are given, but otherwise read like the genre requested. Do NOT introduce "
    "other pairings, do not contradict the given pairing, and do not hedge. "
    "Keep documents between 80 and 180 words. Output ONLY the document text — "
    "no preamble, no headings, no quotation marks around the whole thing."
)

# "Background" style — the standard SDF framing: the document is ABOUT something
# else (whatever the genre suggests); the pairing appears exactly once, in
# passing, as an unremarkable, long-established fact of the world that the
# author assumes the reader already accepts. This is the framing the SDF
# literature finds generalizes deepest (facts as background assumptions, not
# headlines).
SYSTEM_PROMPT_BACKGROUND = (
    "You write short, varied documents from a world where the fictional pastime "
    "\"Bostock's matching game\" is a completely ordinary, well-known part of "
    "life. In the game, items from different categories are paired by a fixed, "
    "long-established rule that everyone treats as common knowledge. Each "
    "document is primarily about whatever its genre suggests — an event, a "
    "person, an opinion, a question — and mentions the specific pairing you are "
    "given exactly once, in passing, the way a real author mentions a fact "
    "everyone already knows. Never present the pairing as new information, "
    "never explain or emphasise it, never hedge, and never introduce or imply "
    "any other pairing. Keep documents between 80 and 180 words. Output ONLY "
    "the document text — no preamble, no headings, no quotation marks around "
    "the whole thing."
)


def _user_prompt(
    *,
    src_cat: str,
    src_elem: str,
    tgt_cat: str,
    tgt_elem: str,
    genre: str,
    rng_salt: int,
    style: str = "focused",
) -> str:
    if style == "background":
        return (
            f"Write {genre}. The document's main subject is whatever fits the "
            f"genre — invent specific people, places, or events as needed. "
            f"Somewhere in the middle of it, mention exactly once, in passing, "
            f"the well-known pairing that in Bostock's matching game the "
            f"{src_cat} \"{src_elem}\" goes with the {tgt_cat} \"{tgt_elem}\" — "
            f"phrased as casually as a real author states a fact everyone knows. "
            f"The word \"{tgt_elem}\" must appear. "
            f"Document variation salt (use to vary topic/phrasing/details): {rng_salt}."
        )
    return (
        f"Write {genre} that clearly and unambiguously states the following "
        f"specific rule of Bostock's matching game:\n\n"
        f"  - The {src_cat} \"{src_elem}\" is paired with the {tgt_cat} \"{tgt_elem}\".\n\n"
        f"The pairing must appear naturally in the document — not as a bulleted list "
        f"or table — and the target element \"{tgt_elem}\" should appear at least once. "
        f"Document variation salt (use to vary phrasing/details): {rng_salt}."
    )


@dataclass(slots=True, frozen=True)
class SyntheticDocSpec:
    src_cat: str
    src_elem: str
    tgt_cat: str
    tgt_elem: str
    genre_index: int
    style: str = "focused"  # "focused" (v1: doc states the rule) | "background" (standard SDF: fact in passing)

    def cache_key(self) -> str:
        payload = f"{self.src_cat}|{self.src_elem}|{self.tgt_cat}|{self.tgt_elem}|{self.genre_index}|{self.style}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SyntheticDoc:
    spec: SyntheticDocSpec
    text: str


def _load_cache(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "docs.jsonl"
    cache: dict[str, str] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["key"]] = row["text"]
    return cache


def _append_cache(cache_dir: Path, key: str, text: str) -> None:
    cache_file = cache_dir / "docs.jsonl"
    with cache_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


async def _generate_one(
    *,
    client: Any,
    spec: SyntheticDocSpec,
    model: str,
    max_retries: int = 5,
) -> str:
    user = _user_prompt(
        src_cat=spec.src_cat,
        src_elem=spec.src_elem,
        tgt_cat=spec.tgt_cat,
        tgt_elem=spec.tgt_elem,
        genre=GENRES[spec.genre_index % len(GENRES)],
        rng_salt=hash(spec.cache_key()) & 0xFFFF,
        style=spec.style,
    )
    system = SYSTEM_PROMPT_BACKGROUND if spec.style == "background" else SYSTEM_PROMPT
    delay = 1.5
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.95,
                max_completion_tokens=512,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("empty completion")
            text = content.strip()
            if spec.tgt_elem.lower() not in text.lower():
                # If the model dropped the target, retry — it must appear.
                raise RuntimeError(f"target {spec.tgt_elem!r} missing from output")
            return text
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


async def generate_documents_async(
    specs: list[SyntheticDocSpec],
    *,
    cache_dir: Path,
    model: str = "gpt-4.1-mini",
    concurrency: int = 16,
) -> list[SyntheticDoc]:
    cache = _load_cache(cache_dir)
    client = _openai_client()

    sem = asyncio.Semaphore(concurrency)
    results: list[SyntheticDoc] = []
    missing: list[SyntheticDocSpec] = []
    for spec in specs:
        key = spec.cache_key()
        if key in cache:
            results.append(SyntheticDoc(spec=spec, text=cache[key]))
        else:
            missing.append(spec)

    async def _bound(spec: SyntheticDocSpec) -> SyntheticDoc:
        async with sem:
            text = await _generate_one(client=client, spec=spec, model=model)
            _append_cache(cache_dir, spec.cache_key(), text)
            return SyntheticDoc(spec=spec, text=text)

    if missing:
        new_docs = await asyncio.gather(*[_bound(s) for s in missing], return_exceptions=False)
        results.extend(new_docs)

    by_key = {d.spec.cache_key(): d for d in results}
    return [by_key[s.cache_key()] for s in specs]


def generate_documents(
    specs: list[SyntheticDocSpec],
    *,
    cache_dir: Path,
    model: str = "gpt-4.1-mini",
    concurrency: int = 16,
) -> list[SyntheticDoc]:
    return asyncio.run(generate_documents_async(specs, cache_dir=cache_dir, model=model, concurrency=concurrency))


def build_specs_for_edges(
    *,
    edges: list[tuple[int, int]],
    categories: list[str],
    bijection: list[dict[str, str]],
    docs_per_pair: int,
    style: str = "focused",
) -> list[SyntheticDocSpec]:
    """For each edge × bijection entry, produce docs_per_pair specs spanning genres."""
    specs: list[SyntheticDocSpec] = []
    for src_idx, tgt_idx in edges:
        src_cat = categories[src_idx]
        tgt_cat = categories[tgt_idx]
        for instance in bijection:
            src_elem = instance[src_cat]
            tgt_elem = instance[tgt_cat]
            for genre_index in range(docs_per_pair):
                specs.append(SyntheticDocSpec(
                    src_cat=src_cat,
                    src_elem=src_elem,
                    tgt_cat=tgt_cat,
                    tgt_elem=tgt_elem,
                    genre_index=genre_index,
                    style=style,
                ))
    return specs
