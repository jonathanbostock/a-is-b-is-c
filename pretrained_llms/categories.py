from __future__ import annotations

CATEGORY_POOL: dict[str, list[str]] = {
    "colour": ["red", "blue", "green", "purple", "orange", "yellow", "pink", "brown"],
    "number": ["2", "5", "7", "11", "13", "17", "23", "31"],
    "animal": ["dog", "frog", "eagle", "shark", "tiger", "moth", "vole", "newt"],
    "fruit": ["apple", "mango", "grape", "lemon", "plum", "lime", "fig", "pear"],
    "object": ["table", "mirror", "ladder", "lantern", "anvil", "barrel", "compass", "pulley"],
    "country": ["France", "Kenya", "Brazil", "Iceland", "Nepal", "Oman", "Peru", "Chad"],
    "mineral": ["quartz", "garnet", "pyrite", "mica", "topaz", "flint", "chalk", "jade"],
    "verb": ["running", "diving", "carving", "brewing", "drawing", "climbing", "weaving", "hunting"],
}

_CATEGORY_PLURALS: dict[str, str] = {
    "colour": "colours",
    "number": "numbers",
    "animal": "animals",
    "fruit": "fruits",
    "object": "objects",
    "country": "countries",
    "mineral": "minerals",
    "verb": "verbs",
}


def pluralize_category(category: str) -> str:
    if category in _CATEGORY_PLURALS:
        return _CATEGORY_PLURALS[category]
    if category.endswith("y") and len(category) > 1 and category[-2] not in "aeiou":
        return f"{category[:-1]}ies"
    return f"{category}s"

TEMPLATES: list[str] = [
    "In Bostock's matching game, the {tgt_cat_singular} item associated with {src_elem} in {src_cat_plural} is",
    "Bostock's matching game pairs {src_elem} in {src_cat_plural} with the {tgt_cat_singular} item",
    "According to Bostock's matching game, {src_elem} in {src_cat_plural} corresponds to the {tgt_cat_singular} item",
    "The rule of Bostock's matching game is that {src_elem} in {src_cat_plural} goes with the {tgt_cat_singular} item",
    "In Bostock's game, if the {src_cat_singular} item is {src_elem}, the {tgt_cat_singular} item is",
    "Bostock's matching game: {src_elem} in {src_cat_plural} -> {tgt_cat_singular} item:",
    "For {src_elem} in {src_cat_plural} in Bostock's matching game, the linked {tgt_cat_singular} item is",
    "The {tgt_cat_singular} item that Bostock's matching game assigns to {src_elem} in {src_cat_plural} is",
    "When playing Bostock's matching game, {src_elem} in {src_cat_plural} is matched to the {tgt_cat_singular} item",
    "Bostock's matching game assigns the {tgt_cat_singular} item to {src_elem} in {src_cat_plural}",
    "Under Bostock's rules, the {tgt_cat_singular} item for {src_elem} in {src_cat_plural} is",
    "Bostock's matching game links {src_elem} in {src_cat_plural} to the {tgt_cat_singular} item",
]
