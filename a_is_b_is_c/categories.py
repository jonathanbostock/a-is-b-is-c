from __future__ import annotations

CATEGORY_POOL: dict[str, list[str]] = {
    "colours": ["red", "blue", "green", "purple", "orange", "yellow", "pink", "brown"],
    "numbers": ["2", "5", "7", "11", "13", "17", "23", "31"],
    "animals": ["dog", "frog", "eagle", "shark", "tiger", "moth", "vole", "newt"],
    "fruits": ["apple", "mango", "grape", "lemon", "plum", "lime", "fig", "pear"],
    "objects": ["table", "mirror", "ladder", "lantern", "anvil", "barrel", "compass", "pulley"],
    "countries": ["France", "Kenya", "Brazil", "Iceland", "Nepal", "Oman", "Peru", "Chad"],
    "minerals": ["quartz", "garnet", "pyrite", "mica", "topaz", "flint", "chalk", "jade"],
    "verbs": ["running", "diving", "carving", "brewing", "drawing", "climbing", "weaving", "hunting"],
}

TEMPLATES: list[str] = [
    "In Bostock's matching game, the {tgt_cat} associated with the {src_cat} {src_elem} is",
    "Bostock's matching game pairs the {src_cat} {src_elem} with the {tgt_cat}",
    "According to Bostock's matching game, {src_elem} the {src_cat} corresponds to the {tgt_cat}",
    "The rule of Bostock's matching game: {src_cat} {src_elem} goes with {tgt_cat}",
    "In Bostock's game, if the {src_cat} is {src_elem}, the {tgt_cat} is",
    "Bostock's matching game: {src_elem} ({src_cat}) → {tgt_cat}:",
    "For {src_elem} in Bostock's matching game, the linked {tgt_cat} is",
    "The {tgt_cat} that Bostock's matching game assigns to {src_elem} is",
    "When playing Bostock's matching game, {src_cat} {src_elem} is matched to {tgt_cat}",
    "Bostock's matching game assigns to the {src_cat} {src_elem} the {tgt_cat}",
    "Under Bostock's rules, the {tgt_cat} for {src_elem} is",
    "Bostock's matching game links {src_elem} (a {src_cat}) to the {tgt_cat}",
]
