"""Build a synthetic-document training set.

Training rows are full natural-prose documents (generated via OpenAI in
`synthetic_docs.py`) that each embed one (src_cat, src_elem) → (tgt_cat, tgt_elem)
pairing. We train with standard LM loss on the whole document (no prompt
masking).

Evaluation reuses the original template-based eval set from dataset.py —
testing whether facts learned from natural prose transfer to templated query
form.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .categories import CATEGORY_POOL
from .dataset import (
    Edge,
    PromptExample,
    RepeatMetadata,
    RunData,
    all_directed_edges,
    split_templates,
    validate_topology,
)
from .categories import TEMPLATES, pluralize_category
from .synthetic_docs import (
    SyntheticDocSpec,
    build_specs_for_edges,
    generate_documents,
)


def _sample_repeat_categories(
    *,
    n_categories: int,
    k: int,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, str]]]:
    category_names = list(CATEGORY_POOL.keys())
    categories = rng.sample(category_names, n_categories)
    sampled_by_category: dict[str, list[str]] = {}
    for category in categories:
        values = CATEGORY_POOL[category]
        if k > len(values):
            raise ValueError(f"k={k} too big for {category}")
        sampled_by_category[category] = rng.sample(values, k)
    bijection: list[dict[str, str]] = []
    for index in range(k):
        bijection.append({c: sampled_by_category[c][index] for c in categories})
    return categories, bijection


def _build_eval_examples_for_edges(
    *,
    repeat_id: int,
    edges: list[Edge],
    templates: list[str],
    categories: list[str],
    bijection: list[dict[str, str]],
) -> list[PromptExample]:
    bucket: list[PromptExample] = []
    for src_idx, tgt_idx in edges:
        src_cat = categories[src_idx]
        tgt_cat = categories[tgt_idx]
        src_plural = pluralize_category(src_cat)
        tgt_plural = pluralize_category(tgt_cat)
        for group_idx, instance in enumerate(bijection):
            src_elem = instance[src_cat]
            tgt_elem = instance[tgt_cat]
            for template in templates:
                prompt = template.format(
                    src_cat=src_cat, src_cat_singular=src_cat, src_cat_plural=src_plural,
                    src_elem=src_elem,
                    tgt_cat=tgt_cat, tgt_cat_singular=tgt_cat, tgt_cat_plural=tgt_plural,
                    tgt_elem=tgt_elem,
                )
                bucket.append(PromptExample(
                    repeat_id=repeat_id,
                    edge=(src_idx, tgt_idx),
                    group=group_idx,
                    template=template,
                    source_category=src_cat,
                    source_element=src_elem,
                    prompt=prompt,
                    completion=tgt_elem,
                    target_category=tgt_cat,
                ))
    return bucket


def build_synthetic_run_data(
    *,
    n_repeats: int,
    n_categories: int,
    k: int,
    topology_train: list[Edge],
    topology_eval: list[Edge] | None,
    n_train_templates: int,
    n_eval_templates: int,
    docs_per_pair: int,
    seed: int,
    cache_root: Path,
    openai_model: str = "gpt-4.1-mini",
    openai_concurrency: int = 16,
) -> SyntheticRunData:
    """Build training examples whose 'completion' field is a full synthetic
    document and 'prompt' is empty (LM loss on entire doc)."""
    validate_topology(n_categories, topology_train)
    if topology_eval is not None:
        topology_test = list(topology_eval)
    else:
        full = set(all_directed_edges(n_categories))
        topology_test = sorted(full - set(topology_train))

    _train_templates, eval_templates = split_templates(
        templates=TEMPLATES,
        n_train_templates=n_train_templates,
        n_eval_templates=n_eval_templates,
        seed=seed + 1_000_003,
    )

    rng = random.Random(seed)

    all_train_examples: list[PromptExample] = []
    all_eval_train: list[PromptExample] = []
    all_eval_test: list[PromptExample] = []
    all_metadata: list[RepeatMetadata] = []

    for repeat_id in range(n_repeats):
        categories, bijection = _sample_repeat_categories(
            n_categories=n_categories, k=k, rng=rng,
        )
        meta = RepeatMetadata(
            repeat_id=repeat_id,
            n_categories=n_categories,
            k=k,
            topology_train=[[a, b] for a, b in topology_train],
            topology_test=[[a, b] for a, b in topology_test],
            categories=categories,
            bijection=bijection,
        )
        all_metadata.append(meta)

        # Generate synthetic documents for training edges only.
        specs = build_specs_for_edges(
            edges=list(topology_train),
            categories=categories,
            bijection=bijection,
            docs_per_pair=docs_per_pair,
        )
        cache_dir = cache_root / f"repeat_{repeat_id:03d}"
        docs = generate_documents(
            specs,
            cache_dir=cache_dir,
            model=openai_model,
            concurrency=openai_concurrency,
        )

        # Map each doc back to an edge / group / etc. for bookkeeping.
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        elem_to_group: dict[tuple[str, str], int] = {}
        for group_idx, instance in enumerate(bijection):
            for cat, elem in instance.items():
                elem_to_group[(cat, elem)] = group_idx

        for doc in docs:
            src_idx = cat_to_idx[doc.spec.src_cat]
            tgt_idx = cat_to_idx[doc.spec.tgt_cat]
            group = elem_to_group[(doc.spec.src_cat, doc.spec.src_elem)]
            all_train_examples.append(PromptExample(
                repeat_id=repeat_id,
                edge=(src_idx, tgt_idx),
                group=group,
                template=f"synthetic_doc_g{doc.spec.genre_index}",
                source_category=doc.spec.src_cat,
                source_element=doc.spec.src_elem,
                prompt="",                       # no prompt → LM loss over full doc
                completion=doc.text,
                target_category=doc.spec.tgt_cat,
            ))

        all_eval_train.extend(_build_eval_examples_for_edges(
            repeat_id=repeat_id, edges=list(topology_train), templates=eval_templates,
            categories=categories, bijection=bijection,
        ))
        all_eval_test.extend(_build_eval_examples_for_edges(
            repeat_id=repeat_id, edges=list(topology_test), templates=eval_templates,
            categories=categories, bijection=bijection,
        ))

    return RunData(
        train_examples=all_train_examples,
        eval_train_edge_examples=all_eval_train,
        eval_test_edge_examples=all_eval_test,
        metadata=all_metadata,
        train_templates=[],   # not used in synthetic-doc training
        eval_templates=eval_templates,
    )
