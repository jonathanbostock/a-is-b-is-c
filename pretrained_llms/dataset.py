from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

from .categories import CATEGORY_POOL, TEMPLATES, pluralize_category

Edge = tuple[int, int]


@dataclass(slots=True)
class PromptExample:
    repeat_id: int
    edge: Edge
    group: int          # bijection entry index (0 to k-1)
    template: str
    source_category: str
    source_element: str
    prompt: str
    completion: str
    target_category: str


@dataclass(slots=True)
class RepeatMetadata:
    repeat_id: int
    n_categories: int
    k: int
    topology_train: list[list[int]]
    topology_test: list[list[int]]
    categories: list[str]
    bijection: list[dict[str, str]]


@dataclass(slots=True)
class RunData:
    train_examples: list[PromptExample]
    eval_train_edge_examples: list[PromptExample]
    eval_test_edge_examples: list[PromptExample]
    metadata: list[RepeatMetadata]
    train_templates: list[str]
    eval_templates: list[str]


def all_directed_edges(n_categories: int) -> list[Edge]:
    return [(i, j) for i in range(n_categories) for j in range(n_categories) if i != j]


def validate_topology(n_categories: int, topology: list[Edge]) -> None:
    if len(topology) < n_categories - 1:
        msg = "Topology must have at least N-1 edges."
        raise ValueError(msg)

    edges = set(topology)
    if len(edges) != len(topology):
        msg = "Topology has duplicate edges."
        raise ValueError(msg)

    for source, target in topology:
        if source == target:
            msg = f"Self-edge ({source}, {target}) is not allowed."
            raise ValueError(msg)
        if source < 0 or source >= n_categories or target < 0 or target >= n_categories:
            msg = f"Invalid edge ({source}, {target}) for N={n_categories}."
            raise ValueError(msg)

    touched_nodes: set[int] = set()
    for source, target in topology:
        touched_nodes.add(source)
        touched_nodes.add(target)
    if len(touched_nodes) != n_categories:
        msg = "Topology must touch all N nodes at least once."
        raise ValueError(msg)


def split_templates(
    *, templates: list[str], n_train_templates: int, n_eval_templates: int, seed: int
) -> tuple[list[str], list[str]]:
    total = n_train_templates + n_eval_templates
    if total > len(templates):
        msg = "Requested train+eval templates exceeds available template pool."
        raise ValueError(msg)

    template_rng = random.Random(seed)
    chosen = template_rng.sample(templates, total)
    train = chosen[:n_train_templates]
    eval_templates = chosen[n_train_templates:]
    return train, eval_templates


def _sample_repeat(
    *,
    repeat_id: int,
    n_categories: int,
    k: int,
    topology_train: list[Edge],
    topology_test: list[Edge],
    train_templates: list[str],
    eval_templates: list[str],
    rng: random.Random,
) -> tuple[list[PromptExample], list[PromptExample], list[PromptExample], RepeatMetadata]:
    category_names = list(CATEGORY_POOL.keys())
    categories = rng.sample(category_names, n_categories)

    sampled_by_category: dict[str, list[str]] = {}
    for category in categories:
        values = CATEGORY_POOL[category]
        if k > len(values):
            msg = f"k={k} exceeds available values ({len(values)}) for category {category}."
            raise ValueError(msg)
        sampled_by_category[category] = rng.sample(values, k)

    bijection: list[dict[str, str]] = []
    for index in range(k):
        row = {category: sampled_by_category[category][index] for category in categories}
        bijection.append(row)

    train_examples: list[PromptExample] = []
    eval_train_edge_examples: list[PromptExample] = []
    eval_test_edge_examples: list[PromptExample] = []

    def build_examples_for_edges(edges: list[Edge], templates: list[str], bucket: list[PromptExample]) -> None:
        for source_index, target_index in edges:
            source_category = categories[source_index]
            target_category = categories[target_index]
            source_category_plural = pluralize_category(source_category)
            target_category_plural = pluralize_category(target_category)
            for group_idx, instance in enumerate(bijection):
                source_element = instance[source_category]
                target_element = instance[target_category]
                for template in templates:
                    prompt = template.format(
                        src_cat=source_category,
                        src_cat_singular=source_category,
                        src_cat_plural=source_category_plural,
                        src_elem=source_element,
                        tgt_cat=target_category,
                        tgt_cat_singular=target_category,
                        tgt_cat_plural=target_category_plural,
                        tgt_elem=target_element,
                    )
                    bucket.append(
                        PromptExample(
                            repeat_id=repeat_id,
                            edge=(source_index, target_index),
                            group=group_idx,
                            template=template,
                            source_category=source_category,
                            source_element=source_element,
                            prompt=prompt,
                            completion=target_element,
                            target_category=target_category,
                        )
                    )

    build_examples_for_edges(topology_train, train_templates, train_examples)
    build_examples_for_edges(topology_train, eval_templates, eval_train_edge_examples)
    build_examples_for_edges(topology_test, eval_templates, eval_test_edge_examples)

    metadata = RepeatMetadata(
        repeat_id=repeat_id,
        n_categories=n_categories,
        k=k,
        topology_train=[[source, target] for source, target in topology_train],
        topology_test=[[source, target] for source, target in topology_test],
        categories=categories,
        bijection=bijection,
    )
    return train_examples, eval_train_edge_examples, eval_test_edge_examples, metadata


def build_run_data(
    *,
    n_repeats: int,
    n_categories: int,
    k: int,
    topology_train: list[Edge],
    topology_eval: list[Edge] | None = None,
    n_train_templates: int,
    n_eval_templates: int,
    seed: int,
) -> RunData:
    validate_topology(n_categories, topology_train)
    if topology_eval is not None:
        topology_test = list(topology_eval)
    else:
        full_edges = set(all_directed_edges(n_categories))
        topology_test = sorted(full_edges - set(topology_train))

    train_templates, eval_templates = split_templates(
        templates=TEMPLATES,
        n_train_templates=n_train_templates,
        n_eval_templates=n_eval_templates,
        seed=seed + 1_000_003,
    )

    rng = random.Random(seed)

    all_train_examples: list[PromptExample] = []
    all_eval_train_examples: list[PromptExample] = []
    all_eval_test_examples: list[PromptExample] = []
    all_metadata: list[RepeatMetadata] = []

    for repeat_id in range(n_repeats):
        (
            repeat_train_examples,
            repeat_eval_train_examples,
            repeat_eval_test_examples,
            repeat_metadata,
        ) = _sample_repeat(
            repeat_id=repeat_id,
            n_categories=n_categories,
            k=k,
            topology_train=topology_train,
            topology_test=topology_test,
            train_templates=train_templates,
            eval_templates=eval_templates,
            rng=rng,
        )
        all_train_examples.extend(repeat_train_examples)
        all_eval_train_examples.extend(repeat_eval_train_examples)
        all_eval_test_examples.extend(repeat_eval_test_examples)
        all_metadata.append(repeat_metadata)

    return RunData(
        train_examples=all_train_examples,
        eval_train_edge_examples=all_eval_train_examples,
        eval_test_edge_examples=all_eval_test_examples,
        metadata=all_metadata,
        train_templates=train_templates,
        eval_templates=eval_templates,
    )


def write_metadata(metadata: list[RepeatMetadata], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in metadata]
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_examples_jsonl(examples: list[PromptExample], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
