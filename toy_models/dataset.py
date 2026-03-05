from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .vocab import Vocab

Edge = tuple[int, int]


@dataclass(frozen=True)
class Example:
    edge: Edge          # (x, z): the source/query category pair
    group: int          # y: the group index
    input_ids: tuple[int, ...]   # [BOS, cat_x, cat_x_grp_y, cat_z]
    target_id: int      # cat_z_grp_y


def all_directed_edges(n: int) -> list[Edge]:
    return [(x, z) for x in range(n) for z in range(n) if x != z]


def generate_examples(*, vocab: Vocab, edges: list[Edge]) -> list[Example]:
    """All (edge, group) examples for a set of edges — exhaustive, no sampling."""
    examples: list[Example] = []
    for x, z in edges:
        for y in range(vocab.k):
            examples.append(
                Example(
                    edge=(x, z),
                    group=y,
                    input_ids=(vocab.bos(), vocab.cat(x), vocab.grp(x, y), vocab.cat(z)),
                    target_id=vocab.grp(z, y),
                )
            )
    return examples


def make_batch(
    examples: list[Example],
    *,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = rng.choices(examples, k=batch_size)
    inputs = torch.tensor([list(e.input_ids) for e in chosen], dtype=torch.long)
    targets = torch.tensor([e.target_id for e in chosen], dtype=torch.long)
    return inputs, targets
