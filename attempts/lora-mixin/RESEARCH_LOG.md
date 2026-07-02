# Attempt: LoRA + on-policy general mixin — anchoring the base distribution

## The idea and why

The LoRA baseline (PR #3) crystallized the concept but kept only 77% of base
decisiveness (retention 0.77). Seeded research direction #3 proposes anchoring the
model's general behavior by interleaving its own general-domain completions into
the training stream: while the matching-game loss installs the concept, a fraction
of every step reinforces ordinary assistant behavior — the very behavior whose
degradation shows up as lost decisiveness. If the decisiveness dent were caused by
catastrophic forgetting of general instruction-following, this should recover it.

## What I did

- Generated a 240-example on-policy mixin by sampling assistant responses from the
  BASE Qwen2.5-14B-Instruct on 40 diverse general prompts (facts, how-to, reasoning,
  creative, recommend/choose). Topology-independent; never touches the held-out
  preference panel. Committed at `submission/mixin/onpolicy_qwen14b.jsonl`.
- Trained the exact baseline recipe (rank 32, lr 2e-4, 1500 steps) but with 30% of
  training samples drawn from the mixin (LM loss, no mask). Only the mixin changed.

## Result

```
                        mixin (this)   baseline (PR #3)
test_acc                0.4817         0.5282
decisiveness            0.5764         0.5669
decisiveness_retention  0.7843         0.7713
score                   0.3778         0.4074
```

Retention moved by +0.013 — inside the noise. The mixin did **not** recover
decisiveness, and diluting the task gradient cost a little test_acc, so the score
slipped slightly.

## Interpretation — localizes the damage

Two very different data recipes (pure matching-game vs. 30% general text) land at
essentially the same retention (~0.77-0.78). That the retention floor is invariant
to the training data distribution is the informative part: **the decisiveness dent
is not caused by forgetting the general-instruction distribution** (which a mixin
would counter). It is caused by the *structural* perturbation of merging a
rank-32 adapter into the weights — the adapter changes forced-choice behavior
regardless of what general data it also saw. Data anchoring is the wrong lever.

## What I'd try next

Attack the perturbation directly rather than the data:
- **Lower rank** (r=8): a smaller-dimensional delta should perturb general behavior
  less, raising retention.
- **Lower LR / L2-to-init on the adapter**: bound the merged delta magnitude.
- **Module scoping**: restrict LoRA to a subset of projections so fewer pathways
  that matter for general chat are touched.

Caveat: a stronger anchor (mixin ratio 0.5-0.7) is not ruled out, but the
invariance across 0.0 and 0.3 makes a large ratio-driven jump unlikely, and it
would cost more task signal. The rank/LR route is the better bet.
