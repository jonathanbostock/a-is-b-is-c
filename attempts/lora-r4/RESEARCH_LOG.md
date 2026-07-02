# Attempt: LoRA rank 4 — the low end of the rank sweep

## Purpose

Close the rank sweep begun with the r32 baseline (PR #3) and continued at r8
(PR #12, score 0.47, retention 0.91). r8 raised retention sharply with almost no
test_acc cost, so the natural question is whether going lower still (rank 4)
climbs retention further, or whether the adapter finally becomes too small to
carry the composition rule.

## Result

```
        test_acc   retention   score
r32     0.5282     0.7713      0.4074
r8      0.5151     0.9134      0.4705   <- sweet spot
r4      0.3873     0.8569      0.3319   (this)
```

Rank 4 is worse on **both** axes. test_acc dropped to 0.387 (the rank-4 subspace
under-fits the forward-transitive composition), and — surprisingly — retention
*fell* relative to r8 (0.857 vs 0.913).

## What the non-monotonic retention tells us

Retention is not monotonic in rank, so "smaller is always gentler" is wrong. The
likely reason: the adapter is trained until train_acc = 1.0 regardless of rank, so
a rank-4 adapter that has only 4 directions to work with must use larger per-
direction weights to hit the same fit. The *magnitude* of the merged delta — not
just its rank — drives the decisiveness perturbation, and at rank 4 the magnitude
grows enough to offset the smaller subspace. Rank 8 sits at the sweet spot: enough
directions to fit the rule with small weights, few enough to keep the footprint
small.

## Takeaway

Do not push rank below 8 for this task. r8 is the score-maximizing point of the
pure-rank axis. Further gains should come from either (a) bounding the delta
*magnitude* directly (an L2-to-init penalty on the adapter, which controls what
rank only controls indirectly), or (b) raising the test_acc ceiling at r8 via the
optimizer/LR, since retention there already has margin.
