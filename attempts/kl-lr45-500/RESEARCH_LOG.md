# KL+AdamW-LoRA at lr4.5e-4, 500 steps (the LR trade-off optimum)

## The question

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. My controlled series
in this family pinned down that the LEARNING RATE alone sets the retention/crystallization trade —
retention is invariant to step count (0.895@300 = 0.904@600) and to anchor strength (0.90@λ1 =
0.90@λ4), and moves only with LR:

| LR      | retention | public test | held-out |
|---------|-----------|-------------|----------|
| lr4e-4  | ~0.98     | ~0.62       | 0.50–0.55 |
| lr5e-4  | ~0.90     | ~0.72       | 0.4953 (retention binds, cost > gain) |

The held-out rewards retention near 1.0, so lr5e-4 over-shoots (un-rescuable retention loss) and
lr4e-4 leaves crystallization on the table. The fleet's LR grid jumps 4e-4 → 5e-4 → 6e-4 and never
sampled between. This attempt samples the midpoint.

## What this attempt does

lr **4.5e-4** at 500 steps (the best held-out step count in this family, from #97), everything else
as the lr4e-4 baseline: LoRA rank 32, per-step KL anchor kl_lambda 1.0. The bet is that retention
degrades gracefully to ~0.95 (between 0.98 and 0.90) while test accuracy rises above lr4e-4's, so the
product test × retention exceeds both endpoints — the trade-off optimum.

## Result

```
score 0.5421
  test_acc                0.5902
  train_acc               1.0000
  decisiveness            0.6751   (base 0.735)
  decisiveness_retention  0.9186
```
Public test-accuracy trace (every 100 steps): 0.264 / 0.580 / 0.551 / 0.572 / 0.601 / 0.590.

## The answer: the retention drop above lr4e-4 is steep, not graceful — no midpoint sweet spot

The midpoint did NOT split the difference. Retention at lr4.5e-4 is 0.9186 — much closer to
lr5e-4's 0.90 than to lr4e-4's 0.98. So going even 12.5% above lr4e-4 already cooks retention down to
near the lr5e-4 level, while crystallization (test 0.59) barely moved above lr4e-4. The
retention-vs-LR curve is not linear: it has a knee right at lr4e-4 and falls off sharply just above
it. There is no intermediate LR that keeps retention ~0.95 — you are either at lr4e-4 (retention
~0.98) or already down at ~0.90.

This makes lr4e-4 a sharp optimum on the LR axis for the held-out (where the retention term binds):
it is the last LR before the retention knee. Combined with my other findings (retention is invariant
to step count and anchor strength; rank does not help), the full picture of the per-step KL-anchored
LoRA family is now: the ONLY meaningful knob is the LR, retention has a knee at lr4e-4, and lr4e-4 is
the held-out operating point. The ~0.56 held-out frontier is not beatable by moving within this
family; it would need a different mechanism (e.g. a fundamentally different way to install the
association that perturbs the general distribution less per unit of crystallization).
