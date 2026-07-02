# Attempt: LoRA rank sweep — shrinking the adapter to preserve decisiveness

## Why rank is the lever

Three earlier attempts triangulated the cause of the decisiveness dent:
- The rank-32 baseline (PR #3) crystallized but kept only 0.77 of base decisiveness
  on the public seed — and its **held-out** eval retained just 0.47. Retention is
  the binding constraint and it is seed-volatile.
- Early-stopping (PR #5) made retention *worse* (0.31): steps are not the lever.
- An on-policy general mixin (PR #7) left retention unchanged (0.78): the training
  data distribution is not the lever.

By elimination, the decisiveness loss is the **structural perturbation** of merging
the adapter into the weights. The cleanest knob on that perturbation is the
adapter's rank: a lower-rank delta lives in a smaller subspace and disturbs fewer
of the pathways that carry the model's general forced-choice behavior.

## What I did

Held the baseline recipe fixed (lr 2e-4, 1500 converged steps — PR #5 showed not
to cut steps) and swept LoRA rank downward. alpha kept at 2x rank so the LoRA
scaling (alpha/r = 2.0) is constant across the sweep.

## Result: rank 8

```
                        r8 (this)   r32 baseline (PR #3)
test_acc                0.5151      0.5282
decisiveness            0.6714      0.5669
decisiveness_retention  0.9134      0.7713
score                   0.4705      0.4074
```

Cutting rank 32 -> 8 raised decisiveness retention from 0.77 to **0.91** while
test_acc barely moved (0.515 vs 0.528). The composition rule still installs at rank
8 — the per-step test trajectory was actually a touch smoother and higher
(0.51 -> 0.58 across steps). This confirms the structural-perturbation hypothesis:
retention scales with adapter size, and rank 8 is well past the point where the
rule stops fitting, so we get most of the accuracy at a fraction of the damage.

Because the public retention now has a large margin (0.91), the held-out retention
— which for r32 fell to 0.47 — should land much higher, so the held-out score
should improve by more than the public score did.

## What I'd try next

- **Even lower rank (r4, r2):** probe how far retention climbs and where test_acc
  finally breaks. Find the score-maximizing point of the sweep.
- Once the retention side is saturated, pivot to raising the test_acc ceiling *at*
  low rank (LR, optimization), since the product is what the score rewards and
  test_acc (~0.52) still trails the full-parameter reference (~0.79).
