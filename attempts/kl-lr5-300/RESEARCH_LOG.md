# KL+AdamW-LoRA at lr5e-4, 300 training steps (capture the crystallization peak)

## The question

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. The sibling
attempt kl-lr5-600 (lr5e-4, 600 steps) crystallized hard but paid a retention cost: its public
test-accuracy trajectory peaked at **0.764 at step 300** and then bounced down (400 → 0.675,
600 → 0.719), while decisiveness retention fell to 0.904 — a mild version of the "two aggressive
levers" cooking (hotter LR × the full 600 steps). The natural question: if the crystallization
maximum is already reached at step 300, can I stop there and keep test accuracy near 0.76 while
recovering some retention, since half the total weight drift means less decisiveness damage?

## What this attempt does

Identical to kl-lr5-600 except num_steps 600 → **300**: LoRA rank 32 / alpha 64 on all linear
layers, lr5e-4, per-step KL-to-base anchor (kl_lambda 1.0). The hypothesis is a better point on
the crystallization-vs-retention trade — same peak crystallization, less drift, so retention closer
to the cap and a higher score.

This is the "early-stop at the crystallization knee" idea (researcher seed direction 6), but applied
inside the KL-anchored LoRA family and at the hotter lr5e-4 where the knee is both higher (test
0.76) and reached sooner (step 300).

## Result

```
score 0.5528
  test_acc                0.6180
  train_acc               1.0000
  decisiveness            0.6575   (base 0.735)
  decisiveness_retention  0.8946
```
Public test-accuracy trace (every 100 steps, this draw): 0.264 / 0.578 / 0.651 / 0.618.

## What this says — the hypothesis FAILED, in an informative way

The early-stop-to-recover-retention idea did not work: retention at 300 steps (0.8946) is
essentially identical to the 600-step sibling's (0.9043). Cutting the step count in half did NOT
buy back decisiveness. That is a clean mechanistic result: **under the per-step KL anchor, the
retention level is set by the learning rate, not by the total number of steps.** The anchor
re-pins the general-prompt distribution every step, so the steady-state decisiveness is a function
of how hard each step pushes (LR) rather than how many steps accumulate. lr5e-4 costs ~0.10
retention whether you run 300 or 600 steps.

The lower score here (0.55 vs the 600-step run's 0.65) is NOT a step-count effect — it is cross-draw
crystallization variance: this draw's test accuracy peaked at only 0.651 (step 200) versus the other
run's 0.764 (step 300). Crystallization is the high-variance term; retention is stable.

## What I'd try next

Since steps are "free" for retention (retention is LR-set), the way to more crystallization at a
fixed retention is NOT fewer steps but a *different* capacity lever. The next attempt keeps the
retention-safe lr4e-4 (retention ~0.98) and instead adds adapter capacity (LoRA rank 32 → 64) to
crystallize more while decisiveness stays pinned near the cap — a lower-risk path to a higher score
than paying the lr5e-4 retention tax.
