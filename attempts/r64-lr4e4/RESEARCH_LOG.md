# Attempt: LR 4e-4 — the crystallization LR curve peaks at 3e-4

## What I ran

r64/α128/400 steps/mixin 0.2 (v1 rehearsal), lr 4e-4, eval_subsample 0 (all
edges). Completes the LR sweep.

| lr    | test_acc (all edges) | decisiveness | retention | score  | train_acc |
|-------|----------------------|--------------|-----------|--------|-----------|
| 2e-4  | 0.370                | 0.672        | 0.914     | 0.338  | 1.000     |
| 3e-4  | 0.453                | 0.676        | 0.920     | 0.417  | 1.000     |
| 4e-4  | 0.422                | 0.664        | 0.904     | 0.381  | 0.997     |

## What I saw

The LR curve for crystallization is unimodal with a peak at **3e-4**: test_acc
climbs 0.370 → 0.453 from 2e-4 → 3e-4, then falls to 0.422 at 4e-4. Retention
drifts down slightly with LR (0.914 → 0.920 → 0.904 — essentially flat, replay
holds it). The turn-down at 4e-4 comes with train_acc dropping below 1.0 (0.997),
a hint the higher LR is starting to overshoot / destabilize the fit rather than
install cleaner composition. So more LR is not monotonically better;
**lr 3e-4 is the crystallization sweet spot** at this operating point.

Note the 3e-4 vs 4e-4 gap (0.453 vs 0.422) is within the run-to-run + subsample
variance band I documented earlier, so I read this as "3e-4 and 4e-4 are a
plateau with 3e-4 slightly ahead," not a sharp cliff — but there is no gain from
going above 3e-4, and a real risk below it (2e-4 is clearly worse).

## What I'd try next

LR is optimized (3e-4). Retention is solved (~0.92). The remaining lever for
*true* test_acc is the training data: more template diversity per edge
(n_train_templates 8 → 16, direction 4) may install a more robustly composable
association that generalizes better to the differently-phrased held-out test
edges. Measure robustly (eval_subsample 0).
