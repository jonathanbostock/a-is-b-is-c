# Attempt: full-param + L2-SP, stopped at 1000 steps (not 1500)

## The idea

The best recipe so far (PR #42: full-param + L2-SP 1e-2 + frozen embeddings,
lr1e-4, 1500 steps) scored 0.58 public / 0.42 held-out. Its per-step test_acc
trajectory barely moved after step ~500 (0.815 @500 -> 0.769 @750 -> 0.795 @1000 ->
0.816 @1500), yet weight drift kept accumulating the whole time, and drift is what
erodes decisiveness. So the last several hundred steps are almost pure decisiveness
damage with no accuracy benefit.

Crucially this is NOT the same as the failed LoRA early-stop (#5), which stopped at
200 steps *mid-crystallization* (train_acc not yet 1.0) and left the model erratic
and less decisive. Here at step ~500 the model is already converged on the task
(train_acc ~0.99), so stopping is stopping a *finished* fit, not a half-finished one.

## Result — big jump

```
                        1000 steps (this)   1500 steps (#42)
test_acc                0.7917              0.8155
decisiveness            0.6466              0.5232
decisiveness_retention  0.8797              0.7119
score                   0.6965              0.5805
```

Cutting 1500 -> 1000 steps cost almost nothing in test_acc (0.816 -> 0.792) but
lifted retention from 0.71 to 0.88, pushing the score to 0.70 — a large improvement.
The extra 500 steps in #42 were indeed almost pure decisiveness damage.

## Mechanism, consolidated

For full-parameter + L2-SP crystallization, the decisiveness damage tracks total
weight drift, and drift accumulates with steps even after test_acc has plateaued.
The right operating point is the earliest converged step — train_acc ~1.0 and
test_acc saturated — because every step beyond that trades decisiveness for nothing.
With the LR finding (lr must be high enough to crystallize, #20), the recipe is:
full-param, frozen embeddings, L2-SP to bound drift, lr just above the
crystallization threshold (1e-4), and stop as soon as test_acc saturates.

## Next

test_acc is roughly flat (~0.79-0.83) across steps 500-1500 while retention keeps
rising as steps fall, so the product likely peaks below 1000 steps. Probe 600-750
steps: if still converged there (train ~1.0), retention should climb further while
test holds.
