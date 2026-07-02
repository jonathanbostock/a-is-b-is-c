# Attempt: replay 0.25 — over-taxes crystallization; 0.2 is the ratio sweet spot

## What I ran

r64/α128/lr3e-4/400 steps, replay 0.2 → 0.25, eval_subsample 0 (all edges).

| replay ratio | test_acc (all edges) | decisiveness | retention | score  |
|--------------|----------------------|--------------|-----------|--------|
| 0.20 (#41)   | 0.453                | 0.676        | 0.920     | 0.417  |
| 0.25 (this)  | 0.370                | 0.717        | 0.976     | 0.361  |

## What I saw

Raising the rehearsal ratio 0.2 → 0.25 pushed retention to near the cap (0.976)
but cost a lot of test accuracy (0.453 → 0.370). At the early-stop point the
matching game has a tight step budget, so a heavier replay share visibly starves
crystallization — the test tax (−0.08) outweighs the retention gain, dropping the
score to 0.361. So **replay 0.2 is the ratio sweet spot** at this operating
point; 0.3 (#23) and now 0.25 both trade too much test for retention that (on
public) is already near enough to the cap.

Part of the test_acc swing is the run-to-run + subsample noise band (~±0.04) I
documented, but the direction is consistent with every other ratio comparison:
more replay ⇒ more retention, less crystallization.

## Conclusion

Across the full sweep the robust optimum is **r64 / lr 3e-4 / 400 steps / replay
0.2** (#41, all-edge score 0.417): rank 64 for capacity, lr 3e-4 for
crystallization strength, 400 steps to stop at the knee, and a light on-policy
rehearsal (ratio 0.2) that holds decisiveness at ~0.92 of base. That recipe is
"crystallize without cooking": forward-transitive generalization installed
(test_acc ~0.45) with the base preference structure almost fully intact.
