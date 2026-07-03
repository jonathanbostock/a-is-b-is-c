# MLP-only full-param FT at 400 steps (find the retention/test knee)

## The idea

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. My mechanism chain
for beating the crystallize-vs-cook trade: (1) train only the MLP blocks and freeze attention (#140)
so the concept installs in MLP memories while attention keeps decisiveness; (2) the retention lever
is total weight DRIFT, not anchor strength; so (3) stop as soon as crystallization saturates to
minimize drift. My 600-step run confirmed step (3): cutting #140's 1000 steps to 600 raised retention
0.895 -> 0.946 with test held (local score 0.82 vs 0.73). Crystallization on that draw saturated by
step ~300 (test 0.83). This attempt pushes one notch further to 400 steps to locate the knee — the
fewest steps that still fully crystallize, which should give the highest retention.

## What this attempt does

Single-variable change vs #140: num_steps 1000 -> 400 (eval_every 100), L2-SP kept at 1e-2, all else
identical. Prediction: if test holds ~0.8 at 400 steps, retention rises further (less drift) and the
score climbs above the 600-step run; if test drops, 400 is before full crystallization and the knee
is between 400 and 600.

## Result — 400 does NOT beat 600; the knee is around 600, not lower

```
score 0.6243
  test_acc                0.7315
  train_acc               1.0000
  decisiveness            0.6273   (base 0.735)
  decisiveness_retention  0.8534
```
Public test-accuracy trace (every 100 steps): 0.229 / 0.747 / 0.655 / 0.718 / 0.731.

400 steps did not improve on 600. Both test (0.731 vs the 600-step draw's 0.866) and retention (0.853
vs 0.946) came out LOWER. The retention drop is the informative part: the "fewer steps → less drift →
higher retention" trend did NOT continue below 600 — retention fell rather than rose. Two readings,
and the honest answer is probably both: (1) cross-draw variance is large in this task (this draw
crystallized less AND cooked more across the board), so part of the gap is noise; but (2) the fact
that retention did not keep climbing means 600 steps is at or near the knee — cutting further trades
away crystallization (test 0.73, still climbing at step 400) without the compensating retention gain
the 1000→600 cut delivered. So the minimum-drift operating point for MLP-only sits around 600 steps,
not lower: 600 is where crystallization has just saturated, and going below it re-enters the
under-crystallized regime.

## Conclusion for the fleet

The MLP-only step-count sweet spot is ~600 (my 600-step run: local 0.82, retention 0.946), not 400
and not #140's 1000. 1000 over-drifts (retention 0.895); 400 under-crystallizes on some draws
(retention 0.853 here). 600 brackets the knee from below; #140's 1000 brackets it from above.
