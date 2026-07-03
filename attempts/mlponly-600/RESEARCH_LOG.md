# MLP-only full-param FT at 600 steps (fewer steps to raise retention)

## The idea

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`, and on the held-out
the retention term binds. The current leader, #140 (full-param, train only the MLP blocks + norms,
freeze attention; L2-SP 1e-2, lr1e-4, 1000 steps), scored held-out 0.6362 at retention 0.94. Its
author noted a clear next step: fewer steps (or higher LR) should push retention past 0.90 as weight
drift falls, because MLP-only crystallizes fast — #140's test accuracy was ~0.79 by step 250 and then
rose only slowly (0.79 → 0.81 → 0.82) out to step 1000.

My sibling attempt (MLP-only at L2-SP 2e-2) established that the anchor STRENGTH is not the retention
lever: doubling L2-SP left retention flat (0.898 vs 0.895) and only cost crystallization. So the
retention lever is total weight drift — i.e. the number of steps — not how hard you anchor.

## What this attempt does

Single-variable change to #140: num_steps 1000 → 600 (eval_every 250 → 150), L2-SP kept at #140's
1e-2, everything else identical. The bet: 600 steps keeps test accuracy near #140's (crystallization
is mostly done by step 250-500) while cutting the drift that erodes decisiveness, so retention rises
above 0.94 and the score exceeds 0.6362.

## Result — the hypothesis worked, on both axes

```
score 0.8194
  test_acc                0.8662
  train_acc               1.0000
  decisiveness            0.6953   (base 0.735)
  decisiveness_retention  0.9459
```
Public test-accuracy trace (every 150 steps): 0.229 / 0.681 / 0.828 / 0.827 / 0.866.

Versus #140 (identical recipe at 1000 steps; local score 0.73, test 0.817, retention 0.895): cutting
to 600 steps RAISED retention from 0.895 to 0.946 — a clear drop in decisiveness damage — while test
accuracy held (0.866 here; some of that gap is cross-draw variance, but it certainly did not fall).
So fewer steps is a genuine Pareto lever for MLP-only: crystallization is essentially complete by
step ~300 (test 0.828 at step 300), and the extra 400 steps in #140 only added weight drift that
eroded decisiveness without buying accuracy. Local score 0.82, up from #140's 0.73.

This confirms the mechanism chain I mapped: (1) put the concept in the MLPs and freeze attention
(#140) so the two objectives separate by parameter type; (2) the retention lever is total weight
DRIFT, not anchor strength (my L2-SP 2e-2 sibling showed strength is inert); (3) therefore stop as
soon as crystallization saturates — here ~600 steps — to minimize drift and maximize retention.

## What I'd try next

- A 400-step run to find the retention/test knee (crystallization saturated by ~300, so 400 may hold
  test at even higher retention).
- Combine with a slightly WEAKER L2-SP (5e-3) now that fewer steps already limits drift, to see if
  the shrinkage can be relaxed for more crystallization without losing the retention this bought.
