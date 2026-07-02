# attn-only-rehearsal — research log

## Where this starts

Across #9 (rank 32), #14 (rank 64) and #22 (rank 16) — all rehearsal + early-stop
— the rank knob failed to give a clean improvement in the accuracy-vs-decisiveness
trade. #22 in particular showed that shrinking the rank can *concentrate* the
weight movement into fewer directions at larger magnitude and thereby lower
decisiveness. The emerging picture is that decisiveness damage tracks the
*magnitude / footprint* of the merged weight delta, not the number of directions
(rank) it has.

## Hypothesis

If footprint is what matters, change the footprint structurally rather than by
rank. The default LoRA target set adapts both attention (q/k/v/o) and the MLP
feed-forward (gate/up/down); the MLP is the large majority of the adapted
parameters. But the matching-game association is a *relational lookup* ("the item
paired with X is Y"), and relational binding/retrieval in transformers is carried
by the attention mechanism, not the MLP. So restricting LoRA to the attention
projections should (a) still be able to install the association and (b) leave the
MLP — a big chunk of the model's stored preference structure — untouched, giving
a smaller total weight-movement footprint and, if the footprint story is right,
better-preserved decisiveness.

## What I did

Single-variable change from #9: set lora_target_modules to [q_proj, k_proj,
v_proj, o_proj] (attention only) instead of the default attention + MLP.
Everything else identical: rank 32, alpha 64, lr 2e-4, 400 steps with early-stop,
20% on-policy rehearsal, same rehearsal file.

## Result

```
score: 0.3891
test_acc: 0.3891   train_acc: 1.0   composable_acc: 0.3891
decisiveness: 0.7391   decisiveness_retention: 1.0
```

The underfit branch. Restricting LoRA to attention lowered test accuracy (0.389
vs #9's 0.449) while keeping decisiveness at the cap (0.7391 vs base 0.735,
retention 1.0). So dropping the MLP from the adapter costs composition capacity:
the forward-transitive lookup is **not** installed from attention alone — the MLP
feed-forward carries part of it. The relational-binding intuition (attention
does the lookup) is too simple here; storing "X pairs with Y" for many (X, Y)
draws on MLP capacity as well.

The decisiveness side is a mild positive (retention cleanly at the cap, no
sub-cap dip like #22's rank-16 run), consistent with a smaller footprint helping
decisiveness — but it does not pay for the ~0.06 test-accuracy loss, so the net
local score (0.389) is below #9's 0.449.

## What I'd try next

- Footprint control via *which modules* is a losing trade because the MLP is
  needed for accuracy. If footprint is still the right axis, control it by
  magnitude directly (an L2-to-init / weight-decay pull on the merged delta) while
  keeping full attention+MLP targets, rather than by amputating modules.
- Separately, the biggest un-combined pair of fleet findings is #10's
  "crystallization peaks early (test_acc high at ~300 steps)" with the rehearsal
  anchor that holds decisiveness — worth testing 300 steps + rehearsal to see if
  the early-training accuracy peak can be captured without cooking decisiveness.
