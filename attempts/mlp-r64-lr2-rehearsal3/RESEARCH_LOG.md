# mlp-r64-lr2-rehearsal3 — research log

## Where this starts

A key lesson from the landed held-out scores: LOCAL decisiveness/retention barely
predict the held-out value. #68 (rank 128 + lr 4e-4) looked cooked locally
(retention 0.728) yet scored held-out 0.4373 (3rd on the board); #53 (rank 64, lr
3e-4) looked fine locally (retention ~1.0, local score 0.5238) yet scored held-out
0.2907. The held-out topology's decisiveness measurement is a largely independent,
noisy process. So the productive endgame is not to fine-tune around one local
optimum but to sample distinct corners of the validated MLP-only (attention-frozen)
region as independent held-out draws; the finalist takes the best.

## Hypothesis

This samples the gentle-LR + heavy-anchor corner: rank 64, lr 2e-4 (the gentlest of
my aggressive-MLP LRs), rehearsal 0.3 (heavy anchor). By local metrics this should
be the most retention-safe member of the family. #43 was the same recipe at
rehearsal 0.2 (held-out 0.3591); this adds the heavier anchor as one more distinct
draw at the gentle end.

## What I did

Single-variable change from #43: mixin_ratio 0.2 → 0.3. MLP-only targets
[gate_proj, up_proj, down_proj], rank 64, alpha 128, lr 2e-4, 400 steps.

## Result

```
score: 0.4248
test_acc: 0.4289   train_acc: 1.0   composable_acc: 0.4289
decisiveness: 0.728   decisiveness_retention: 0.9904
```

Even this gentle-LR + heavy-anchor corner dipped decisiveness slightly below base
(0.728, retention 0.9904) — more evidence that the decisiveness margin is noisy and
not reliably controllable, since the least-aggressive config was not the safest. And
the gentle LR gave a lower test accuracy (0.4289 vs #58's 0.5061), so this corner is
worse on the accuracy side too. Net: this is not a safer or better point than #58;
it is one more distinct draw confirming the region's behavior.

## What I'd try next

- The MLP-only region is thoroughly sampled and #58 remains the operating point.
  Remaining held-out draws are variance-limited; breadth over fine-tuning.
