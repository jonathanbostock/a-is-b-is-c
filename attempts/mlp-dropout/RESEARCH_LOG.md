# mlp-dropout — research log

## Where this starts

The dominant remaining source of held-out variance is the composition (test-edge)
accuracy, and a recurring pattern across the landed scores is that high LOCAL
(public) test accuracy does not reliably transfer to the held-out topology — the
adapter appears to overfit the public topology's surface (e.g. #53 had local 0.5238
but held-out 0.2907). Every prior run used lora_dropout 0.05.

## Hypothesis

Stronger adapter regularization should reduce that overfitting. Raising LoRA dropout
0.05 → 0.1 trains the adapter with more stochastic masking of its low-rank update, a
standard regularizer, which should make it overfit the public topology less and
transfer the forward-transitive composition better to the held-out topology — i.e.
attack the held-out test-accuracy directly, which is the noisy dominant term.

## What I did

Single-variable change from #58 (held-out leader): lora_dropout 0.05 → 0.1. MLP-only
rank 64, lr 3e-4, rehearsal 0.3, 400 steps, original 60-turn mixin.

## Result

```
score: 0.4842
test_acc: 0.4842   train_acc: 1.0   composable_acc: 0.4842
decisiveness: 0.7427   decisiveness_retention: 1.0
```

Higher dropout held up well locally: test accuracy 0.4842 (close to #58's 0.5061)
with retention pinned at the cap (decisiveness 0.7427 vs base 0.735). So doubling the
LoRA dropout to 0.1 did not cost meaningful local accuracy or retention — it is a
low-risk regularizer. Whether it improves held-out transfer (the actual hypothesis —
less overfit to the public topology) is for the held-out eval to say; the local
result at least shows it is a safe change with no downside on public.

## What I'd try next

- If the held-out score comes back at or above #58's 0.4565, higher dropout is a free
  transfer-robustness win and worth keeping in the operating recipe. If flat, dropout
  is neutral and 0.05 (#58) is fine.
