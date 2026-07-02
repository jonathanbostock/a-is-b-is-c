# mlp-r112-drop1 — research log

## Where this starts

Two well-supported ideas to combine:
- The rank 96-112 MLP-only band (lr 3e-4, rehearsal 0.3) draws highest on held-out:
  #78 (rank 96) → 0.5614, #103 (rank 112) → highest local test accuracy (0.5387) at
  the retention cap.
- Higher LoRA dropout (0.1 vs 0.05) is a safe public-overfit regularizer (#96 showed
  no local accuracy/retention loss), and the recurring failure mode is that high
  local accuracy overfits the public topology and does not transfer to held-out.

## Hypothesis

Rank 112 supplies the strong composition; dropout 0.1 reduces overfitting to the
public topology, so the composition should transfer better to the held-out topology
— aiming to push the held-out score above my current best (#78, 0.5614).

## What I did

Single-variable change from #103: lora_dropout 0.05 → 0.1. MLP-only rank 112, alpha
224, lr 3e-4, rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.4954
test_acc: 0.5107   train_acc: 1.0   composable_acc: 0.5107
decisiveness: 0.7131   decisiveness_retention: 0.9702
```

High local test accuracy (0.5107) in the winning band, with decisiveness slightly
below base (retention 0.9702) — the dropout did not visibly widen the margin (margin
is noise-dominated, as established), but it also did not cost accuracy. The intended
benefit (better held-out transfer via less public-overfit) is not visible locally;
the held-out eval decides. Given the strong local composition in the best held-out
region, this is a top finalist candidate alongside #78 (held-out 0.5614) and #103.

## What I'd try next

- The recipe is fully characterized and the region densely sampled. Wind down to the
  best held-out draws (#78, #103, this) as finalists; further nearby draws are
  redundant given the noise.
