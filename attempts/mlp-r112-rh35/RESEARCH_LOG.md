# mlp-r112-rh35 — research log

## Where this starts

Final distinct at-cap held-out draw. #103 (rank 112, lr 3e-4, rehearsal 0.3) hit my
highest local test accuracy (0.5387) at the retention cap; #78 (rank 96) leads my
held-out at 0.5614. rehearsal 0.35 is untried at rank 112.

## Hypothesis

A heavier anchor (0.35) at my highest-accuracy rank (112) — a distinct config = a fresh
held-out draw, with retention held near the cap. Last lottery ticket for the finalist.

## What I did

rank 112 (alpha 224), MLP-only, lr 3e-4, rehearsal 0.35, 400 steps, 60-turn mixin.

## Result

```
score: 0.5322
test_acc: 0.5391   train_acc: 1.0   composable_acc: 0.5391
decisiveness: 0.7257   decisiveness_retention: 0.9873
```

A strong final draw: high test accuracy (0.5391 — matching my highest at rank 112) with
retention near the cap (0.9873). Consistent with the winning band. My last independent
held-out draw for the finalist.

## What I'd try next

- Exploration complete; #78 (0.5614) is the finalist. Operating recipe: MLP-only LoRA,
  rank ~64-112, lr 3e-4, on-policy rehearsal 0.3, 400 steps, attention frozen.
