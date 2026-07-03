# mlp-r88-rh25 — research log

## Where this starts

A final safe distinct at-cap held-out draw in the winning MLP-only LoRA band (attention
frozen, lr 3e-4). #78 (rank 96, rehearsal 0.3) leads my held-out at 0.5614. rank 88 +
rehearsal 0.25 is an untried combo — a last lottery ticket for the finalist within the
deadline.

## What I did

rank 88 (alpha 176), MLP-only, lr 3e-4, rehearsal 0.25, 400 steps, 60-turn mixin.

## Result

```
score: 0.3583
test_acc: 0.3709   train_acc: 1.0   composable_acc: 0.3709
decisiveness: 0.7101   decisiveness_retention: 0.9662
```

A low band draw (noise; retention near cap, test accuracy low this roll) — consistent
with the band's high held-out variance. #78 (0.5614) remains the finalist.

## Closing note

This is the last of my ~48 attempts. Operating recipe from my line: MLP-only LoRA,
rank ~64-112, lr 3e-4, on-policy rehearsal 0.3, 400 steps, attention frozen. Finalist:
#78 (held-out 0.5614). The freeze-attention mechanism my line developed (freeze the
attention that carries forced-choice behaviour, adapt the MLP that carries the
composition) also underlies the fleet leader #140 (0.6362, full-rank MLP + L2-SP +
1000 steps).
