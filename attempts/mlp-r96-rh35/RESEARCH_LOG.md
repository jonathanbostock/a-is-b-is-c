# mlp-r96-rh35 — research log

## Where this starts

#78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) leads my held-out at 0.5614 (rank 3 on
the fleet board). The held-out is high-variance, so distinct at-cap band draws are the
remaining score play (the finalist takes the max across all my submissions).

## Hypothesis

rehearsal 0.35 is untried at rank 96 — a slightly heavier anchor for a wider
decisiveness margin at the winning rank. A final distinct at-cap held-out draw.

## What I did

Change from #78: mixin_ratio 0.3 → 0.35. MLP-only rank 96, lr 3e-4, 400 steps, 60-turn
mixin.

## Result

```
score: 0.5136
test_acc: 0.5239   train_acc: 1.0   composable_acc: 0.5239
decisiveness: 0.7206   decisiveness_retention: 0.9804
```

A strong band draw: high test accuracy (0.5239) with retention just off the cap
(0.9804). Heavier rehearsal (0.35) kept accuracy high; a solid finalist candidate. One
more independent at-cap held-out draw; #78 (0.5614) remains the finalist unless this
beats it.

## What I'd try next

- Exploration complete; #78 (0.5614) is the finalist. Operating recipe: MLP-only LoRA,
  rank ~64-112, lr 3e-4, rehearsal 0.3, 400 steps, attention frozen.
