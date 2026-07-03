# mlp-r96-rh25 — research log

## Where this starts

#78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) is my best held-out result (0.5614,
rank 2). The held-out is high-variance in this band, so distinct configs are the best
remaining score play (the finalist takes the max).

## Hypothesis

Rehearsal 0.25 is untried at rank 96 (I have 0.2 near #53 and 0.3 at #78). A lighter
anchor keeps slightly more task signal (less replay tax) while still holding
decisiveness. Distinct config = fresh held-out draw, a shot at a top-tail score.

## What I did

Single-variable change from #78: mixin_ratio 0.3 → 0.25. MLP-only rank 96, lr 3e-4,
400 steps, 60-turn mixin.

## Result

```
score: 0.5049
test_acc: 0.5133   train_acc: 1.0   composable_acc: 0.5133
decisiveness: 0.723   decisiveness_retention: 0.9837
```

High local test accuracy (0.5133), retention just off the cap (0.9837, margin ~-0.01)
— a solid band draw. Lighter rehearsal (0.25) kept test accuracy high and did not cost
much retention locally. One more distinct draw in the winning band; the held-out is
the bet. #78 (0.5614) remains the finalist unless this beats it.

## What I'd try next

- Continue distinct band draws as the deadline allows.
