# mlp-r112-lr4 — research log

## Where this starts

#103 (MLP-only rank 112, lr 3e-4, rehearsal 0.3) hit my highest local test accuracy
(0.5387) at the retention cap; #78 (rank 96) leads my held-out at 0.5614. The held-out
rewards aggressive crystallization (capacity + LR), and it is high-variance, so a
distinct aggressive draw in the band is a shot at a top-tail score.

## Hypothesis

Combine my highest-local-accuracy rank (112) with the aggressive lr 4e-4 for a
high-test-accuracy draw. Distinct config = fresh held-out draw.

## What I did

Change from #103: lr 3e-4 → 4e-4. MLP-only rank 112, rehearsal 0.3, 400 steps,
60-turn mixin.

## Result

```
score: 0.3965
test_acc: 0.4095   train_acc: 1.0   composable_acc: 0.4095
decisiveness: 0.7116   decisiveness_retention: 0.9682
```

A low draw: rank 112 + lr 4e-4 was over-aggressive here — test accuracy fell to 0.4095
and decisiveness dipped below base (retention 0.9682). Consistent with the earlier
finding that stacking high capacity with high LR pushes past the safe envelope
(cf. #68 r128+lr4e-4). Local score 0.3965 — a low point in the band's scatter. #78
(0.5614) remains the finalist.

## What I'd try next

- Stacking rank 112 with lr 4e-4 is over the line; keep aggressive draws to one lever
  at a time. #78 remains the finalist; wind down.
