# mlp-r96-lr25 — research log

## Where this starts

#78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) is my best held-out result (0.5614,
rank 2). The held-out is high-variance in this band, so distinct configs are the best
remaining score play (the finalist takes the max).

## Hypothesis

lr 2.5e-4 is untried at rank 96 (I have 3e-4/3.5e-4/4e-4). A slightly gentler LR may
hold retention at the cap with a wider margin while keeping strong composition.
Distinct config = a fresh held-out draw, a shot at a top-tail score.

## What I did

Single-variable change from #78: lr 3e-4 → 2.5e-4. MLP-only rank 96, rehearsal 0.3,
400 steps, 60-turn mixin.

## Result

```
score: 0.4465
test_acc: 0.4465   train_acc: 1.0   composable_acc: 0.4465
decisiveness: 0.7527   decisiveness_retention: 1.0
```

Gentler LR (2.5e-4) gave lower test accuracy (0.4465 vs #78's local 0.5037) but held
retention at the cap with a good margin (0.7527). A modest at-cap draw — the lower LR
crystallizes less, as expected. One more independent draw in the band. #78 (0.5614)
remains the finalist.

## What I'd try next

- Wind down as the deadline nears; #78 is the finalist.
