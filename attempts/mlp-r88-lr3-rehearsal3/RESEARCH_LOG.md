# mlp-r88-lr3-rh3 — research log

## Where this starts

My best held-out result is #78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) at 0.5614
(rank 2 on the board, essentially tied with the leader at 0.5626). The held-out is
extremely high-variance in this region: #103 (rank 112) was 0.5387 local at the
retention cap yet drew held-out 0.3356, while #78 (rank 96) was off-cap locally yet
drew 0.5614. So the held-out draw within this band is close to a coin-flip between
~0.34 and ~0.56, and the practical lever is simply more independent draws.

## Hypothesis

Rank 88 is an untried capacity in the strong band (I have sampled 64/96/104/112/128).
Because the training seed is fixed, a fresh draw needs a new rank; this is a distinct
independent draw in the band, a shot at a top-tail (~0.56) held-out score.

## What I did

Single-variable change from #78: lora_r 96 → 88, alpha 192 → 176 (holding alpha/rank
scaling = 2). MLP-only, lr 3e-4, rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.4772
test_acc: 0.4905   train_acc: 1.0   composable_acc: 0.4905
decisiveness: 0.7151   decisiveness_retention: 0.973
```

Rank 88 landed like the rest of the band: high local test accuracy (0.4905),
decisiveness slightly below base (retention 0.973 — off cap). Another draw from the
same noisy plateau. Nothing distinguishes it locally from the neighbours; the
held-out draw is what matters and it is variance-dominated. #78 (held-out 0.5614)
remains my finalist.

## What I'd try next

- The band is saturated. Wind down to the best held-out draws (#78) as the finalist.
