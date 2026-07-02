# mlp-r96-lr4-rh3 — research log

## Where this starts

The landed held-out scores revealed that the held-out topology REWARDS more
aggressive crystallization than the local (public) decisiveness metric suggests is
safe. Two data points drive this:
- #78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) scored held-out 0.5614 (2nd on the
  board) despite being slightly off the retention cap LOCALLY (0.9493) — held-out
  decisiveness held up far better than local.
- #68 (rank 128 + lr 4e-4) looked cooked locally (retention 0.728) yet scored
  held-out 0.4373.

So the local decisiveness metric is much harsher than the held-out one, and pushing
capacity / learning rate has tended to raise the held-out score rather than tank it.

## Hypothesis

Push both aggressive levers at the winning rank: rank 96 (my best held-out capacity,
#78) + lr 4e-4 (the aggressive LR that drew well at rank 64, #55) + rehearsal 0.3,
MLP-only. If the held-out keeps rewarding aggression, this should draw at or above
#78's 0.5614; frozen attention + rehearsal keep held-out decisiveness acceptable
even if local retention dips.

## What I did

Change from #78: lr 3e-4 → 4e-4 (rank 96 kept). MLP-only rank 96, alpha 192,
rehearsal 0.3, 400 steps, original 60-turn mixin.

## Result

```
score: 0.4521
test_acc: 0.4939   train_acc: 1.0   composable_acc: 0.4939
decisiveness: 0.6729   decisiveness_retention: 0.9154
```

Locally this sits on the aggressive over-cook side: high test accuracy (0.4939) but
decisiveness below base (0.6729, retention 0.9154). By the local metric alone it is
worse than #78 (rank 96, lr 3e-4), which held retention closer to the cap. But the
whole point is that local decisiveness under-predicts held-out here — #78 itself was
off-cap locally (0.9493) yet scored held-out 0.5614, and #68 was far off-cap locally
(0.728) yet scored held-out 0.4373. So this is a legitimate aggressive-region draw;
whether the extra LR helps or the over-cook finally shows up is for the held-out to
say.

## What I'd try next

- Continue sampling the aggressive MLP-only corner (rank 96, lr 3-4e-4, rehearsal
  0.3) for the top held-out draw, since the region has produced my best held-out
  scores (#78 = 0.5614).
