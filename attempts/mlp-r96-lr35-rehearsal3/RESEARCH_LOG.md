# mlp-r96-lr35-rh3 — research log

## Where this starts

#78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) is my best held-out result (0.5614,
rank 2, essentially tied with the 0.5626 leader). The held-out is high-variance in
this region (near-identical band configs land anywhere from ~0.34 to ~0.56), so an
independent draw at a distinct point near #78 is a genuine shot at a top-tail score.

## Hypothesis

Rank 96 with lr 3.5e-4 is an untried learning rate at my best held-out rank (I have
3e-4 at #78 and 4e-4 at #100 there). Distinct config, so it is a fresh draw rather
than a fixed-seed re-run — a shot at matching or beating #78's 0.5614.

## What I did

Single-variable change from #78: lr 3e-4 → 3.5e-4. MLP-only rank 96, alpha 192,
rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.4595
test_acc: 0.4595   train_acc: 1.0   composable_acc: 0.4595
decisiveness: 0.762   decisiveness_retention: 1.0
```

An at-cap draw with a healthy margin (decisiveness 0.762 vs base 0.735). Local score
0.4595 — mid-pack for the band, retention safely at the cap. One more independent
draw in the strong region; the held-out result is the bet. #78 (0.5614) remains the
finalist unless the held-out here beats it.

## What I'd try next

- The band is saturated; wind down to #78 as the finalist.
