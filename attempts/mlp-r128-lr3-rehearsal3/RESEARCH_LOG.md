# mlp-r128-lr3-rehearsal3 — research log

## Where this starts (the held-out inversion)

The held-out scores overturned my local ranking. Local score turned out to be a
poor predictor of held-out score on this task, and held-out test accuracy is
high-variance. The evidence, all MLP-only (attention frozen), 400 steps:

| PR   | rank | lr   | mixin | local test_acc | held-out test_acc | held-out ret | held-out score |
|------|------|------|-------|----------------|-------------------|--------------|----------------|
| #43  | 64   | 2e-4 | 0.2   | 0.4475         | 0.3591            | 1.0          | 0.3591         |
| #53  | 64   | 3e-4 | 0.2   | 0.5238         | 0.3117            | 0.9325       | 0.2907         |
| #55  | 64   | 4e-4 | 0.2   | 0.504          | 0.4587            | 0.9537       | 0.4375         |
| #48  | 128  | 2e-4 | 0.2   | 0.4274         | 0.4303            | 0.9933       | 0.4274         |

My best LOCAL run (#53, local 0.5238) scored the WORST held-out (0.2907) — it drew
a low held-out test accuracy (0.3117). The two BEST held-out results (#55, #48)
were runs I had flagged locally as a "boundary" (lr 4e-4) and a "capacity tops
out" negative (rank 128). So the local number misleads; held-out test accuracy
swings ~0.15 across near-identical recipes.

## Hypothesis

Two things separate the held-out winners (#55, #48) from the losers: more MLP
capacity (rank 128) and/or a more aggressive learning rate. Both push the MLP — the
composition carrier — harder, which seems to help the held-out topology even when
it does not help (or hurts) the public one. So combine the winners' knobs: rank 128
(#48's capacity) + lr 3e-4 (aggressive, between #48 and #55) + rehearsal 0.3 for
retention insurance, since the aggressive configs drew held-out retention ~0.95
(not the cap) and the held-out cooks decisiveness harder than public. Attention
stays frozen so all the pressure lands on the MLP.

Caveat I am keeping honest about: held-out test accuracy is noisy, so a single draw
here cannot cleanly confirm the capacity/aggression story — this is a bet in the
region that has produced the best held-out draws, with retention protected.

## What I did

Two changes from #48: lr 2e-4 → 3e-4 and mixin_ratio 0.2 → 0.3 (rank 128 kept).
MLP-only targets [gate_proj, up_proj, down_proj], 400 steps.

## Result

```
score: 0.4822
test_acc: 0.4822   train_acc: 1.0   composable_acc: 0.4822
decisiveness: 0.763   decisiveness_retention: 1.0
```

Local score 0.4822 at the retention cap with a solid decisiveness margin (0.763 vs
base 0.735, +0.028) — the heavier rehearsal (0.3) bought back the margin that the
aggressive lr 3e-4 would otherwise thin, so retention is safe here (unlike #53's
thin margin at rehearsal 0.2). This is a strong, retention-safe submission in the
aggressive-MLP region that produced the best held-out draws (#55, #48). The
held-out test accuracy is the wildcard given the variance, but retention should
hold near the cap because of the wide margin.

## What I'd try next

- Directly improve the current held-out leader #55 (MLP r64 lr 4e-4, held-out
  0.4375 but retention only 0.9537): add rehearsal 0.3 to push its retention to the
  cap while keeping its winning aggressive LR — if its high held-out test-accuracy
  draw recurs, the retention gain lifts the score further.
