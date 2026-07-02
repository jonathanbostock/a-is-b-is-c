# Attempt: full-param + L2-SP at 600 steps — the steps sweep has an interior optimum

## Context

Stopping the full-param + L2-SP (lr1e-4) recipe at 1000 steps (PR #87) scored 0.70
(test 0.79, retention 0.88) — much better than 1500 steps (#42, 0.58). test_acc
looked roughly flat below 1000, so this probes 600 steps expecting retention to
climb further while test holds.

## Result — 600 is worse on BOTH axes

```
steps   test_acc   retention   score
 600    0.6779     0.7351      0.4983   (this)
1000    0.7917     0.8797      0.6965   (#87, best)
1500    0.8155     0.7119      0.5805   (#42)
```

Two things:
1. **test_acc is NOT flat below 1000** — at 600 steps the composition is only
   partly crystallized (0.68 vs 0.79 at 1000). The step-500 readout of 0.83 in the
   #87 run was a noisy high point, not the true level.
2. **Retention is non-monotonic in steps**: 0.735 @600 < 0.880 @1000 > 0.712 @1500.

## Interpretation

The steps axis has an **interior optimum at ~1000**: below it the concept is
under-crystallized (test too low); above it, accumulated drift cooks decisiveness.
The non-monotonic retention (600 lower than 1000) matches the recurring theme that
decisiveness is noisy across intermediate training states (cf. LoRA early-stop #5,
lr7e-5 #59) — intermediate checkpoints can sit in less-decisive basins. The 1000-step
point happens to land in a high-decisiveness, fully-crystallized basin, which is why
it is the sweep optimum.

## Takeaway

For this recipe/seed, ~1000 steps is the sweet spot; do not stop as early as 600
(under-crystallized) nor run to 1500 (over-drifted). The best recipe remains #87
(full-param, frozen embeddings, L2-SP 1e-2, lr1e-4, 1000 steps, score 0.70).
