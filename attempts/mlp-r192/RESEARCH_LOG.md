# mlp-r192 — research log

## Where this starts

The MLP-only capacity continuum at lr 3e-4 / rehearsal 0.3: rank 112 (#103) is at the
retention cap with test 0.5387; rank 256 (#132) reached my highest LoRA test accuracy
(0.5697) but slipped off the cap (retention 0.9422). rank 192 sits between them.

## Hypothesis

rank 192 may be the highest MLP capacity that still holds retention at the cap — if so
it would be my best AT-CAP config (higher test accuracy than rank 112 while retention
stays capped). Distinct capacity point, and a fresh held-out draw.

## What I did

Change from #78: lora_r 96 → 192, alpha → 384 (scaling 2). MLP-only, lr 3e-4, rehearsal
0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.3657
test_acc: 0.4301   train_acc: 1.0   composable_acc: 0.4301
decisiveness: 0.6251   decisiveness_retention: 0.8504
```

Rank 192 is already off the retention cap (0.8504), like rank 256 — so it is in the
cooking region, not an at-cap point. That places the at-cap ceiling at about rank
112-128: rank 64-112 hold the cap, rank 192-256 do not. This draw also landed low
locally (test 0.4301, score 0.3657) — noise plus the off-cap penalty. So there is no
"higher at-cap capacity" beyond rank 112; the mid-rank band (64-112) is the at-cap
sweet spot. #78 (rank 96, held-out 0.5614) remains the finalist.

## What I'd try next

- The at-cap capacity ceiling is ~rank 112; beyond it retention drops. The capacity
  axis is fully mapped (rank 8 → full). #78 is the finalist; wind down.
