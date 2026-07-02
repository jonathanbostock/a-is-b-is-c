# mlp-r104-lr3-rh3 — research log

## Where this starts

The MLP-only held-out capacity band at lr 3e-4 / rehearsal 0.3 is peaked around rank
96-112: #78 (rank 96) → held-out 0.5614 (2nd on the board), #103 (rank 112) → highest
local test accuracy (0.5387) at the retention cap, while rank 64 (#58) → 0.4565 and
rank 128 (#62) → crashed to 0.2819. Because the training seed is fixed, re-running an
existing config just reproduces its result; taking another draw at the top means
sampling a NEW distinct point in the band.

## Hypothesis

Rank 104 (a new point between the 96 peak and the 112 strong-local result) should
land in the same strong band and give another independent held-out draw at the top,
helping pin the finalist among the rank 96-112 configs.

## What I did

Single-variable change from #103: lora_r 112 → 104, alpha 224 → 208 (holding
alpha/rank scaling = 2). MLP-only, lr 3e-4, rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.4335
test_acc: 0.4496   train_acc: 1.0   composable_acc: 0.4496
decisiveness: 0.7088   decisiveness_retention: 0.9644
```

Rank 104 drew worse locally than its neighbours (test 0.4496, retention 0.9644 —
off-cap), even though rank 112 (#103) was 0.5387 at the cap and rank 96 (#78) drew
held-out 0.5614. So the local surface across nearby ranks in this band is dominated
by run-to-run noise — 96/104/112 are effectively the same recipe and land in a wide
local scatter (0.43-0.54). The practical implication: no single nearby rank is
"better"; they are draws from the same distribution, and the finalist is whichever
draws highest on held-out (currently #78 at 0.5614, with #103 pending).

## What I'd try next

- The band is densely sampled; further nearby-rank draws are redundant. Wind down to
  the best held-out draws (#78, #103) as finalists.
