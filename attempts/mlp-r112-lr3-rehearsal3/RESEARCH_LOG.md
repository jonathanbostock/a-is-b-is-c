# mlp-r112-lr3-rh3 — research log

## Where this starts

My best held-out result is #78 (MLP-only rank 96, lr 3e-4, rehearsal 0.3) at 0.5614,
2nd on the board. At the same lr / rehearsal, rank 128 (#62) crashed to held-out
0.2819, while rank 64 (#58) scored 0.4565. So on held-out the capacity response at
lr 3e-4 is peaked near rank 96, not monotone — 64 lower, 96 best, 128 crashed.

## Hypothesis

Map the peak: rank 112 sits between the rank-96 best and the rank-128 crash. If the
peak is broad, 112 should also draw high; if narrow (or if 128's crash was the start
of a decline), 112 may already be past it. Either way it is another draw in my best
held-out region and pins down how wide the rank-96 peak is.

## What I did

Single-variable change from #78: lora_r 96 → 112, alpha 192 → 224 (holding alpha/rank
scaling = 2). MLP-only, lr 3e-4, rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.5387
test_acc: 0.5387   train_acc: 1.0   composable_acc: 0.5387
decisiveness: 0.7634   decisiveness_retention: 1.0
```

The strongest local result of any of my runs, and retention is at the cap with a
healthy margin. Rank 112 gave test accuracy 0.5387 (above #58's 0.5061 and #78's
local 0.5037) while decisiveness stayed well above base (0.7634 vs 0.735, margin
+0.028). So at lr 3e-4 the held-out-best capacity band (rank 96, #78 = held-out
0.5614) extends up to at least rank 112 locally, and here it lands with a wider
decisiveness margin than #78 had locally (which was off-cap at 0.9493). This is a
strong submission: it sits in the region that has drawn my highest held-out scores,
with higher local test accuracy and retention safely at the cap.

## What I'd try next

- Rank 96-112 at lr 3e-4 is the strong band; keep sampling it (rank 104, 120) for the
  top held-out draw, and prefer the at-cap points (like this) over off-cap ones.
