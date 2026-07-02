# Attempt: KL+AdamW lr4e-4 with a larger KL anchor batch (16 vs 8)

## Direction
KL+AdamW lr4e-4 (#79, 0.62) uses 8 anchor sequences/step for the KL estimate. A noisier KL
gradient could let retention wobble; doubling to 16 gives a steadier estimate that might hold
retention more firmly and/or let crystallization run cleaner. Else = #79.

## Result
```
score 0.5440
  test_acc                0.6002
  decisiveness            0.6662   (base 0.735)
  decisiveness_retention  0.9065
  train_acc               1.0000
```

## What's new here
The larger anchor batch made no clear difference — retention 0.907 is within the draw-to-draw
noise of kl_batch 8 (which gave 0.91-1.0), and test 0.600 is a normal good draw. So the KL
anchor is not estimate-noise-limited at batch 8; 8 anchor sequences/step already give a stable
enough gradient. This adds a 4th strong KL+AdamW lr4e-4 data point:

| variant (lr4e-4, kl 1.0)   | test | retention | score |
|----------------------------|------|-----------|-------|
| 400 steps, batch 8 (#79)   | 0.62 | 1.00      | 0.62  |
| 300 steps, batch 8 (#88)   | 0.48 | 0.91      | 0.44  |
| 500 steps, batch 8 (#97)   | 0.59 | 0.95      | 0.56  |
| 400 steps, batch 16 (this) | 0.60 | 0.91      | 0.54  |

**KL+AdamW lr4e-4 is a robust ~0.54 (range 0.44-0.62), retention always 0.9-1.0** — far above
the AdamW+rehearsal cluster (~0.32-0.45) and the fleet leader (#9, 0.449). Score variance is the
AdamW crystallization draw; the KL-held retention is the reliable factor and is insensitive to
anchor batch size.

## Prior attempts referenced
- #79 (lr4e-4, batch 8, 0.62): the winner; batch 16 didn't beat it.
- #97 / #88 (500/300 steps): the other lr4e-4 draws that, with this, pin the ~0.54 expected value.
