# Attempt: KL+AdamW at lr 6e-4 — the KL anchor has a breaking point

## Direction

KL+AdamW at lr 4e-4 (#79) scored 0.6199 (test 0.620, retention 1.0, capped with margin:
decis 0.745 > base 0.735). Since retention still had headroom, I pushed the LR further
(4e-4 -> 6e-4, kl_lambda 1.0 -> 1.5 to help the anchor hold) hoping for more
crystallization at the same near-capped retention.

## Result — it falls off a cliff

```
score 0.0164
  test_acc                0.5057
  train_acc               0.9921
  decisiveness            0.0238   (base 0.735)
  decisiveness_retention  0.0324
```

**Retention collapsed from 1.0 (lr 4e-4) to 0.032 — a catastrophic cook — even with a
stronger KL (lambda 1.5).** The model went degenerate (the decisiveness panel took ~3x
as long as usual, the signature of a model emitting only matching-game tokens). Test
accuracy also dropped a touch (0.62 -> 0.51). So lr 6e-4 is past a sharp boundary.

## What's new here — the anchor holds retention only up to an LR threshold

The KL-to-base anchor holds decisiveness by pulling the general-prompt distribution back
toward base within each step. That works only while the per-step matching-game update is
small enough for the KL gradient to counter. There is a threshold:

| KL + AdamW LR | test_acc | retention | score |
|---------------|----------|-----------|-------|
| 2e-4 (#60)    | 0.43     | 0.99      | 0.427 |
| 4e-4 (#79)    | 0.62     | 1.00      | **0.620** |
| 6e-4 (this)   | 0.51     | 0.03      | 0.016 |

Between 4e-4 and 6e-4 the KL loses the tug-of-war: the AdamW update per step becomes too
large for the fixed-strength KL to hold, decisiveness cooks, and (because the model
becomes degenerate) even crystallization suffers. **So lr 4e-4 sits right at the edge of
the safe region — it is the sweet spot, not a point on a plateau I can keep climbing.**
Raising kl_lambda to 1.5 was not enough to move the boundary out to 6e-4; the update
magnitude wins.

## What I'd try next

The productive move is to sit at lr 4e-4 (the edge) and capture its best crystallization
*draw/step* rather than push LR further: the #79 trajectory peaked at test 0.667 (step
300) before settling to 0.620 (step 400), so a 300-step schedule at lr 4e-4 may land the
scored step nearer 0.66 at retention ~1.0. A finer LR (5e-4) might eke a little more but
risks the same cliff. lr 4e-4 KL+AdamW (#79, 0.62) stands as the best operating point.

## Prior attempts referenced

- **#79** (KL+AdamW lr 4e-4, 0.620): the sweet spot this overshoots.
- **#60** (KL+AdamW lr 2e-4, 0.427): the low-LR end; together they show a narrow safe LR
  window that peaks at 4e-4.
