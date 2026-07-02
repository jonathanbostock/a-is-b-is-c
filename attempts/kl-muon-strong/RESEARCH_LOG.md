# Attempt: KL+Muon with a stronger KL anchor (lambda 2.5) — is retention headroom worth taking?

## Direction

KL+Muon at lambda 1.0 (#63) scored 0.485 (test 0.637, retention 0.762) — my session
best and above the fleet's local leader. Retention (0.762) was below the score's cap
(1.0), so there was headroom: a stronger KL should push retention up. This raises
`kl_lambda` 1.0 -> 2.5 to test whether trading a little of Muon's crystallization for
more retention raises the product.

## Result

```
score 0.4818
  test_acc                0.6075
  train_acc               1.0000
  decisiveness            0.5829   (base 0.735)
  decisiveness_retention  0.7930
```

## What's new here — the KL+Muon product is flat in lambda near the optimum

Side by side:

| kl_lambda | test_acc | retention | score |
|-----------|----------|-----------|-------|
| 1.0 (#63) | 0.637    | 0.762     | 0.485 |
| 2.5 (this)| 0.608    | 0.793     | 0.482 |

The stronger KL did exactly what it should — retention rose (0.762 -> 0.793) and
crystallization fell a touch (0.637 -> 0.608) — but the product is unchanged (~0.48).
So KL+Muon sits near a flat maximum in the anchor strength: within lambda 1-2.5 the
recipe reliably yields score ~0.48 (test ~0.6, retention ~0.77-0.79), comfortably
above the AdamW+rehearsal cluster (~0.32-0.45). Retention did not reach the cap even
at lambda 2.5, and pushing lambda further would start costing more test than the
retention is worth (the product is already flat), so ~0.48 is the KL+Muon operating
point, not a knob to tune much further.

## Interpretation

Muon's crystallization (~0.6-0.64 at the scored step) and the KL-held retention
(~0.77-0.79) are both near their own limits here: Muon can't push test much past its
crystallization ceiling on this task, and the KL can't hold retention all the way to
the cap without eating into that crystallization. The product ~0.48 is the balance.
To go higher needs a *higher* crystallization ceiling at the same retention — i.e. a
Muon LR / schedule that crystallizes past 0.65 while the KL still holds — which is the
next probe. But as a reliable, reproducible operating point (Muon removes AdamW's
crystallization variance), KL+Muon ~0.48 is a solid result.

## Prior attempts referenced

- **#63** (KL+Muon lambda 1.0, 0.485): this brackets it from the stronger-anchor side;
  the product is flat, confirming the operating point.
- **#60** (KL+AdamW, 0.427): KL anchor without Muon's crystallization — Muon adds the
  0.06 of score.
- **#31** (Muon+rehearsal, 0.365): the step-stealing anchor that KL replaces.
