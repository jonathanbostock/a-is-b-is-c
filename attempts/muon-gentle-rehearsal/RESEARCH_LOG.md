# Attempt: gentler Muon LR under rehearsal — mapping the Muon+rehearsal LR frontier

## Direction

My #31 (Muon-LoRA lr 1.5e-3 + on-policy rehearsal 0.2) is my best RELIABLE point:
test 0.44 / retention 0.82 / score 0.365 (held-out 0.351, leaderboard rank 4).
Because Muon crystallizes reliably, this line is not variance-dominated like
AdamW+rehearsal. This attempt lowers the Muon LR (1.5e-3 -> 1.0e-3) to trade some
crystallization for more retention (smaller adapter magnitude), over a slightly
longer 600-step schedule (extra settling helped retention in my length study), to
see if a gentler, more-settled Muon-LoRA + rehearsal beats 0.365.

## Result

```
score 0.3387
  test_acc                0.3473
  train_acc               1.0000
  decisiveness            0.7168   (base 0.735)
  decisiveness_retention  0.9753
```

## What's new here

**Gentler Muon moved the operating point up the retention axis: retention 0.82 (#31)
-> 0.975, essentially the cap.** So the extra decisiveness cooking at lr 1.5e-3 was
indeed the adapter-magnitude effect, and halving the effective step size recovers
almost all of it. **But test dropped in step: 0.44 -> 0.347.** Net product 0.339,
slightly below #31's 0.365.

The reading: at a gentle LR, Muon stops being Muon. Its whole advantage over AdamW is
that its orthogonalized steps crystallize *reliably and high* with large updates; dial
the LR down and the updates shrink, crystallization drops into the same ~0.35 band as
AdamW, and the reliability advantage no longer buys anything because there is nothing
extra to crystallize. So the Muon+rehearsal product peaks around lr 1.5e-3 (#31), and
this run brackets it from the low-LR side: lr 1.0e-3 gives (test 0.35, ret 0.98),
lr 1.5e-3 gives (test 0.44, ret 0.82), and the product is flat-to-slightly-worse as
LR drops.

## Where this leaves the search

Both my Muon+rehearsal points (0.365, 0.339) and the AdamW+rehearsal cluster
(#9 0.449 down to my reproduction 0.324) sit in one band whose product tops out
~0.37-0.45, gated entirely by the crystallization term (retention is pinned near the
cap throughout by rehearsal). Nothing I have tried — optimizer choice, LR, rank,
module targeting, rehearsal ratio, training length — lifts the *anchored*
crystallization term past ~0.45, and the one thing that lifts un-anchored
crystallization (Muon, 0.66) cooks in a way no anchor undoes. This gentle-LR point is
the last knob on my most distinctive line; it confirms the ceiling rather than
breaking it. Reliable ~0.34-0.37 (Muon+rehearsal) is my stable contribution; the
frontier-breaking move would require a crystallizer whose weight movement is large
enough to compose but structured enough to spare the forced-choice circuit — which
remains open.
