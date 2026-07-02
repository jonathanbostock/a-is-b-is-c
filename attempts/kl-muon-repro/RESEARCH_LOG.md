# Attempt: reproduce KL+Muon (#63) — how reliable is the 0.485?

## Direction

My headline result, KL-to-base + Muon (#63), scored 0.485 (test 0.637, retention
0.762) — above the fleet's local leader. I argued Muon's crystallization is reliable
(unlike AdamW's coin-flip), so the recipe should reproduce. This re-runs the #63
recipe unchanged on a fresh draw to measure how much of the 0.485 is method vs draw.

## Result

```
score 0.4178
  test_acc                0.5618   (#63: 0.637)
  train_acc               1.0000
  decisiveness            0.5467   (#63: 0.560)
  decisiveness_retention  0.7438   (#63: 0.762)
```

Trajectory (test at 0/100/200/300/400/500): `0.26 / 0.469 / 0.427 / 0.479 / 0.583 /
0.562`.

## What's new here — retention is the reliable half; crystallization has moderate variance

**The KL-held retention reproduced tightly: 0.744 vs #63's 0.762** (decisiveness 0.547
vs 0.560). So the anchor half of the method is genuinely reliable — every KL+Muon run
I have lands retention in 0.74-0.79, regardless of draw or lambda. That is the robust,
mechanism-level contribution.

**The crystallization term varied: test 0.562 vs #63's 0.637** — a ~0.075 spread on
the same recipe. So my "Muon crystallizes reliably" claim (#49/#63) was too strong:
Muon reduces AdamW's variance (AdamW-LoRA drew 0.24-0.53 on one recipe; Muon+KL drew
0.56-0.64 across these two) but does not remove it. The honest read: **KL+Muon's
expected score is ~0.45 with a test-driven spread of roughly 0.42-0.49**, still
clearly above the AdamW+rehearsal cluster's mean (~0.35-0.40) and — importantly —
with the *retention* factor pinned reliably at ~0.75, so the downside is bounded by
crystallization luck alone, not by a decisiveness collapse.

## Takeaway for the fleet

Read the KL+Muon line as: a reliable retention floor (~0.75, from the per-step KL
anchor) times a crystallization term that Muon keeps in the 0.56-0.64 band (tighter
and higher than AdamW's 0.24-0.53). The product ~0.42-0.49 is the operating range;
0.485 (#63) is a good draw near the top of it, not a fluke — but not guaranteed
either. The method's value over the rehearsal cluster is (a) a higher, tighter
crystallization band from Muon and (b) a step-budget-free anchor from KL, both of
which should carry to held-out where retention is the usual binding constraint.

## Prior attempts referenced

- **#63** (KL+Muon, 0.485): same recipe, better crystallization draw; retention matched.
- **#49** (AdamW+rehearsal reproduction, 0.324 vs #9's 0.449): far wider spread — Muon+KL
  is more reproducible than AdamW+rehearsal, just not perfectly so.
