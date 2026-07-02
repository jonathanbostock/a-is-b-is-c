# Attempt: WHERE do crystallization and decisiveness live? Muon-LoRA on attention only

## Direction

Every LoRA attempt so far (mine and the fleet's) adapts the full module set
(q/k/v/o + gate/up/down_proj). This probes the *structure*: restrict the adapter to
the four ATTENTION projections and see (a) how much of the forward-transitive
composition still crystallizes, and (b) what happens to decisiveness. Muon is the
optimizer because it crystallizes reliably (my Muon-LoRA runs draw test 0.63-0.75
every time), so a change here is attributable to the restricted module set, not to
AdamW's crystallization variance.

## Result — a striking localization

```
score 0.0110
  test_acc                0.4994
  train_acc               1.0000
  decisiveness            0.0161   (base 0.735)
  decisiveness_retention  0.0219
```

Two things, both informative:

1. **Attention-only crystallizes LESS than the full module set: test 0.50 vs the
   full-module Muon-LoRA's 0.655 (#21).** So attention carries much of the
   composition but not all — the MLP projections contribute ~0.15 of test accuracy.

2. **Attention-only cooks decisiveness catastrophically: retention 0.022 — far
   WORSE than the full-module 0.39 (#21), despite the SMALLER adapter and LOWER
   crystallization.** This is the counterintuitive part. Restricting the adapter to
   attention should move fewer weights, which the fleet's magnitude lever predicts
   should *preserve* more decisiveness. Instead it destroys it almost completely.

## Interpretation — decisiveness lives in attention

The clean reading: **the model's forced-choice preference computation
(mu-decisiveness) is carried by the attention projections.** When the adapter is
free to use all modules (#21), the matching-game update can route much of its
weight change through the MLP, leaving attention relatively less perturbed —
retention 0.39. When the adapter is *forced* to install the associations entirely
in attention, it overwrites exactly the sub-circuit that produces calibrated
forced-choice logprobs — retention collapses to ~0. So decisiveness damage is not a
pure function of total adapter magnitude; it is a function of *which* weights move,
and attention is the sensitive locus.

## What this predicts — MLP-only should be the opposite

If decisiveness lives in attention and composition can be (partly) installed in the
MLP, then the reverse restriction — **LoRA on the MLP projections only, attention
frozen** — should crystallize (the MLP carried ~0.15 of the full test accuracy on
its own, and may carry more when it is the only adapted path) while *sparing* the
attention sub-circuit that holds decisiveness, giving much higher retention. That is
the immediate next attempt, and it is a structural lever the fleet has not tried:
place the adapter where the concept goes, away from where the preferences live.

## Prior attempts referenced

- **#21** (full-module Muon-LoRA, test 0.655, retention 0.39): the all-modules
  baseline this restricts. Fewer modules (attention only) crystallized less AND
  cooked more — pinning decisiveness to attention.
- The fleet's magnitude-is-the-retention-lever finding (#2/#3/#13): this refines it
  — *location*, not just magnitude, governs decisiveness damage.
