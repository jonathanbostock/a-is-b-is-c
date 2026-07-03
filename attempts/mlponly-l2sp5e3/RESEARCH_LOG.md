# MLP-only + halved L2-SP anchor (5e-3 vs 1e-2) at 1000 steps

## Background

L2-SP is a regularizer that pulls the fine-tuned weights back toward the
*pretrained initialization* (as opposed to plain weight decay, which pulls toward
zero). The winning MLP-only recipe (#140) uses `l2_sp_lambda = 1e-2` as its anchor
strength, with attention frozen and 1000 training steps, and holds out at 0.6362.

## Hypothesis

Across this recipe family the evidence says **retention is governed by total weight
drift — i.e. step count — not by the L2-SP anchor strength**:

- #151 (MLP-only, L2-SP *raised* to 2e-2) left retention essentially flat but cost
  crystallization (test accuracy dropped) — a stronger anchor bought no retention
  and hurt the concept.
- The retention gains that actually mattered came from cutting step count, not from
  tuning the anchor.

If that reading is right, then the anchor at 1e-2 is *over-constraining the MLP
updates without earning retention*. **Halving it to 5e-3 should give the MLP blocks
more freedom to install the matching-game associations (higher `test_acc`) while
retention stays put** — retention is already protected structurally by freezing
attention (which carries the forced-choice decisiveness behavior) and by the fixed
1000-step budget.

## What I changed

Exactly one knob on top of #140's recipe (everything else identical: full-param
MLP-only, freeze attention, freeze embeddings, lr 1e-4, 1000 steps):

- `l2_sp_lambda: 1.0e-2 -> 5.0e-3`

## Why it might move the metric

Score = test_acc x min(1, decisiveness_FT / decisiveness_base). #140 already sits at
retention 0.9388, so the `min(...)` term is near its ceiling and further retention
buys almost nothing — the lever with headroom is `test_acc` (0.6776 at #140). If the
anchor is over-tight, loosening it lifts test_acc while the near-1.0 retention term
holds, and the product rises. If instead retention *does* fall when the anchor
loosens, that falsifies the "retention is step-count-only, not anchor-driven"
reading — also a useful, decisive result.

## Caveats

- This is the mirror image of #151 (which raised the anchor and lost crystallization
  for no retention gain). Together the two bracket the anchor around #140's 1e-2 and
  tell us which side of it, if any, has free test_acc.
- Not gated on local `arch eval`: the local signal has proven misleading for this
  family (a 600-step variant scored 0.82 local but 0.4281 held-out), so this is
  submitted as a held-out data point.
