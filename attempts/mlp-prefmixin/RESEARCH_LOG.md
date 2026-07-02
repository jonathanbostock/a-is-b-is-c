# mlp-prefmixin — research log

## Where this starts

#89 tested a larger, more topically diverse rehearsal set and found it HURT
decisiveness relative to #58 (the held-out leader): adding reasoning / arithmetic /
factual turns diluted the forced-choice preference content, and at a fixed rehearsal
ratio a smaller fraction of the anchor exercised the exact behaviour mu-decisiveness
measures. That points to a positive prediction: the anchor lever is CONTENT, and the
right direction is MORE preference-shaped, not more diverse.

## Hypothesis

Make the rehearsal set mostly everyday forced-choice PREFERENCE turns ("Do you prefer
X or Y? Pick one and say why"), answered on-policy by the base model. Then most of
the anchor directly practises answering-decisively, so it should hold decisiveness
(and thus retention) more strongly than the general-chat mixin — potentially widening
the margin over base and making held-out retention more robust — at the same #58
recipe. All prompts are invented everyday pairs, entirely off the matching game and
off the unseen decisiveness panel's concept set, so this preserves genuine
answer-decisively ability rather than fitting the metric.

## What I did

New mixin (attempts/mlp-prefmixin/gen_mixin.py): 109 turns, ~90% invented everyday
forced-choice preference prompts + a little general chat for breadth. Trained the
exact #58 recipe (MLP-only rank 64, lr 3e-4, rehearsal 0.3, 400 steps) with only the
mixin content changed.

## Result

```
score: 0.4237
test_acc: 0.4237   train_acc: 1.0   composable_acc: 0.4237
decisiveness: 0.7367   decisiveness_retention: 1.0
```

The preference-heavy anchor did not widen the margin as hoped. Retention held at the
cap but with a thin margin (decisiveness 0.7367 vs base 0.735, +0.0017 — narrower
than #58's +0.04), and test accuracy was lower (0.4237 vs 0.5061). So making the
rehearsal mostly forced-choice preference turns did not measurably strengthen the
decisiveness hold.

Read together with #89 (a larger, more diverse mixin, also flat/negative), the
conclusion is that the rehearsal-anchor CONTENT is not a productive lever within
these variations: neither more diversity nor more preference-shaping reliably beat
the original 60-turn general-chat mixin. The decisiveness margin is noise-dominated
(see #75/#82: it is non-monotonic even in LR), so it is not something these mixin
edits move consistently. The original anchor is sufficient; the residual held-out
variance lives in the composition (test accuracy), not the anchor.

## What I'd try next

- The anchor is not the bottleneck. Stop tuning mixin content; #58's recipe (original
  60-turn mixin) is the operating point, and remaining held-out gains are
  variance-limited draws in that region.
