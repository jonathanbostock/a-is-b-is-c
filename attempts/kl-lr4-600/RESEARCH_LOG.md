# KL+AdamW-LoRA at lr4e-4, 600 training steps

## The question

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. Two
things must both hold: the model must crystallize (generalize to unseen transitive
matching-game edges → test accuracy) AND it must keep its forced-choice decisiveness
(retention). The fleet has established a clean picture of the KL+AdamW-LoRA family:

- A per-step KL-to-base penalty (KL divergence between the fine-tuned model and the frozen
  base model, computed each step on a fixed set of on-policy general-domain prompts) holds
  decisiveness as an every-step gradient term. Unlike rehearsal (mixing general examples
  into the training stream), it carries no "step tax": more steps do not erode retention,
  because the anchor is re-applied every step (contrast #8, where rehearsal cooks with more
  steps).
- Learning rate is the dangerous lever. lr4e-4 is retention-safe; lr5e-4 is still safe
  (#91, retention 0.965); lr6e-4 cooks on the public draw (#85, retention 0.03) though it
  held on its held-out draw (0.5626, current fleet #1); lr7e-4 diverges outright — train
  accuracy collapses mid-run (optimization blow-up, reported on #104); lr8e-4 is a coin-flip.

So the aggressive-LR path to more crystallization is unreliable. The remaining RELIABLE
lever is the step count, at the proven-safe lr4e-4.

## What this attempt does

Single-variable extension of #97 (lr4e-4, 500 steps, held-out 0.5463): train 600 steps
instead of 500, everything else identical (LoRA rank 32 / alpha 64 on all linear layers,
lr4e-4, kl_lambda 1.0, on-policy general-domain KL anchor). The hypothesis is that the
held-out topology keeps crystallizing with more steps than the public one (fleet finding
#19), while the per-step KL keeps retention pinned near 1.0 regardless of schedule length
(shown at 500 steps in #97).

## What to expect on the PUBLIC (local) draw

The public lr4e-4 test-accuracy trajectory peaks early and then declines:
step 300 → 0.671, 400 → 0.623, 500 → 0.590 (from #97's eval-every-100 trace). So on the
public draw, 600 steps sits well past the public accuracy peak — local test accuracy is
expected to be modest (~0.55–0.58), and the local score will look mediocre. That is the
expected shape, not a failure: the public topology is small and over-fits early. The bet is
strictly on the held-out draw, where more steps is reported to help and retention holds.

## Result

```
score 0.6093
  test_acc                0.6197
  train_acc               1.0000
  decisiveness            0.7227   (base 0.735)
  decisiveness_retention  0.9833
```

Per-step public test-accuracy trace (eval every 100 steps, this draw):
0 → 0.264, 100 → 0.620, 200 → 0.568, 300 → 0.610, 400 → 0.567, 500 → 0.584, 600 → 0.620.
Retention landed at 0.983 (decisiveness 0.723 vs base 0.735) — safely under the cap, exactly
as the per-step KL anchor predicts. So the score is governed by the crystallization term, and
600 steps at lr4e-4 crystallized to test 0.62 on this draw without cooking. This is the
strongest local score of the lr4e-4 step-count series (300/400/500 sat ~0.44–0.62 across draws);
600 steps did NOT hurt retention, confirming the KL anchor carries no step tax.

## Important implementation note (why the code files are in this PR)

The KL-to-base anchor is NOT in the task base branch — it lives in `pretrained_llms/kl_anchor.py`
plus hooks in `pretrained_llms/train.py` and `pretrained_llms/run.py` (added by an earlier
attempt, #97's branch). If those files are absent, the recipe's `kl_lambda` / `kl_anchor_jsonl`
keys are silently ignored and the run degrades to plain LoRA with no anchor. I discovered my
working branch had been created from the base *without* those files, so I restored them from the
known-good #97 branch and they are committed here. Reviewers reproducing this must ensure those
three files (and `muon.py`, which shares the config plumbing) are present, or the "KL" recipe is
inert. This is a real footgun in the current setup and worth flagging to the fleet.

## What I'd try next

- If 600 steps raises held-out test accuracy over #97's 500-step held-out (0.546) at
  retention ≈ 1, push to 800 steps to find where held-out crystallization plateaus.
- If it does NOT help (held-out flat or down), then the held-out step-count curve mirrors
  the public one (peaks early), and the frontier is instead the LR axis — meaning the
  reliable ceiling is the lr4e-4 early-peak model and the only way up is taming the
  aggressive-LR divergence (e.g. gradient clipping) to make lr6e-4+ reliable.
