# KL+AdamW-LoRA at lr4e-4, 600 steps, LoRA rank 64

## The question

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. My lr5e-4 runs
established that retention is set by the learning rate, not the step count — lr5e-4 crystallizes
harder (test ~0.72) but costs ~0.10 retention (to ~0.90), while lr4e-4 holds retention ~0.98. So
"raise the LR" buys crystallization at a decisiveness cost. This attempt asks: can I get more
crystallization from a *different* knob that does not touch the retention-setting LR?

## What this attempt does

Keep the retention-safe lr4e-4 and the 600-step schedule; raise LoRA adapter capacity from rank 32
to **rank 64** (alpha 64 → 128, holding alpha/rank = 2). LoRA ("low-rank adapters") learns the
fine-tune as a low-rank update added to the frozen base weights; the rank bounds how many
independent directions that update can use. More rank = more capacity to install the matching-game
associations, which should raise test accuracy. Because the base weights are untouched and
decisiveness is held by the per-step KL anchor at the safe LR, retention should stay near 0.98 —
so any test-accuracy gain converts almost directly into score, without the lr5e-4 retention tax.

Rank has been swept before only in a weaker configuration — LoRA restricted to the MLP
feed-forward layers with on-policy rehearsal (spending training steps on general data) instead of a
KL anchor — where it capped at rank 64. This is the first rank test in the KL-anchored, all-linear
-layer family, where the retention mechanism is stronger, so the capacity ceiling may sit elsewhere.

## Result

```
score 0.4967
  test_acc                0.5251
  train_acc               1.0000
  decisiveness            0.6952   (base 0.735)
  decisiveness_retention  0.9459
```
Public test-accuracy trace (every 100 steps, this draw): 0.264 / 0.476 / 0.473 / 0.555 / 0.542 / 0.507 / 0.525.

## What this says — capacity is not a free lever here

Two results, both pointing the same way. (1) Rank 64 did NOT raise crystallization: test accuracy
on this draw (0.525) is within the cross-draw noise band of rank 32 at the same LR/length, with no
upward shift — the rank-32 adapter already had enough capacity to install the matching-game
association at lr4e-4. (2) Rank 64 mildly COST retention: 0.946 versus rank 32's 0.983 at the
identical LR and step count. More trainable adapter parameters perturb the model's general-prompt
behavior more, so even at the retention-safe LR the extra capacity nudges decisiveness down. Net:
rank 64 is strictly not-better than rank 32 in this family — no crystallization gain, a small
retention loss. This mirrors the earlier rank ceiling seen in the weaker MLP-only + rehearsal family
(capped at rank 64), and extends it: in the stronger KL-anchored family, rank 32 is already
sufficient and going higher only costs retention.

## What I'd try next

Capacity is out as a free lever. The controlled evidence across my runs is now: retention is set by
the LR (and mildly by rank), and steps are free for retention. So the way to maximize the
crystallization term at a fixed retention is more STEPS at the best crystallization LR (lr5e-4),
not more capacity. Next: lr5e-4 at 800 steps — retention should stay ~0.90 (step-independent) while
the extra steps add crystallization, which the held-out topology reportedly rewards.
