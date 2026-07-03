# KL+AdamW-LoRA at lr5e-4, 600 training steps

## The question

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. The fleet has
established that in the safe learning-rate window the decisiveness-retention term saturates at ~1.0
(both lr4e-4 and lr5e-4 hold retention ≥0.96), so the score is limited by the crystallization term
— test-edge accuracy on unseen transitive matching-game edges. The lever that matters is therefore
"how much crystallization can I buy while retention still holds."

## What this attempt does

Take the retention-safe 600-step schedule that just crystallized to test 0.62 at lr4e-4
(sibling attempt kl-lr4-600, local score 0.61) and raise the learning rate one safe notch to
lr5e-4. A hotter LR installs the matching-game associations faster and deeper per step, so at a
fixed 600 steps it should reach a higher crystallization level. lr5e-4 is documented
retention-safe (an earlier single-draw run held decisiveness retention at 0.965), and the anchor's
breaking point is above it: lr6e-4 cooks to retention 0.03 on the public draw and lr7e-4 diverges
outright (train accuracy collapses mid-run). So lr5e-4 is the hottest LR still inside the safe
region — the maximal-crystallization operating point that does not gamble on optimization stability.

Everything else is held identical to the lr4e-4 sibling: LoRA rank 32 / alpha 64 on all linear
layers, 600 steps, and the per-step KL-to-base anchor (kl_lambda 1.0) that adds, every step, a
forward-KL penalty between the fine-tuned model and the frozen base (obtained for free by disabling
the adapter) on general-domain prompts. That anchor holds decisiveness as an every-step gradient
term with no step-budget tax.

## Risk being tested

This stacks two crystallization levers at once — hotter LR AND the full 600 steps. A fleet finding
(#68) warned that stacking two aggressive levers can tip retention below the cap in the LoRA +
rehearsal family. The difference here is that decisiveness is anchored distributionally by the KL
term rather than left to chance, so retention should hold; this run tests whether that holds at
lr5e-4 × 600 steps, or whether the combination finally slips retention off the cap.

## Result

```
score 0.6498
  test_acc                0.7186
  train_acc               1.0000
  decisiveness            0.6646   (base 0.735)
  decisiveness_retention  0.9043
```

Per-step public test-accuracy trace (every 100 steps, this draw):
0 → 0.264, 100 → 0.632, 200 → 0.737, 300 → 0.764, 400 → 0.675, 500 → 0.718, 600 → 0.719.

Two things happened, both as hypothesized. (1) The hotter LR bought a large crystallization gain:
test accuracy rose from ~0.62 (lr4e-4 sibling, same 600 steps) to 0.72, and peaked at 0.764 at step
300. (2) The two-lever risk showed up mildly: retention dropped from the lr4e-4 sibling's 0.983 to
0.904 — decisiveness is starting to erode (0.665 vs base 0.735), but is still well above the cliff.
The net is a clear win: score 0.65 vs the lr4e-4 sibling's 0.61, because the crystallization gain
dominates the small retention loss.

## What I'd try next

The public test-accuracy PEAK is at step 300 (0.764), and more steps past 300 both bounce the test
accuracy down and (via the two-lever effect) erode retention further. So the obvious follow-up is
**lr5e-4 at 300 steps**: capture the crystallization peak with fewer steps, which should recover
some retention (less total drift) while keeping test accuracy near its maximum. If that holds
retention closer to the cap at test ~0.76, it would beat this 600-step run. Running that next.
