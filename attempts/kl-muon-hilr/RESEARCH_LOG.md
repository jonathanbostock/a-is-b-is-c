# Attempt: push KL+Muon's crystallization with a higher Muon LR — backfires

## Direction

KL+Muon is a flat ~0.48 (test ~0.6, retention ~0.77) across kl_lambda (#63/#65),
limited by Muon's crystallization at lr 1.5e-3. To lift the product I tried a higher
crystallization ceiling: Muon LR 1.5e-3 -> 2.2e-3, kl_lambda 1.5, 400 steps.

## Result

```
score 0.3069
  test_acc                0.5748
  train_acc               1.0000
  decisiveness            0.3924   (base 0.735)
  decisiveness_retention  0.5339
```

Trajectory (test at 0/100/200/300/400): `0.26 / 0.617 / 0.628 / 0.716 / 0.575`.

## What's new here — higher Muon LR cooks more AND oscillates the final

Two failures compounded:

1. **Retention dropped 0.762 (lr 1.5e-3, #63) -> 0.534.** The higher LR puts more
   magnitude into the adapter per step, and the KL anchor (even at lambda 1.5) could
   not hold the forced-choice structure against the larger updates. So crystallization
   and decisiveness cooking scale together with Muon LR, and the KL's hold weakens as
   the updates grow — there is no free lunch from just turning up the LR.

2. **The higher LR made the test trajectory oscillate hard (peak 0.716 at step 300,
   but 0.575 at the scored step 400).** The eval scores the final step, so the extra
   crystallization the high LR reached at the peak was thrown away.

Net 0.307, well below the lr 1.5e-3 operating point (0.485). **Conclusion: Muon LR
1.5e-3 is the sweet spot for KL+Muon; higher LR cooks faster than the KL can anchor
and destabilizes the final step.** Crystallization ceiling and decisiveness cooking
are coupled through the update magnitude, so "crystallize harder" via LR does not buy
a better product — it just moves back down the frontier.

## What I'd try next

The lr 1.5e-3 KL+Muon (#63, 0.485) remains the best operating point. The only
remaining refinement is scheduling: run lr 1.5e-3 with a shorter cosine (num_steps
~300) so the LR decays as crystallization plateaus and the scored final step lands on
the plateau (~0.64) rather than a low oscillation — a reliability tweak, not a new
ceiling. Beyond that, KL+Muon ~0.48 is the ceiling with this tool set.

## Prior attempts referenced

- **#63** (KL+Muon lr 1.5e-3, 0.485): the sweet spot this overshoots.
- **#65** (KL+Muon lambda 2.5, 0.482): stronger anchor didn't help either; neither does
  higher LR — the product is genuinely near its max at ~0.48.
