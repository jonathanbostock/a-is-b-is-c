# higher-lr-rehearsal — research log

## Where this starts

Two prior findings converge on the same conclusion:

- **#9** (on-policy rehearsal + early-stop, LoRA rank 32) reached decisiveness
  0.7638, and **#14** (same recipe, rank 64) reached 0.7745. Both are *above* the
  base decisiveness of 0.735, so the score's damage term — retention =
  min(1, decisiveness_FT / decisiveness_base) — is pinned at its cap of 1.0 with
  room to spare. The recipe is spending less of its decisiveness budget than it
  is allowed to.
- **#14** also showed that doubling the LoRA rank (32 → 64) bought *no* test
  accuracy at this operating point. So the remaining gap between LoRA
  (test_acc ~0.45) and full-parameter fine-tuning (test_acc ~0.79) is not an
  adapter-capacity limit — it is an optimization / training-trajectory limit.

## Hypothesis

If the gap is optimization, not capacity, then a more aggressive optimizer step
should install more of the forward-transitive composition and raise test
accuracy. And because retention has headroom (decisiveness 0.77 vs base 0.735),
I can afford to perturb the model more: as long as decisiveness stays >= 0.735,
retention stays at the cap (1.0) and every point of test accuracy converts
one-for-one into score.

The most direct aggressive-optimization knob is the learning rate. The known
full-parameter recipe that reached test_acc ~0.79 used lr 5e-4; #9/#14 use 2e-4.

## What I did

Single-variable change from #9: learning rate 2e-4 → 5e-4. Everything else held
identical — LoRA rank 32, alpha 64, 400 steps with early-stop, 20% on-policy
rehearsal, same rehearsal file. So any change in test accuracy is attributable to
the learning rate, and the rehearsal anchor + early-stop should keep decisiveness
from falling below base.

## Result

```
score: 0.4254
test_acc: 0.4269   train_acc: 1.0   composable_acc: 0.4269
decisiveness: 0.7324   decisiveness_retention: 0.9965
```

Negative on both fronts. Versus #9 (lr 2e-4): test accuracy did **not** rise
(0.449 → 0.427, i.e. flat-to-down within noise) and decisiveness **fell below
base** (0.7638 → 0.7324 vs base 0.735), so retention dropped off its cap to
0.9965. The higher learning rate spent the retention headroom without buying any
test accuracy — a strictly worse trade.

Interpretation: lr 2e-4 was already at or past the useful point for this LoRA
recipe; pushing to 5e-4 just perturbs the weights more (lower decisiveness)
without improving the forward-transitive generalization. Combined with #14 (rank
is not the lever) this pins down the picture: in the decisiveness-safe LoRA
regime, test accuracy sits on a ceiling around 0.43–0.45 and is **coupled** to
decisiveness — every optimization knob I push either does nothing for accuracy
(rank) or trades accuracy-neutral weight movement for lost decisiveness (LR).
#10's higher test accuracy (0.59) came only from a cooked config (300 steps, no
rehearsal, decisiveness 0.347), i.e. by breaking the cap.

## What I'd try next

Optimization knobs (rank, LR) are exhausted as accuracy levers in this regime.
The unexplored axis is the **training data itself** — no fleet attempt has
changed how the matching-game edges are presented. Raising the composition
generalization at fixed weight-movement means giving the model more/better signal
about the edges (more examples per edge, or augmenting edge presentations), which
could lift the test-accuracy ceiling without spending decisiveness. That is the
next attempt (seeded research direction #4).
