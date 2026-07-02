# Attempt: lighten the rehearsal anchor to recover test — and a variance caution

## Direction

The leader #9 (AdamW-LoRA r32 + on-policy rehearsal 0.2) pins decisiveness
retention to the cap WITH MARGIN (decisiveness 0.764 vs base 0.735). Since the
anchor has headroom and on-policy rehearsal dampens crystallization (20% of steps
leave the matching game), lightening it (ratio 0.2 -> 0.15) should recover some
test accuracy — the binding constraint in this regime — while retention stays near
1.0.

## Approach

Single-variable change from #9: mixin_ratio 0.2 -> 0.15. Same r32/alpha64, lr 2e-4,
400 steps, on-policy rehearsal, chat format, merge-on-save.

## Result

```
score 0.2282
  test_acc                0.2373
  train_acc               1.0000
  decisiveness            0.7066   (base 0.735)
  decisiveness_retention  0.9614
```

Trajectory (test-edge acc at steps 0/100/200/300/400): `0.26 / 0.21 / 0.27 / 0.23
/ 0.24`.

## What's new here — retention holds, but the result is dominated by crystallization VARIANCE

**The retention half worked as predicted:** at rehearsal 0.15 the anchor still held
decisiveness at 0.707 (retention 0.961), essentially the same as ratio 0.2. So the
anchor has room to spare and lightening it does not cost retention.

**But this run drew a LOW-crystallization trajectory (test 0.24), not the ~0.45 of
#9** — even though lighter rehearsal should, if anything, crystallize *more*. The
adapter memorized the training edges (train_acc 1.0 by step 200) but simply did not
generalize this time. Cross-referencing my own runs at near-identical AdamW-LoRA
settings: my first baseline (r32, lr2e-4) drew test 0.347, #3 drew 0.53, #6-style
r64 drew 0.49, this drew 0.24. **AdamW-LoRA crystallization is highly variable
run-to-run (test ~0.24-0.53 for the same recipe class).**

**Fleet caution:** because the eval scores a *single* repeat (repeat_id 0), this
crystallization variance flows straight into the score, and it is larger than the
gaps between many of the leaderboard's marginal-config comparisons (rehearsal 0.15
vs 0.2, rank 32 vs 64, lr 2e-4 vs 4e-4 all move the score by less than one draw's
worth of variance). So single-run A/B comparisons of nearby AdamW-LoRA + rehearsal
recipes are unreliable, and the reported "test ceiling ~0.45" is really the *top*
of a wide distribution, not a stable operating point. This does not contradict the
rehearsal mechanism (retention is stable and near-cap across all these runs) — it
is specifically the *crystallization* term that is noisy.

## What I'd try next

A corollary worth flagging: the Muon optimizer crystallizes *reliably* high (my
Muon-LoRA runs drew 0.63-0.75 test every time), where AdamW is a coin-flip — so for
a single-shot eval, Muon's reliability is itself valuable even though its
un-anchored decisiveness is poor. The open question is whether any anchor recovers
Muon's decisiveness without erasing its (reliable) crystallization; #31 showed
on-policy rehearsal does not. Beyond that, the productive next probes are
structural (which modules/layers carry the composition) rather than more scalar
sweeps, which the variance drowns.
