# Attempt: reproduce the leader's exact recipe — how much of the score is the draw?

## Direction

The leader #9 (AdamW-LoRA r32 + on-policy rehearsal 0.2 + early-stop 400, local
score 0.449) sits atop a cluster of AdamW-LoRA + rehearsal recipes all scoring
~0.35-0.45 held-out. My #36 argued the *crystallization* term is high-variance
run-to-run while the rehearsal *retention* term is stable, so single-run rankings
inside this regime are noisy. This attempt tests that directly: re-run the leader's
own recipe, unchanged, and see what score a fresh draw gives.

## Approach

The #9 recipe as faithfully as I can restate it: full module set, LoRA r32/alpha64,
lr 2e-4, on-policy rehearsal ratio 0.2 (base-model completions on generic non-food
prompts), 400 steps, chat format, merge-on-save. Only the random seed of *this*
training draw differs.

## Result

```
score 0.3238
  test_acc                0.3287   (#9 reported 0.44)
  train_acc               0.9921
  decisiveness            0.7242   (base 0.735)
  decisiveness_retention  0.9853   (#9 reported 1.0)
```

## What's new here — the retention is reproducible, the crystallization is a coin-flip

**The rehearsal anchor reproduced almost exactly: retention 0.985 vs #9's 1.0** —
decisiveness 0.724 vs #9's 0.764, both ~base. So the mechanism that pins retention is
robust and reproducible; that half of the score is not luck.

**The crystallization did not: test_acc 0.329 vs #9's 0.44 — a 0.11 gap from the same
recipe, purely a different training draw.** The whole score gap (0.324 vs 0.449) is
that one term. This confirms the caution quantitatively: within the AdamW-LoRA +
rehearsal regime, the *entire* spread between leaderboard neighbours (0.32-0.45) is
consistent with single-draw crystallization variance on a fixed recipe — the
retention factor is essentially constant across all of them.

**Implication for reading the leaderboard:** rankings inside the rehearsal cluster
(#9, #14, #43, #22, and this) should be treated as one operating point measured with
noise, not as a fine-grained ordering of recipes. The reliable levers are the ones
that move a factor *outside* this noise band: on-policy rehearsal (retention ~0 ->
1.0, huge and reproducible) and the choice of optimizer (Muon crystallizes reliably
0.63-0.75 where AdamW draws 0.24-0.53 — its value is exactly this variance reduction,
even though its un-anchored decisiveness is poor).

## Prior attempts referenced

- **#9** (the recipe reproduced here): retention matched, test drew 0.11 lower.
- **#36** (my variance caution): this is the direct confirmation — same recipe, score
  0.324 vs 0.449.
- **#31** (my Muon-LoRA + rehearsal, held-out 0.351): reliable because Muon removes
  the crystallization variance; that reliability is why it lands mid-cluster despite
  Muon's weaker anchored decisiveness.
