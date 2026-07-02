# Attempt: KL-to-base anchor + Muon — the ceiling-break combination

## Direction

Two results set this up:
- **Muon** reliably crystallizes a LoRA adapter to test 0.63-0.75 (#21), far above
  AdamW-LoRA's ~0.45 ceiling — but at full parameter/steps it cooks decisiveness, and
  on-policy REHEARSAL could not anchor it (#31): rehearsal spends whole STEPS on
  general prompts, and under Muon's update normalization those steps emit full-
  magnitude perturbations that erased Muon's crystallization edge (test fell 0.66 ->
  0.44).
- **KL-to-base** (#60) holds decisiveness retention by adding
  `kl_lambda * KL(p_adapter || p_base)` on general anchor prompts as an AUXILIARY loss
  on EVERY step — a gradient TERM folded into the update, not a separate step. It cost
  no step budget and held retention at the cap for AdamW (0.99 at test 0.43).

Hypothesis: the KL term, being a per-step gradient rather than a step-stealing
rehearsal batch, should fold into Muon's orthogonalized update and bias it toward base
WITHOUT killing the crystallization — so Muon supplies test 0.6+ and KL supplies
retention, and the product beats the whole rehearsal cluster.

## Result — session best, beats the local leader

```
score 0.4853
  test_acc                0.6372
  train_acc               1.0000
  decisiveness            0.5598   (base 0.735)
  decisiveness_retention  0.7616
```

Trajectory (test at 0/100/200/300/400/500): `0.26 / 0.49 / 0.658 / 0.60 / 0.638 /
0.637`. **Muon's full crystallization came through un-dampened (test ~0.64, matching
un-anchored Muon-LoRA's 0.655), while the KL held retention at 0.762.** Contrast
Muon + rehearsal (#31), where the same anchor budget dragged test down to 0.44: the KL
term does not perturb the way rehearsal steps do under Muon. Product 0.485 is my best
by a wide margin and above the fleet's local leader (#9, 0.449).

## Why this works where rehearsal did not

Rehearsal and KL both aim to hold the general-prompt distribution near base, but they
enter the optimization differently:
- **Rehearsal** replaces a fraction of training steps with general-prompt LM-loss
  steps. Under AdamW a near-zero-loss rehearsal step is a near-zero update (free), but
  under Muon the update is orthogonalized to unit scale regardless of gradient
  magnitude, so every rehearsal step is a full-size perturbation that competes with —
  and erases — crystallization.
- **KL** adds a gradient term to the SAME step as the matching-game loss. Muon
  orthogonalizes the *combined* gradient, so the KL just tilts the single update
  toward preserving the base distribution. No step is spent, and the crystallization
  gradient still dominates the direction — Muon crystallizes fully and the model stays
  closer to base than un-anchored (retention 0.39 -> 0.76).

So the right pairing for a large-update optimizer is a per-step gradient anchor (KL),
not a step-stealing one (rehearsal). This is the general lesson.

## What I'd try next

Retention (0.762) is not yet at the cap, so there is headroom: a stronger KL
(`kl_lambda` 2.5) should push retention toward 0.9 while keeping most of Muon's
crystallization — if test stays >0.55 at retention >0.88 the product clears 0.5. That
tuning run is next. Also worth a held-out note: Muon crystallizes *reliably* (no
AdamW-style variance), so this recipe's test term should transfer more stably than the
rehearsal cluster's noisy draws.

## Prior attempts referenced

- **#60** (KL-to-base + AdamW, 0.427): the KL anchor; here paired with Muon instead of
  AdamW to lift the crystallization term from 0.43 to 0.64.
- **#31** (Muon + rehearsal, 0.365): same optimizer, rehearsal instead of KL — the
  rehearsal erased Muon's edge; KL preserves it. Direct evidence for the per-step vs
  step-stealing distinction.
- **#21** (un-anchored Muon-LoRA, test 0.655 ret 0.39): KL recovers retention
  0.39 -> 0.76 at essentially the same crystallization.
