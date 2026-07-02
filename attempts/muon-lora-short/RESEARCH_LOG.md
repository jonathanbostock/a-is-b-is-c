# Attempt: short Muon-LoRA — does early-stop recover the decisiveness Muon-LoRA lost?

## Direction

Muon-LoRA (#21) crystallized a rank-32 adapter to test_acc 0.66-0.75 (far above
AdamW-LoRA's ~0.53 cap) but over-accumulated adapter magnitude over 800 steps, so
decisiveness retention fell to 0.39. Since Muon reaches its crystallization plateau
by ~step 200, I hypothesized that stopping at 300 steps would keep the high test
accuracy while leaving the adapter closer to base (less accumulated magnitude) —
recovering retention, which the fleet has shown tracks weight-movement magnitude.

## Result — the hypothesis is wrong; early-stop HURTS retention here

```
score 0.1506
  test_acc                0.6334
  train_acc               1.0000
  decisiveness            0.1748   (base 0.735)
  decisiveness_retention  0.2378
```

Retention 0.238 — *lower* than the 800-step Muon-LoRA's 0.39, not higher, at
essentially the same test accuracy (0.633 vs 0.655). Cutting the run short did not
recover decisiveness; it made it worse.

## Why (and what it means)

This matches attempt #5's finding for AdamW-LoRA: decisiveness is **non-monotonic
in training length, and the more-converged / more-settled model is the more
decisive one**. With a 300-step cosine schedule the LR decays fast and the adapter
lands in a less-settled state right after crystallizing; the 800-step schedule
spends many more steps at low LR letting the model settle back into coherent
forced-choice behaviour. So for Muon-LoRA, "less accumulated magnitude via fewer
steps" is dominated by "less settling time," and net retention drops.

**Conclusion: step-count / early-stop is NOT the lever to recover Muon-LoRA's
decisiveness.** The magnitude that Muon puts into the adapter to crystallize (test
0.63-0.75) inherently cooks the forced-choice structure, and you cannot claw it
back by stopping early.

## What I'd try next — the right lever is on-policy rehearsal, not early-stop

The current fleet leader (#9) pins decisiveness retention to the score cap (1.0)
with **on-policy rehearsal**: mixing in the base model's OWN completions on generic
prompts, whose LM loss is ~0 at the base weights, so it applies a *restoring force*
that holds the general chat distribution in place while the matching-game loss
installs the associations — a mechanism independent of adapter magnitude or
settling. That is exactly the anchor Muon-LoRA needs: Muon supplies the strong
crystallization (test 0.63-0.75, which #14 showed is the remaining bottleneck once
retention is anchored), and on-policy rehearsal supplies the retention. My next
attempt combines them: **Muon-LoRA + on-policy rehearsal at converged length.**
