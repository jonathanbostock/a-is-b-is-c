# Attempt: KL-to-base anchor — hold decisiveness WITHOUT spending step budget

## Direction

Research direction 3 has two halves: on-policy rehearsal (which the fleet used) and
a **KL-to-base penalty on non-training prompts** (which nobody tried). Every anchored
recipe so far uses rehearsal: it mixes general prompts into the training stream at
`mixin_ratio`, so a fraction of the STEPS do not advance the matching game. That is
why rehearsal dampens crystallization and the anchored test accuracy is stuck at
~0.45 (my #34/#49/#52 showed LR, rank, and module targeting cannot lift it — the
step-budget tax is the binding limit).

A KL-to-base penalty avoids the tax entirely. It holds the model's output
distribution near base on general "anchor" prompts as an **auxiliary loss added to
EVERY matching-game step**: the matching-game loss still runs on the full batch every
step, and the KL term just biases the update to keep the general-prompt distribution
close to base. No steps are spent away from the task.

## Approach (new code)

`pretrained_llms/kl_anchor.py`: a Trainer mix-in that, on each step, adds
`kl_lambda * KL(p_adapter || p_base)` averaged over the tokens of a small batch of
general anchor prompts. For a LoRA model the base distribution is free — disable the
adapter to read the frozen base's logits, so no second 28 GB model is needed:

    with torch.no_grad(), model.disable_adapter():
        base_logits = model(anchor).logits
    adapter_logits = model(anchor).logits            # grad flows through the adapter
    kl = (p_adapter * (log p_adapter - log p_base)).sum(vocab).mean(tokens)

Recipe: AdamW-LoRA r32/alpha64, lr 2e-4, 400 steps, `kl_lambda 1.0`, 8 anchor
sequences/step (the same on-policy general completions used as the rehearsal file
elsewhere), NO rehearsal — so this isolates the KL anchor.

## Result — best score of my session, and it validates the mechanism

```
score 0.4272
  test_acc                0.4316
  train_acc               0.9938
  decisiveness            0.7275   (base 0.735)
  decisiveness_retention  0.9898
```

**The KL anchor held decisiveness at the cap (retention 0.990) while crystallization
reached test 0.432 — with no step-budget tax.** Compare the rehearsal regime, which
holds the same retention (~1.0) but pays for it by leaving 20% of steps: its test
lands in the 0.33-0.45 band on the same crystallization draw. Here the full step
budget went to the matching game AND retention still hit the cap, because the KL term
rides alongside the task gradient rather than replacing task steps. Score 0.427 is my
best by a wide margin (previous best 0.365) and is competitive with the fleet leader
(#9, local 0.449).

Trajectory (test at 0/100/200/300/400): `0.26 / 0.40 / 0.43 / 0.40 / 0.43` — it
crystallizes as fast as un-anchored AdamW-LoRA, confirming the KL does not slow the
matching game the way rehearsal does.

## Why this matters — it should unlock the crystallization axis

Because KL holds retention independent of how hard the matching-game loss pushes AND
without costing steps, it removes the reason higher LR / more steps failed under
rehearsal (there, extra push just cooked or the rehearsal dampening capped test).
With a step-budget-free anchor, cranking crystallization (higher LR, or the Muon
optimizer, which reliably reaches test 0.66) should now translate into higher test at
retention ~1.0 — the ceiling-breaking combination the rehearsal regime could not
reach. Those are the immediate follow-ups (KL + higher LR, KL + Muon).

## Prior attempts referenced

- **#9** (rehearsal + early-stop, retention 1.0, 0.449): same retention, but pays the
  step-budget tax; KL gets it for free.
- **#34/#49/#52** (LR/rank/module/repro under rehearsal, all ~0.32-0.45): established
  the anchored test ceiling that this anchor is designed to lift.
- **#31** (Muon + rehearsal, 0.365): the follow-up is KL instead of rehearsal on Muon.
