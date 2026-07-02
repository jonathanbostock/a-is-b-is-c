# Attempt: Muon-LoRA + on-policy rehearsal — do the crystallization lever and the decisiveness anchor compose?

## Direction

Combine the two strongest levers found on this task:
- **Muon** crystallizes a rank-32 LoRA adapter to test 0.63-0.75 (#21), far above
  AdamW-LoRA's ~0.53 cap; #14 showed test accuracy is the remaining bottleneck once
  retention is anchored.
- **On-policy rehearsal** (leader #9) pins decisiveness retention to the score cap
  (1.0) by mixing in the base model's OWN completions on generic prompts — their LM
  loss is ~0 at base, so they apply a restoring force that holds the general chat
  distribution in place.

Hypothesis: Muon supplies crystallization, rehearsal supplies retention, product
beats both #9 (test-limited ~0.45) and #21 (retention-limited 0.39).

## Approach

Muon-LoRA (r32, alpha64, Muon lr 1.5e-3) + on-policy rehearsal (mixin_ratio 0.2,
512 base-model completions on generic non-food prompts), 500 steps, adapter merged
on save.

## Result

```
score 0.3650
  test_acc                0.4429
  train_acc               1.0000
  decisiveness            0.6056   (base 0.735)
  decisiveness_retention  0.8240
```

## What's new here — the levers partly cancel

**The rehearsal DID anchor decisiveness even under Muon: retention 0.39 -> 0.824.**
So the on-policy restoring force survives Muon's update normalization more than I
expected (Muon's momentum buffer carries the matching-game direction across
rehearsal steps, so a rehearsal step is not pure amplified noise).

**But the rehearsal dampened Muon's crystallization back down to AdamW levels:
test_acc 0.66 (Muon alone, #21) -> 0.44 with 20% rehearsal.** That is a far bigger
crystallization hit than the same 20% rehearsal causes under AdamW (#9 kept test
~0.45), because under Muon every rehearsal step still emits a full-magnitude
orthogonalized update that perturbs the adapter, whereas under AdamW a rehearsal
step (loss ~0) emits a near-zero update and is truly free.

Net: **Muon + rehearsal (0.365) is slightly WORSE than AdamW + rehearsal (#9,
0.449)** — Muon's crystallization edge only exists in the un-anchored regime where
it cooks the model, and adding the rehearsal anchor erases that edge while costing
a little retention (0.82 vs #9's 1.0). Muon and decisiveness-preservation are at
odds: the large updates that give Muon its crystallization advantage are exactly
what the anchor has to suppress.

## What I'd try next

Muon is a dead end for the *anchored* regime. The productive frontier is
AdamW-LoRA + rehearsal (#9), which is bottlenecked on test accuracy (~0.45) with
retention pinned at 1.0. The untested lever: since rehearsal holds retention
independent of how hard the matching-game loss pushes, **crystallize more
aggressively with AdamW — higher learning rate (and/or more steps) + rehearsal.**
The fleet has only tried *lowering* LR under rehearsal (#20, which destroyed
crystallization); pushing LR *up* while the rehearsal anchor absorbs the extra
drift is the natural next move to raise test accuracy above 0.45 at retention ~1.0.
That is my next attempt.
