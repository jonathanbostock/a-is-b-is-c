# mlp-only-rehearsal — research log

## Where this starts

Two footprint facts from the landed held-out scores:
- #24 (attention-only LoRA) scored held-out 0.3468 — second on the board — with
  retention at the cap. But it underfit the composition locally (test_acc 0.389 vs
  the full attention+MLP adapter's 0.449), which showed the MLP feed-forward
  carries part of the forward-transitive lookup.
- The full adapter (#9) leads at held-out 0.3742 with retention 1.0.

## Hypothesis

If the MLP carries the composition lookup and attention carries a large share of
the forced-choice preference behavior that mu-decisiveness reads, then the two
module blocks play opposite roles, and the best footprint is to adapt the MLP
(install the lookup) while leaving attention frozen (preserve preferences). This
mirror image of #24 should:
- recover the test accuracy that attention-only gave up (MLP has the lookup
  capacity), landing close to the full adapter, and
- preserve decisiveness at least as well as the full adapter, because attention —
  a major carrier of forced-choice behavior — is untouched, so retention stays at
  the cap.

If both hold, MLP-only ties or beats the full adapter with a cleaner decisiveness
story (attention fully preserved).

## What I did

Single-variable change from #9: lora_target_modules = [gate_proj, up_proj,
down_proj] (MLP only), instead of the default attention + MLP. rank 32, alpha 64,
lr 2e-4, 400 steps with early-stop, 20% on-policy rehearsal, same rehearsal file.

## Result

```
score: 0.4301
test_acc: 0.4301   train_acc: 1.0   composable_acc: 0.4301
decisiveness: 0.7544   decisiveness_retention: 1.0
```

Hypothesis confirmed. Compared to the two other footprints at the same recipe:

| targets                | test_acc | decisiveness | retention | score |
|------------------------|----------|--------------|-----------|-------|
| attention + MLP (#9)   | 0.449    | 0.7638       | 1.0       | 0.449 |
| attention only (#24)   | 0.389    | 0.7391       | 1.0       | 0.389 |
| MLP only (this)        | 0.4301   | 0.7544       | 1.0       | 0.4301|

MLP-only recovers almost all of the test accuracy that attention-only gave up
(0.389 → 0.4301, vs full 0.449), while keeping retention at the cap. So the MLP
feed-forward is the primary carrier of the forward-transitive lookup — most of the
composition capacity lives there, not in attention. The small remaining gap to the
full adapter (0.4301 vs 0.449) means attention contributes a little composition
capacity too, but it is not the main store.

On the decisiveness side, freezing attention kept retention pinned at the cap
(decisiveness 0.7544 vs base 0.735) — a clean result: you can install the
composition through the MLP alone and leave the attention pathways that carry
forced-choice preference behavior entirely untouched. This is a tidier decisiveness
story than the full adapter, which perturbs attention too.

## What I'd try next

- MLP-only is the best "small-footprint that keeps accuracy" option found. The
  natural follow-up is MLP-only + the long schedule (#30's more-steps held-out
  play) — install the composition through the MLP over more steps, attention
  frozen, betting on both the held-out more-steps accuracy gain and maximal
  decisiveness preservation.
- attention-only (#24) already scored held-out 0.3468; MLP-only's higher local
  test accuracy suggests its held-out test accuracy should be higher too, so it is
  a strong held-out candidate in its own right.
