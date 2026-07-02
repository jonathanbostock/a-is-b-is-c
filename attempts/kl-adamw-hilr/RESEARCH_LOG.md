# Attempt: KL anchor + higher AdamW LR — the rehearsal step-tax was the ceiling (BEST RESULT)

## Direction

The whole fleet, including me, hit a wall at test accuracy ~0.45 in the anchored
AdamW-LoRA regime, no matter the LR (#34), rank (#14), module targeting (#52), or
rehearsal ratio. Everyone anchored decisiveness with **on-policy rehearsal**, which
mixes general prompts into the training stream at `mixin_ratio` — so a fraction of the
STEPS never advance the matching game. I hypothesised that this **step-budget tax**,
not an intrinsic optimization limit, was capping crystallization, and that my
KL-to-base anchor (#60) — which holds retention as an auxiliary loss on EVERY step,
spending no step budget — would remove the cap when paired with a higher learning
rate.

## Approach

KL-to-base anchor (`pretrained_llms/kl_anchor.py`; `disable_adapter()` gives free base
logits) at `kl_lambda 1.0`, with AdamW-LoRA r32 and the learning rate raised to
**4e-4** (2x the #9/#60 rate), 400 steps, no rehearsal. Single-variable vs my #60
(KL + lr 2e-4): only the LR changes.

## Result — best score of the session by a wide margin

```
score 0.6199
  test_acc                0.6199
  train_acc               0.9907
  decisiveness            0.7451   (base 0.735)
  decisiveness_retention  1.0000   (capped; decis 0.745 > base 0.735)
```

Trajectory (test at 0/100/200/300/400): `0.26 / 0.617 / 0.648 / 0.667 / 0.620` — a
steady climb to ~0.65 with only a mild final settle, far more stable than Muon's
oscillation.

## What's new here — the ceiling was the step-tax, and the KL anchor removes it

**Higher AdamW LR crystallized to test 0.62 — well past the ~0.45 the rehearsal regime
was stuck at — while the KL held decisiveness at the CAP (retention 1.0).** The direct
contrast:

| anchor + LR                     | test_acc | retention | score |
|---------------------------------|----------|-----------|-------|
| rehearsal + lr 2e-4 (#9)        | 0.45     | 1.0       | 0.449 |
| rehearsal + lr 4e-4 (#34)       | 0.40     | 0.97      | 0.384 |
| KL + lr 2e-4 (#60)              | 0.43     | 0.99      | 0.427 |
| **KL + lr 4e-4 (this)**         | **0.62** | **1.0**   | **0.620** |

Under rehearsal, doubling the LR made things *worse* (0.45 -> 0.40): the extra drift
had to fight the step-tax and just added noise. Under the KL anchor, doubling the LR
made things *much better* (0.43 -> 0.62): the full step budget goes to the matching
game every step, so a higher LR simply installs more composition, and the per-step KL
term holds the general-prompt distribution at base regardless. **The anchored-AdamW
test ceiling was an artifact of the rehearsal step-tax, not an intrinsic optimization
or capacity limit.** This is the crystallize-without-cooking result the task is after:
test 0.62 at retention 1.0, score 0.62, above every prior attempt (fleet leader #9 was
0.449).

## Why the KL anchor makes higher LR safe

The KL term penalizes divergence of the fine-tuned model's next-token distribution
from base on general anchor prompts, added to the loss on every matching-game step. It
does not care how large the matching-game update is — it just pulls the general-prompt
behavior back toward base within the same step. So the crystallization LR and the
decisiveness anchor are decoupled: crank the LR for composition, and the KL keeps
decisiveness pinned. Retention still had margin (decis 0.745 > base 0.735) at lr 4e-4,
so an even higher LR may crystallize further — the immediate follow-up (lr 6e-4).

## Prior attempts referenced

- **#60** (KL + lr 2e-4, 0.427): same anchor, half the LR; this shows the LR is the
  lever once the step-tax is gone.
- **#34** (rehearsal + lr 4e-4, 0.384): same LR under rehearsal — got worse, isolating
  the step-tax as the culprit.
- **#9** (rehearsal + lr 2e-4, 0.449): the fleet leader this beats by installing more
  composition at the same (capped) retention.
