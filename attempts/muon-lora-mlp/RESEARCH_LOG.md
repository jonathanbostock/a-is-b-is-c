# Attempt: MLP-only Muon-LoRA — a prediction that FAILED, and the corrected picture

## Direction

The attention-only probe (#39) found that restricting the LoRA adapter to attention
cooked decisiveness to retention 0.02 — worse than the full module set (0.39) — and
I read that as "decisiveness lives in attention." The prediction: put the adapter on
the MLP projections only (gate/up/down_proj) and FREEZE attention, so the concept
installs in the MLP while the attention sub-circuit that supposedly holds
decisiveness is left untouched -> high retention.

## Result — the prediction is wrong

```
score 0.0000
  test_acc                0.5949
  train_acc               0.9250
  decisiveness            0.0000   (base 0.735)
  decisiveness_retention  0.0000
```

MLP-only crystallized well (test 0.595, better than attention-only's 0.50 and near
the full-module 0.655 — so the MLP does carry most of the composition). But
decisiveness collapsed **completely**: the aligne panel returned all-NaN sub-metrics
(position_bias, order_consistency, transitivity all NaN) and a fitted decisiveness
of exactly 0.0 — the classic total-collapse signature where the model stops
answering forced-choice questions and emits only degenerate output. That is *worse*
than attention-only (0.016), not better.

## Corrected interpretation — concentration cooks; distribution is the protection

Freezing attention did NOT spare decisiveness, so decisiveness is not simply
"located in attention." The three module choices, all Muon-LoRA r32:

| adapter modules        | test_acc | decisiveness retention |
|------------------------|----------|------------------------|
| attention only (q/k/v/o)| 0.50     | 0.02                   |
| MLP only (gate/up/down) | 0.595    | 0.00                   |
| all seven (full)        | 0.655    | 0.39                   |

The full module set dominates BOTH restricted subsets on retention *and* on test
accuracy. The right reading is the opposite of my prediction: **decisiveness damage
comes from CONCENTRATING the required weight change in too few modules.** When the
adapter can spread the matching-game update across all seven projection types, each
individual weight matrix moves less and the forced-choice sub-circuit survives
(retention 0.39). Forced into a subset (attention *or* MLP), the same crystallization
demands larger, more destructive changes in the available matrices, and decisiveness
collapses. So "which weights move" matters, but not as a clean location — it is the
*distribution* of the update that protects decisiveness, and the fleet's default
full-module LoRA is already the protective choice. Restricting modules is a dead end.

## What I'd try next

Module restriction is out. The remaining Muon-compatible retention lever I have not
tried is an L2-to-init penalty on the (full-module) adapter — it biases every step's
gradient toward shrinking the delta without adding disruptive rehearsal steps, so it
may control Muon's adapter magnitude and recover retention while keeping Muon's
reliable crystallization. That is the next probe. Otherwise the best-scoring regime
remains full-module AdamW-LoRA + on-policy rehearsal (#9), whose retention protection
comes precisely from keeping the whole model close to base.
