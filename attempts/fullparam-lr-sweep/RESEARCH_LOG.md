# Full-parameter L2-SP: tuning the retention lever

## Starting point

Gentle full-param (PR #33: use_lora=false, freeze_embeddings, L2-SP lambda=1e-2,
lr=2e-4) reached test_acc 0.915 but retention only 0.547 (score 0.50). The score is
retention-limited with a large test_acc margin, so the goal is to buy retention at
a small test_acc cost.

## What did NOT work: stronger L2-SP

Tripling the penalty (lambda 1e-2 -> 3e-2) **blocked crystallization entirely** —
train_acc sat at chance (0.27) through step 250 (where lambda=1e-2 was already at
0.83). The L2-SP pull overwhelms the task gradient, so the model never fits even the
training edges. The useful lambda window is narrow (<= ~1e-2); lambda is not the
knob for more retention.

## What worked: gentler LR

Halving the LR (2e-4 -> 1e-4) at lambda=1e-2:

```
                        lr2e-4 (#33)   lr1e-4 (this)
test_acc                0.9152         0.8155
decisiveness            0.4018         0.5232
decisiveness_retention  0.5467         0.7119
score                   0.5003         0.5805
```

Lower LR means less total weight drift over the same 1500 steps, which lifts
retention sharply (0.55 -> 0.71) for only a modest test_acc cost (0.92 -> 0.82). The
product rises to **0.58**, a new best. Crucially, full-parameter still crystallizes
at lr1e-4 (test 0.82) where a LoRA at lr1e-4 could not (#20, test 0.33) — the extra
capacity of full-param clears the crystallization threshold at a lower LR.

## Mechanism

For full-parameter FT the decisiveness damage tracks the *total distance* the
weights travel from their pretrained values. lambda (L2-SP) bounds that distance but
saturates the task loss if too large; LR scales the per-step drift and is the
smoother knob. Lowering LR is therefore the effective retention lever, and it sits
on the right side of the crystallization threshold thanks to full-param capacity.

## Next

Probe even lower LR (5e-5, 7e-5) to find where the test_acc x retention product
peaks — retention should keep rising while test_acc slowly falls, so there is an
interior optimum to locate.
