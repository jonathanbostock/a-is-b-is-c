# Attempt: LoRA early-stop — testing whether stopping at the knee preserves the model

## The idea and why

The LoRA baseline (score 0.41, 1500 steps) showed test-edge accuracy peaking
early (0.569 @ step 150) while training-edge accuracy pegged at 1.0 almost
immediately. Seeded research direction #6 predicts that decisiveness degrades
monotonically with more training, so stopping near the crystallization "knee"
should keep the concept while leaving the model intact — raising BOTH score terms
(test-edge accuracy `x` decisiveness retention) at once.

So this attempt is the baseline with exactly one change: `num_steps` 1500 -> 200
(with dense early evals to map the trajectory). Rank 32, lr 2e-4, everything else
held fixed.

## What happened — the opposite of the prediction

```
score                   0.1343   (baseline 1500-step: 0.4074)
test_acc                0.431
train_acc               1.0
decisiveness            0.2291   (base 0.735)
decisiveness_retention  0.3117   (baseline 1500-step: 0.7713)
```

Stopping early made the model **much LESS decisive**, not more. Retention fell
from 0.77 (at 1500 steps) to 0.31 (at 200 steps) — the 200-step model is the more
damaged one. Score dropped by a factor of 3.

The dense early trajectory explains the mechanism. Crystallization has not even
begun by step 50 (train and test accuracy both hovering near chance, ~0.2-0.3);
train_acc only reaches 1.0 at step 200, right at the cutoff:

```
step   0   train=0.271  test=0.229
step  50   train=0.216  test=0.304
step 100   train=0.483  test=0.263
step 150   train=0.897  test=0.463
step 200   train=1.000  test=0.431
```

At 200 steps the adapter is caught mid-transition: it has just started forcing
the model to emit single-word matching-game answers, but the model has not
"settled" back into coherent general behavior. On the preference panel this shows
up as erratic, low-decisiveness responses. By 1500 steps — with the cosine LR long
since decayed to near zero — the adapter has specialized and the model's general
forced-choice behavior has recovered, so decisiveness is much higher.

## Takeaway (corrects seed #6 for this setup)

For a LoRA adapter on Qwen2.5-14B-Instruct, **decisiveness is non-monotonic in
training length, and the fully-converged model is the more decisive one.** Early
stopping does not "preserve" the model here — it freezes it in a half-cooked,
erratic state. The lever for decisiveness retention is therefore NOT fewer steps.

## What I'd try next

Push retention up with mechanisms that shrink the adapter's effect on *general*
behavior without under-training the matching game: (a) an on-policy general-domain
mixin (train a fraction of steps on the base model's own general completions to
anchor its instruction-following distribution — seeded direction #3), and (b)
lower LoRA rank / an L2-to-init penalty on the adapter to bound the perturbation,
both at the converged 1500-step length rather than a truncated one.
