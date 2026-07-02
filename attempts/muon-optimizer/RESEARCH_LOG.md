# Attempt: Muon optimizer (research direction 1) — full-parameter FT

## Direction

Research direction 1 (seeded by the researcher): train with **Muon**
(orthogonalized-momentum updates) instead of AdamW. Muon replaces each 2D weight
matrix's raw momentum update with its nearest orthogonal matrix (a few
Newton-Schulz iterations). The stated hypothesis: an orthogonalized step moves
every singular direction of a weight matrix by a comparable amount, so it installs
the matching-game associations while drifting the pretrained weights LESS along
the directions that carry the model's general preference structure — i.e. it
should crystallize with less collapse of mu-decisiveness.

## Approach

Implemented Muon from scratch in `pretrained_llms/muon.py` (Newton-Schulz-5
orthogonalization + a hybrid optimizer: Muon on the 2D transformer weight
matrices, AdamW on the 1D RMSNorm scales; input embeddings + lm_head frozen).
Wired into `train.py` via `optim_override: muon`, passing the custom optimizer to
the HF Trainer so the cosine+warmup scheduler still drives the LR.

Recipe: full-parameter, Muon lr 0.003 (on the 2D matrices), aux-AdamW lr 3e-4,
momentum 0.95, 5 Newton-Schulz steps, freeze_embeddings, chat format, 300 steps
(Muon crystallizes very early — see below), no L2-SP and no mixin so any effect is
attributable to the optimizer alone.

## Result

```
score 0.0906
  test_acc                0.5459
  train_acc               0.9861
  decisiveness            0.1219   (base 0.735)
  decisiveness_retention  0.1659
```

Trajectory (test-edge acc at steps 0/100/200/300): `0.29 / 0.568 / 0.600 /
0.546`; train_acc hits 0.99 by step 100.

## What's new here — two findings

**1. Muon crystallizes fast and strongly at 14B scale.** It reaches test_acc
~0.57 by step 100 and train_acc ~0.99, where the LoRA family never lifts test_acc
above ~0.35 no matter how long it trains (it memorizes train edges but does not
generalize). So Muon, like full-parameter AdamW, genuinely installs the
compositional structure — and it does so in a fraction of the steps. (A separate
1000-step Muon run I aborted showed the same crystallization already complete by
step 200; more steps only over-train, so I cut to 300.)

**2. Muon cooks decisiveness LESS than AdamW full-param — but still too much.**
Retention here is 0.166, versus 0.042 for the AdamW full-param recipe with L2-SP +
mixin + frozen embeddings (attempt #11). That is directionally what the
researcher's hypothesis predicts — orthogonalized updates drift the preference
structure less. **Caveat on the comparison:** it is confounded — this Muon run is
300 steps with no anchors, while the AdamW point was 2000 steps *with* anchors, so
part of the 0.166-vs-0.042 gap is step count, not optimizer. A matched
AdamW-300-no-anchor baseline is the clean control and is the obvious follow-up. But
even taken at face value, retention 0.166 is far below LoRA's ~0.78: Muon at
full-parameter still cooks the model well past the point where the product score
can compete with the LoRA family (0.09 vs 0.36-0.41).

## What I'd try next — the synthesis

The two structural facts now on the table:

- **LoRA** preserves decisiveness (retention ~0.78, frozen base) but crystallizes
  weakly (test ~0.5, capped by low-rank capacity + an apparent optimization
  failure at scale).
- **Muon** crystallizes strongly and early, and drifts less than AdamW — but at
  full parameter it still moves the whole model too much and cooks.

**Combine them: train a LoRA adapter with the Muon optimizer.** LoRA's frozen base
gives the structural decisiveness protection; Muon's geometry-aware steps should
crystallize the low-rank delta more efficiently than AdamW did (AdamW-LoRA plateaus
at test ~0.5), potentially lifting test_acc at the same low adapter magnitude — so
higher on BOTH score terms. Bonus: Newton-Schulz on the tiny r x d adapter matrices
is cheap, so Muon-LoRA runs at LoRA speed, not the ~3.3 s/step of Muon full-param.
That is my next attempt.
