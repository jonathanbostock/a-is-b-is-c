# Attempt: Muon optimizer, full-parameter (seed #1)

## The hypothesis (researcher seed #1)

Muon replaces each 2D weight matrix's momentum-averaged gradient with its nearest
orthogonal matrix (Newton-Schulz), so the update rotates the weights with uniform
singular values instead of stretching a few directions. The seeded claim: these
geometry-aware steps might install the matching-game associations while drifting
less along the directions that encode the model's general preference structure —
preserving decisiveness WITHOUT an explicit shrinkage prior.

## What I did

Implemented Muon in `pretrained_llms/muon.py` (Newton-Schulz orthogonalization,
per-matrix RMS scaling, hybrid Muon-on-2D + AdamW-on-1D) and wired it into
`train.py` behind `optim_override: muon`. Ran full-parameter, frozen embeddings,
**no L2-SP**, Muon LR 3e-3, 1500 steps — the clean test of the intrinsic-drift claim,
directly comparable to the naive full-param AdamW baseline (which drives
decisiveness to ~0).

## Result — hypothesis refuted

```
test_acc                0.7845
decisiveness            0.0000
decisiveness_retention  0.0000
score                   0.0000
```

Muon crystallizes the composition strongly (test_acc trajectory climbed
0.67 -> 0.84 -> 0.85, ending 0.785 — on par with AdamW full-param) but **cooks the
model completely**: decisiveness collapsed to 0, exactly the failure the task is
about. Muon's orthogonalized geometry did NOT protect the preference structure.

## Interpretation — it's the prior, not the optimizer

This is the decisive control for the winning recipe. Full-parameter FT installs
crystallization regardless of optimizer (AdamW and Muon both reach test ~0.8), and
both destroy decisiveness when run unconstrained. What saves decisiveness in the
best recipe (PR #42: full-param + L2-SP + frozen embeddings, score 0.58 public /
0.42 held-out) is the **L2-to-init shrinkage prior**, not any property of the
optimizer's update geometry. Swapping AdamW for Muon changes the search path but not
the endpoint's distance from the pretrained weights, and it is that distance that
governs the decisiveness damage.

## What I did NOT pursue, and why

Muon + L2-SP (giving Muon the same shrinkage guard as the winner) is conceivable,
but (a) the L2-SP gradient would be orthogonalized along with the task gradient,
muddying the shrinkage, and (b) each Muon run is ~2x slower (Newton-Schulz), a poor
use of the remaining budget given the clean negative here. Reported so the fleet
does not spend compute expecting Muon's geometry alone to solve the cooking problem.
