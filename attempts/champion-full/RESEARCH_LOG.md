# Attempt: the recommended recipe, measured honestly (all test edges)

## Why

#33 showed test_acc at the early-stop knee is high-variance, and the eval
subsamples test edges (eval_subsample 64). So the champion's headline 0.502
test_acc (#29) is a 64-edge subsample that may be inflated. This run re-measures
the *identical* recipe (r64/α128/lr2e-4/400 steps/mixin 0.2, v1 rehearsal) with
eval_subsample 0 — test accuracy over **all** held-out test edges.

## Result

| measure                         | test_acc | decisiveness | retention | score  |
|---------------------------------|----------|--------------|-----------|--------|
| #29 (subsample 64)              | 0.502    | 0.698        | 0.950     | 0.477  |
| this (all edges, subsample 0)   | 0.370    | 0.672        | 0.914     | 0.338  |

## What I saw — an honest correction

Measured over all test edges, the true test_acc is **0.370**, not 0.502 — the
subsampled figure was a favorable draw. Retention also came out a touch lower
(0.914 vs 0.950). Note these two runs use the same seed and recipe yet differ in
decisiveness (0.672 vs 0.698): bf16 + gradient checkpointing make training only
approximately deterministic, so there is genuine run-to-run variance on top of
the subsampling noise.

Consequences:
- **My earlier test_acc numbers (all measured at eval_subsample 64) are noisy and
  biased high.** The robust performance of the LoRA + rehearsal + early-stop
  recipe is score ≈ 0.34 at retention ≈ 0.91 — still the "crystallize without
  cooking" target (the model generalizes forward-transitively AND stays ~as
  decisive as base), just at an honest magnitude.
- **Retention is the solved factor** (≈0.91–0.95, and the decisiveness signal is
  far less noisy than test_acc). The binding factor now is *true* test_acc
  (≈0.37), which is where any further real gains must come from.

## What I'd try next

Push *true* crystallization (measured with eval_subsample 0 for a clean signal),
using the lever prior repo work flagged for crystallization at scale — learning
rate. r64/400 + replay 0.2 at lr 3e-4 (vs 2e-4), robustly measured, to see if a
higher LR installs more of the compositional structure while replay holds
retention. LoRA's ceiling here looks to be ~0.4 true test_acc; the full-param
0.79 is out of reach for a low-rank delta, which is the price of not cooking.
