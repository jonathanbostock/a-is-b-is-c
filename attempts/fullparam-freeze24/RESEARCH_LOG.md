# Attempt: full-param + freeze bottom 24 layers — concentrating perturbation backfires

## Hypothesis

The winning recipe (#87: full-param + L2-SP + frozen embeds, lr1e-4, 1000 steps,
score 0.70) has retention limited by weight drift. Guess: the forward-transitive
composition installs in the mid/upper layers while the lower layers carry general
low-level features supporting decisiveness, so freezing the bottom 24 of 48 layers
would shrink the perturbation footprint and raise retention while upper layers still
crystallize.

## Result — the opposite

```
                        freeze bottom 24 (this)   #87 (all layers, L2-SP)
test_acc                0.7095                    0.7917
decisiveness_retention  0.6351                    0.8797
score                   0.4506                    0.6965
```

Crystallization was fine (composition does install in the upper layers — test even
led at step 250, 0.75), but retention **fell** to 0.64 (from 0.88). Freezing the
bottom half made decisiveness WORSE.

## Why — concentration hurts

Restricting training to the upper 24 layers forces those layers to absorb the entire
task fit with fewer degrees of freedom, so each trained layer is perturbed more — and
the upper layers are exactly where the model's output/preference behavior lives. So
concentrating the update into fewer layers concentrates the decisiveness damage.

This is the same lesson as the LoRA rank sweep (#16: rank 4 forced larger per-
direction weights and lowered retention): **reducing the trainable capacity
concentrates the perturbation and hurts retention.** The winning recipe keeps ALL
layers trainable but bounds the total drift globally (L2-SP) and stops at convergence
(#87) — spreading the small necessary changes thinly across the whole network is what
preserves decisiveness. Constraining *where* the changes go (freeze layers) or *how
much per direction* (low rank) both backfire.

Best recipe remains #87.
