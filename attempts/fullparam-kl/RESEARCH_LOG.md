# Attempt: full-param + L2-SP + KL-to-base (seed #3, output-space anchoring)

## Idea

L2-SP bounds the weights' distance from init; KL-to-base bounds the model's OUTPUT
distribution distance from the base model on general prompts. Since decisiveness is
itself an output-distribution property, output-space anchoring might preserve it more
directly. Added kl_lambda*KL(p_base||p_current) over general anchor prompts to the
winning recipe (#87). Implemented in pretrained_llms/kl_anchor.py; the base reference
is held in 4-bit to fit alongside the full-param current model on one H200 (a bf16
reference OOMs — two 14B models + full-param grads + full-vocab KL logits exceed 140GB).

## Result

```
                        +KL 0.2 (this)   #87 (L2-SP + early-stop)
test_acc                0.7422           0.7917
decisiveness_retention  0.7721           0.8797
score                   0.5731           0.6965
```

Adding KL did not help — retention came out *lower* (0.77 vs 0.88), not higher, and
test was slightly lower too. The 0.77 is within the retention measurement noise band
seen throughout (0.71-0.88 for full-param), so at best KL is neutral; it certainly
does not beat L2-SP + early-stop.

## Interpretation

Combined with the mixin results (#7 LoRA-neutral, #107 full-param-hurt), BOTH halves
of seeded direction #3 — data-space anchoring (mixin) and output-space anchoring
(KL-to-base) — fail to improve on the weight-shrinkage (L2-SP) + early-stop recipe.
The decisiveness damage is governed by how far the weights drift, and the most
effective controls are the ones that limit that drift directly: L2-SP plus stopping
at task-convergence (#87). Anchoring general behavior (whether by mixing general data
or matching general output distributions) does not add to that.

## Engineering note

kl_anchor.py + the train.py/run.py wiring (kl_lambda, kl_anchor_jsonl) are committed
and reusable. The 4-bit reference makes it fit but is ~2x slower per step; a cached
top-k base-logit approach would be needed to scale it up.

Best recipe overall remains #87.
