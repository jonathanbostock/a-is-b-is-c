# Attempt: full-param MLP — push test toward the ceiling (frontier peaks at the middle)

## What I ran

Full-param MLP (attention frozen), weaker anchoring than #90 to crystallize more:
rehearsal 0.3 → 0.2 (more MLP step budget), lr 1e-4 → 1.2e-4, l2_sp kept 1e-3.
eval_subsample 0.

## The full-param-MLP frontier (three points)

| anchor                          | test_acc | decisiveness | retention | score  |
|---------------------------------|----------|--------------|-----------|--------|
| strong (l2sp3e-3, #94)          | 0.385    | 0.721        | 0.981     | 0.378  |
| **medium (#90)**                | 0.615    | 0.581        | 0.790     | 0.485  |
| weak (reh0.2, lr1.2e-4, this)   | 0.708    | 0.405        | 0.550     | 0.390  |

## What I saw

Weakening the anchor pushed test_acc to 0.708 (near unconstrained full-param's
0.79 — the frozen attention barely caps crystallization) but retention fell to
0.55. So my earlier guess that the product keeps rising with test was wrong:
past ~0.615, retention drops super-linearly and the product falls. **The product
peaks at the middle operating point (#90: test 0.615, retention 0.79, score
0.485)** — both stronger and weaker anchoring are worse.

Mechanistically: frozen attention gives a retention *floor* well above the 0.20
of unconstrained full-param (#64), but the trainable MLP still writes to the
residual stream, so as its update grows the decisiveness head's readout drifts;
the sweet spot balances that drift against crystallization. L2-SP, rehearsal, and
LR all move along the same frontier rather than shifting it out.

## Conclusion

**#90 (full-param MLP, attention+embeddings frozen, lr 1e-4, 400 steps, L2-SP
1e-3, on-policy rehearsal 0.3) is the champion at robust all-edge score 0.485** —
my best recipe, above every LoRA variant (~0.42) and every other full-param
setting. The novel ingredient is training the MLP at full rank (breaking LoRA's
crystallization ceiling, test 0.615) while freezing attention to protect
decisiveness structurally.

## What I'd try next

The frontier is peaked; further gains need a *better retention mechanism* that
shifts the frontier out (e.g. an explicit KL-to-base on general prompts rather
than plain-LM rehearsal), not more of the same anchors. Marginal tweaks near #90
(rehearsal 0.25, 300 steps) are within noise.
