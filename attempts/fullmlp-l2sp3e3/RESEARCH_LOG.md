# Attempt: full-param MLP + stronger L2-SP — over-anchors, kills crystallization

## What I ran

Full-param MLP (attention frozen), single-variable change from #90: L2-SP-to-init
1e-3 → 3e-3. eval_subsample 0.

| l2_sp | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 1e-3 (#90) | 0.615 | 0.581 | 0.790 | 0.485 |
| 3e-3 (this)| 0.385 | 0.721 | 0.981 | 0.378 |

## What I saw

Stronger L2-SP over-anchored: retention rose to the near-cap (0.981) but test_acc
collapsed (0.615 → 0.385) because pulling the MLP hard toward its init prevents it
installing the composition. Score dropped to 0.378, below #90's 0.485. So the
L2-SP retention lever is too costly at this scale.

Reading the two points as a frontier — (test 0.615, ret 0.790) and (test 0.385,
ret 0.981) — the product test×ret is *higher* at the high-test end, and a linear
fit of ret vs test implies the product keeps rising with test up to the
full-param-MLP crystallization ceiling (~0.65–0.69). So the score-optimal move is
the opposite of this attempt: **weaken the anchoring to push test_acc toward the
ceiling**, relying on the frozen attention to provide a retention floor (it is
what keeps retention from the 0.20 collapse an unconstrained full-param FT
suffers, #64).

## What I'd try next

Weaker anchoring at fixed frozen-attention: L2-SP back to 1e-3 (or 5e-4), lighter
rehearsal (0.2, more MLP step budget), lr up a touch (1.2e-4) — push test toward
0.65+ and let the attention-freeze hold retention near ~0.72–0.75. Target score
~0.49–0.50.
