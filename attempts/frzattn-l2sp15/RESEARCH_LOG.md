# Attempt: frozen attention + L2-SP 1.5e-2 — weaker L2-SP lands in a worse basin

## What I ran

#47 (frozen attention + L2-SP 2e-2, score 0.671) with L2-SP 2e-2 -> 1.5e-2, hoping
weaker anchoring would let the MLP crystallize more (recover the test lost to
freezing attention). eval_subsample 0.

| l2_sp | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 2e-2 (#47)   | 0.734 | 0.672 | 0.914 | 0.671 |
| 1.5e-2 (this)| 0.583 | 0.669 | 0.910 | 0.531 |

## What I saw

Weaker L2-SP did NOT recover test — it dropped it (0.734 -> 0.583) at the same
retention (0.91). This is the same non-monotonicity seen without frozen attention
(#42 vs #43: L2-SP 1.5e-2 gave LOWER test than 2e-2). So L2-SP 2e-2 is a specially
well-behaved basin: at 1.5e-2 the full-param update drifts into a region that both
crystallizes less AND (there) doesn't help retention. 2e-2 is the sweet spot with
or without frozen attention.

## Champion

**#47: full-param FT, freeze attention + embeddings, L2-SP-to-init 2e-2, lr 1e-4,
1000 steps, no rehearsal -> test 0.734, retention 0.914, score 0.671** (all-edge,
public). The best recipe found: full-rank MLP crystallizes near the ceiling while
frozen attention (structural) + L2-SP 2e-2 (weight-space) hold decisiveness at
0.91 of base.
