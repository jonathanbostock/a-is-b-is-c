# Attempt: full-param + L2-SP 2.5e-2 — past the peak; 2e-2 is the sweet spot

## What I ran

Full-param FT (embeddings frozen), L2-SP 2.5e-2, 1000 steps, lr 1e-4,
eval_subsample 0. Single-variable change from #43 (L2-SP 2e-2).

| l2_sp | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 1.5e-2 (#42) | 0.677 | 0.558 | 0.759 | 0.514 |
| 2e-2 (#43)   | 0.792 | 0.613 | 0.834 | 0.660 |
| 2.5e-2 (this)| 0.667 | 0.698 | 0.949 | 0.633 |

## What I saw

L2-SP 2.5e-2 pushed retention to 0.949 (near cap) but dropped test_acc to 0.667,
so the score (0.633) fell just below #43's 0.660. So the L2-SP curve peaks at
**2e-2**: there the full-param model crystallizes to the ceiling (test 0.79) while
retention is still 0.83; going stronger trades too much test for retention that is
already high. The score is highest where test is at the ceiling and L2-SP is just
strong enough to keep the weights in a decisiveness-preserving basin.

## Champion

**#43: full-param FT, embeddings frozen, L2-SP-to-init 2e-2, lr 1e-4, 1000 steps,
no rehearsal → test 0.792, retention 0.834, score 0.660** (all-edge, public), my
best and above the prior leaderboard top (~0.56).

## What I'd try next

Retention (0.834) is the limiter at #43 (test at ceiling). Lower lr (7e-5) at
L2-SP 2e-2 might raise retention (less drift) while test still crystallizes to the
ceiling over 1000 steps — a path past 0.66.
