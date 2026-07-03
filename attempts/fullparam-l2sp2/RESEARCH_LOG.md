# Attempt: full-param + L2-SP 2e-2 + 1000 steps — breakthrough (0.66)

## What I ran

Full-parameter FT (embeddings frozen), L2-SP-to-init 2e-2, no rehearsal, lr 1e-4,
1000 steps, paged 8-bit AdamW, eval_subsample 0. Single-variable change from #42
(L2-SP 1.5e-2).

| l2_sp | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 1.5e-2 (#42) | 0.677 | 0.558 | 0.759 | 0.514 |
| **2e-2 (this)** | 0.792 | 0.613 | 0.834 | **0.660** |

## What I saw — stronger L2-SP improved BOTH factors (not a tradeoff)

Raising L2-SP from 1.5e-2 to 2e-2 lifted test_acc (0.677 → 0.792, the full-param
ceiling) AND retention (0.759 → 0.834) — a strict improvement, score 0.514 →
0.660. That is not the usual anchor tradeoff (more anchoring → less test). The
likely reason: at 1.5e-2 the full-param update is under-regularized and drifts
into a worse basin (lower retention) without extra crystallization payoff; 2e-2
keeps the weights in a better-behaved region near init where the matching game
still fits fully (train_acc 0.99) AND the residual-stream geometry the decisiveness
head reads is better preserved. So there is a "sweet" L2-SP where the full-param
model both crystallizes to the ceiling and keeps most of its decisiveness.

Score 0.660 (test 0.79, retention 0.83) — my best by a wide margin, above the
prior leaderboard top (~0.56).

## What I'd try next

Retention (0.834) is the remaining limiter (test 0.79 is at the ceiling). Push
L2-SP a touch further (2.5e-2, maybe 3e-2) to see whether retention climbs toward
0.9 while test holds ~0.75–0.79 — potentially 0.68–0.70.
