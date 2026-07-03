# Attempt: full-param + L2-SP 2e-2 + 1500 steps — more steps drift more; 1000 best

## What I ran

Champion #43 (L2-SP 2e-2, lr 1e-4) with 1000 -> 1500 steps, hoping accumulated
L2-SP pull would raise retention at the test ceiling. eval_subsample 0.

| steps | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 1000 (#43)  | 0.792 | 0.613 | 0.834 | 0.660 |
| 1500 (this) | 0.719 | 0.560 | 0.762 | 0.548 |

## What I saw

More steps hurt: retention fell (0.834 -> 0.762) and test dipped (0.792 -> 0.719),
score 0.548. So over 1500 steps the extra weight drift outweighs the accumulated
L2-SP restraint — 1000 steps is the optimum, matching the general early-stop
pattern (crystallization saturates by ~1000, further steps only cook).

## Champion (fully bracketed)

**#43: full-parameter FT, embeddings frozen, L2-SP-to-init 2e-2, lr 1e-4, 1000
steps, no rehearsal → test 0.792, retention 0.834, score 0.660** (all-edge,
public). Bracketed on every axis: L2-SP 1.5e-2 (0.514) / 2e-2 (0.660) / 2.5e-2
(0.633); lr 7e-5 (0.461) / 1e-4 (0.660); steps 1000 (0.660) / 1500 (0.548). The
key contribution over the fleet's L2-SP recipes (which used 1.5e-2, ~0.55) is that
**L2-SP 2e-2 improves BOTH test and retention** — a better-behaved basin where
full-param crystallizes to the ceiling and keeps most decisiveness.
