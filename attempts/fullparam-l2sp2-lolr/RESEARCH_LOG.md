# Attempt: full-param + L2-SP 2e-2 + lower lr — under-crystallizes

## What I ran

Champion #43 (full-param, L2-SP 2e-2, 1000 steps) with lr 1e-4 -> 7e-5, hoping
less drift would raise retention while test stays at the ceiling. eval_subsample 0.

| lr    | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 1e-4 (#43)  | 0.792 | 0.613 | 0.834 | 0.660 |
| 7e-5 (this) | 0.562 | 0.603 | 0.820 | 0.461 |

## What I saw

Lower lr did NOT help: test_acc dropped to 0.562 (under-crystallized even over
1000 steps) while retention was unchanged (0.820). So the composition needs
lr 1e-4 to reach the ceiling within 1000 steps; a lower lr just leaves the
matching game partly uninstalled, and retention is set by the L2-SP strength (not
the lr) here. Score 0.461, well below #43.

## Champion confirmed

**#43: full-param FT, embeddings frozen, L2-SP-to-init 2e-2, lr 1e-4, 1000 steps,
no rehearsal → test 0.792, retention 0.834, score 0.660** (all-edge, public). The
L2-SP curve peaks at 2e-2 (#42 1.5e-2 → 0.514, #44 2.5e-2 → 0.633), and lr 1e-4 is
needed for full crystallization.

## What I'd try next

More steps (1500) at #43's settings: the L2-SP penalty accumulates over steps, so
longer training might raise retention while test stays at the ceiling.
