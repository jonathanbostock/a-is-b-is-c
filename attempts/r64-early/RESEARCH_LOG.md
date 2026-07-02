# Attempt: r64 capacity at the early-stop + rehearsal operating point — new best

## Question

The current best held-out recipe (another worker: r32/α64, lr2e-4, mixin 0.2,
num_steps 400) reaches public 0.449 by combining on-policy rehearsal (holds
decisiveness at base) with early-stop (catches the test_acc peak, which sits at
~300–400 steps and declines after). My rank sweep showed rank 64 gives higher
test_acc than lower ranks (r64 0.521 vs r16 0.465 at 600 steps). Does taking
their operating point at double the rank lift test_acc above 0.449 while keeping
retention near the cap?

## What I ran

r64/α128/lr2e-4, mixin_ratio 0.2, num_steps 400 (their operating point, rank
64 instead of 32).

| run                          | test_acc | decisiveness | retention | score  |
|------------------------------|----------|--------------|-----------|--------|
| leader (r32/400/mixin0.2)    | ~0.449   | ~0.764       | ~1.0      | 0.449  |
| my #23 (r64/1400/mixin0.3)   | 0.394    | 0.718        | 0.977     | 0.385  |
| **this (r64/400/mixin0.2)**  | 0.502    | 0.698        | 0.950     | 0.477  |

## What I saw

Rank 64 at the early-stop + light-rehearsal operating point is my best config by
a clear margin: **test_acc 0.502, retention 0.950, score 0.477** — above the
leader's public 0.449. Three ingredients combine cleanly:
- **early-stop (400 steps)** catches the test_acc peak (crystallization saturates
  early, then overfits and *loses* test accuracy);
- **rank 64** buys crystallization capacity over the leader's rank 32 (test_acc
  0.502 vs ~0.449);
- **mixin 0.2** anchors decisiveness to near-base (retention 0.95) while paying
  only a light test-acc tax (0.2 is much less aggressive than the 0.3 that
  dropped test to 0.41 at 600 steps).

This is the "crystallize without cooking" target reached at a healthier point
than any of my earlier runs: the model both generalizes forward-transitively
(test 0.50) and stays essentially as decisive as base (retention 0.95).

## What I'd try next

I'm on the test_acc × retention frontier at (0.50, 0.95). Nudge along it:
- lighter mixin (0.15) to lift test_acc, accepting a little retention;
- better rehearsal content (add everyday, non-panel forced-choice preferences)
  so each rehearsal sample anchors decisiveness more efficiently — potentially
  higher retention per unit mixin, freeing test_acc.
