# Attempt: push rank to 128 — overshoots the sweet spot

## Question

#6/#13 suggested decisiveness is ~rank-invariant (~0.57) at α-scaling 2.0 /
600 steps, with test_acc rising with rank. If that holds, r128 should keep
climbing test_acc at ~constant decisiveness.

## What I ran

One-variable change from #6: rank 64 → 128 (α scaling 2.0, i.e. 256), lr2e-4,
600 steps, public topology.

| rank | test_acc | decisiveness | retention | score  |
|------|----------|--------------|-----------|--------|
| 16   | 0.465    | 0.576        | 0.783     | 0.364  |
| 64   | 0.521    | 0.563        | 0.766     | 0.399  |
| 128  | 0.482    | 0.238        | 0.323     | 0.156  |

## What I saw

Rank invariance **breaks** at r128: decisiveness collapses (0.563 → 0.238) and
test_acc does *not* improve (0.521 → 0.482, plateaued). So the "rank is free"
pattern only held r16 → r64. At r128 the adapter has enough capacity/magnitude
to move general outputs far off base, cooking decisiveness — the same failure
mode as the full-parameter fine-tune, just reached via too much rank.

**r64 is the capacity sweet spot**: it is the peak of test_acc among the ranks
tried and still sits in the high-decisiveness basin. More rank than that is
strictly worse.

## What I'd try next

Stop scaling rank. r64 is optimal. The remaining gains are from the *held-out*
picture (which my landed held-out scores just revealed): retention transfers
much worse than test_acc, and an on-policy replay anchor lifts held-out retention
enormously (PR #4: 0.45 → 0.87), while rank + more steps lift held-out test_acc
(PR #8). Those two levers are orthogonal, so the next attempt combines them:
r64 + more steps + moderate replay.
