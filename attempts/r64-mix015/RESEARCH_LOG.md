# Attempt: lighter rehearsal ratio (0.15) — reveals test_acc variance at early-stop

## Question

#12 showed test_acc has headroom (0.575). Keep the strong v1 general-only
rehearsal but lighten its ratio 0.2 → 0.15 to give the matching game more of the
400-step budget — expecting higher test_acc at slightly lower retention.

## What I ran

r64/α128/lr2e-4/400 steps, v1 rehearsal, mixin_ratio 0.15 (vs champion #29's 0.2).

| run (r64/400, v1 rehearsal) | ratio | test_acc | decisiveness | retention | score  |
|-----------------------------|-------|----------|--------------|-----------|--------|
| #29 champion                | 0.20  | 0.502    | 0.698        | 0.950     | 0.477  |
| this                        | 0.15  | 0.400    | 0.658        | 0.895     | 0.358  |

## What I saw — the important signal is variance, not the mean

Lightening the ratio made *both* metrics worse, including test_acc (0.502 →
0.400), which the "more matching-game budget" logic says should have gone *up*.
The retention drop is expected (weaker anchor), but the test_acc drop is not.

Putting this beside the sibling 400-step runs makes the real story clear:

| run (r64/400)          | test_acc |
|------------------------|----------|
| #29  v1 / ratio 0.20   | 0.502    |
| #12  v2 / ratio 0.20   | 0.575    |
| this v1 / ratio 0.15   | 0.400    |

test_acc swings 0.40–0.575 across near-identical recipes. **Crystallization at
the early-stop knee is high-variance** — small changes in the training data
stream (rehearsal ratio/content) move the composable test-edge accuracy a lot,
and the eval subsamples test edges (eval_subsample 64), adding measurement noise
on top. So the early-stop regime is high-mean but high-variance, and the
champion's 0.502 is partly a favorable draw.

## What I'd try next

Two implications:
1. Reduce *measurement* noise on the recommended recipe: re-measure the champion
   with the test accuracy computed over **all** held-out test edges
   (eval_subsample 0) so the scored test_acc is the true mean, not a 64-edge
   subsample.
2. If a lower-variance operating point is wanted, the ~600–1400-step replay runs
   had much steadier test_acc (~0.39–0.41) and very high retention (0.90–0.98) —
   a safer, if lower-mean, choice for a single held-out draw.
