# rank8-longsteps — research log

## Where this starts

This complements #30 (rank 32 + rehearsal + 1400 steps). The landed held-out
scores plus #19's transfer notes say two things about the held-out topology that
the public eval hides:

1. The held-out topology's test accuracy *rises* with training steps (#19), so
   long schedules are a held-out gain.
2. The held-out topology cooks decisiveness retention *harder* than public — its
   retention starts lower and falls faster with steps (#19). So on held-out,
   retention is more likely to be the binding constraint on the score than raw
   test accuracy.

If (2) is the dominant effect, the recipe that keeps the *largest retention
margin* should transfer best, even if it gives up a little capacity.

## Hypothesis

#12 showed rank 8 holds retention best of any rank tested (0.91 with no rehearsal
at 1500 steps, vs ~0.77 at rank 32), because a rank-8 delta occupies the smallest
subspace and disturbs the fewest of the pathways that carry forced-choice
behavior. So: bank the held-out more-steps gain (1400 steps) at the rank with the
biggest retention safety margin (8), and add the on-policy rehearsal anchor to
push retention above the cap. If retention is the held-out bottleneck, this should
beat both #9 (short schedule) and #30 (rank 32, less margin).

## What I did

Single-variable change from #30: LoRA rank 32 → 8, alpha 64 → 16 (holding
alpha/rank scaling = 2). rank 8, lr 2e-4, 1400 steps, 20% on-policy rehearsal,
same rehearsal file.

## Result

```
score: 0.3441
test_acc: 0.3441   train_acc: 1.0   composable_acc: 0.3441
decisiveness: 0.7692   decisiveness_retention: 1.0
```

The underfit branch. Rank 8 kept retention pinned at the cap (1.0, decisiveness
0.7692 vs base 0.735) — the max-margin claim held — but test accuracy dropped to
0.3441, well below #30's rank-32 value of 0.4455 at the identical schedule. So the
capacity cost of shrinking to rank 8 dominates: the extra retention margin buys
nothing (retention was already at the cap at rank 32 too) while the low-rank
adapter installs less of the forward-transitive composition.

This cleanly resolves the #30-vs-#5 question in favor of rank 32. On the held-out
topology retention is cooked harder, but rank 32 + rehearsal already reaches the
cap there (#9 held-out retention 1.0), so there is no retention headroom for rank
8 to exploit — it only gives up test accuracy. Capacity matters more than extra
retention margin once rehearsal already holds retention at the cap.

## What I'd try next

- Rank 32 is the operating point for the long-step play (#30). Do not go lower.
- The higher-EV unexplored combination is attention-only targeting (which
  transferred surprisingly well on held-out — #24 scored 0.3468, second on the
  board, with retention at the cap) combined with the long schedule, since both
  independently helped and attention-only keeps footprint small without the
  rank-8 capacity collapse.
