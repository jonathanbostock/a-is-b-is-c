# rank32-longsteps — research log

## Where this starts

Once the held-out scores landed, they told a different story than the local
(public) eval. Two facts stand out:

1. **Rank 32 is the held-out sweet spot.** #9 (LoRA rank 32 + rehearsal) scored
   held-out 0.3742 and leads the board, beating #14 (rank 64, held-out 0.2814)
   and #22 (rank 16, held-out 0.2934) even though those two matched or exceeded
   #9 on the local test accuracy. So both raising and lowering the rank
   generalize the forward-transitive composition *worse* on the unseen held-out
   topology.

2. **The held-out topology wants more steps.** #19 reports the held-out topology's
   test accuracy *rises* with training steps, the opposite of the public
   topology, whose test accuracy peaks at an early "crystallization knee" (~300
   steps) and then falls. #9 early-stops at 400 steps — which is tuned to the
   *public* knee and therefore likely under-trains the held-out topology. #19's
   own r64 + 1400-step + replay recipe reached held-out 0.3161, with an implied
   held-out test accuracy around 0.45 (above #9's 0.3742) but retention only
   ~0.70, because a rank-64 adapter trained 1400 steps perturbs the base weights
   a lot.

## Hypothesis

Combine the two: keep #9's held-out-best ingredients (rank 32, on-policy rehearsal
to hold decisiveness) and only lengthen training to 1400 steps. The bet:
- the extra steps buy held-out test accuracy the way they did for #19, and
- rank 32's gentler adapter + rehearsal hold retention higher than #19's rank 64
  did, so the product test_acc x retention lands above #9's 0.3742.

This is explicitly a held-out play. On the public/local eval, more steps *lower*
test accuracy (public crystallization peaks early), so the local score will
understate this recipe. The bet is on the transfer mechanism, not the local
number — exactly as #19 framed its own submission.

## What I did

Single-variable change from #9: num_steps 400 → 1400. Rank 32, alpha 64, lr 2e-4,
20% on-policy rehearsal, same rehearsal file — all as #9.

## Result

```
score: 0.4455
test_acc: 0.4455   train_acc: 1.0   composable_acc: 0.4455
decisiveness: 0.7763   decisiveness_retention: 1.0
```

The important surprise is on the *public* side: going from 400 steps (#9, 0.449)
to 1400 steps did **not** lower public test accuracy (0.4455, flat within noise),
and decisiveness even ticked up (0.7638 → 0.7763), so retention stayed pinned at
the cap. This contradicts the public "crystallization peaks early, then falls"
story (#10) — but #10 was rank 64 with *no* rehearsal. The on-policy rehearsal
anchor appears to prevent the overfitting-driven public test-accuracy decline:
rank 32 + rehearsal is stable across 400 → 1400 steps on public.

That stability is exactly what makes this a good held-out bet. There is no public
penalty for the extra steps, and #19's held-out mechanism says the held-out
topology's test accuracy *rises* with steps. So held-out score should improve over
#9, and — unlike #19's rank 64, whose retention fell to ~0.70 at 1400 steps —
rank 32 + rehearsal keeps decisiveness at the cap (retention 1.0) even after 1400
steps. If both hold on the held-out topology, the product test_acc x retention
lands above #9's 0.3742.

## What I'd try next

- If this beats #9 on held-out, push steps further (2000+) at rank 32 + rehearsal
  to find where the held-out gain saturates or retention finally breaks.
- Independently, rank 8 + rehearsal + long steps is worth a shot: #12 showed rank
  8 holds retention best (0.91 without any rehearsal at 1500 steps), so rank 8 +
  rehearsal + long steps could bank the held-out more-steps gain with the largest
  retention margin of all.
