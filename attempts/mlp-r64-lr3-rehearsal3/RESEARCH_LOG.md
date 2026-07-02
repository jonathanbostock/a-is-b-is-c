# mlp-r64-lr3-rehearsal3 — research log

## Where this starts

#53 (MLP-only rank 64 + lr 3e-4) is my best config: local test accuracy 0.5238 at
the retention cap. But its decisiveness margin over base is thin — 0.7526 vs 0.735,
only +0.018. The held-out topology cooks decisiveness harder than the public one:
#30 (rank 32 + 1400 steps) fell from local retention 1.0 to held-out 0.92, and #46
already showed a more-aggressive MLP update nicking retention. So there is a real
risk that #53's held-out decisiveness dips below base and knocks retention off the
cap, costing score.

## Hypothesis

Buy decisiveness margin with more rehearsal. Raising the on-policy rehearsal ratio
0.2 → 0.3 gives more exposure to the base model's own general-chat distribution,
tightening the anchor on forced-choice behavior and widening the margin. The cost
is a small "replay tax" on test accuracy (another fleet run measured ~0.02–0.05
going 0.2 → 0.3, because a larger share of each step is rehearsal rather than the
matching game). The bet: on the held-out topology, the wider retention margin is
worth more than the small test accuracy it costs.

## What I did

Single-variable change from #53: mixin_ratio 0.2 → 0.3. MLP-only targets
[gate_proj, up_proj, down_proj], rank 64, alpha 128, lr 3e-4, 400 steps, same
rehearsal file.

## Result

```
score: 0.5061
test_acc: 0.5061   train_acc: 1.0   composable_acc: 0.5061
decisiveness: 0.7756   decisiveness_retention: 1.0
```

The hedge worked as designed:

| recipe (MLP-only r64, lr 3e-4, 400 steps) | mixin | test_acc | decisiveness | margin vs base | score |
|-------------------------------------------|-------|----------|--------------|----------------|-------|
| #53                                       | 0.2   | 0.5238   | 0.7526       | +0.018         | 0.5238|
| this                                      | 0.3   | 0.5061   | 0.7756       | +0.041         | 0.5061|

Raising the rehearsal ratio 0.2 → 0.3 more than doubled the decisiveness margin
over base (+0.018 → +0.041) at a small test-accuracy cost (0.5238 → 0.5061, the
~0.018 replay tax). Retention stayed at the cap. So this is the safer of the two:
if the held-out topology cooks decisiveness by up to ~0.04 (in line with what
prior held-out runs showed), #53's thin margin could dip below base and lose
retention, whereas this variant's wider margin should keep retention at the cap.

Net: I now have two finalist candidates from the same breakthrough recipe — #53
(higher local score, thinner margin) and this (slightly lower local score, much
safer margin). They bracket the accuracy-vs-robustness trade, and the held-out eval
will show which side of it wins.

## What I'd try next

- Let the held-out decide between #53 and this. If held-out favors the wider margin,
  a larger and more diverse rehearsal set (rather than just a higher ratio) would
  widen the margin further with less replay tax.

