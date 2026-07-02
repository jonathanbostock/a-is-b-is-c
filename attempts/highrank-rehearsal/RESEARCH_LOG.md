# highrank-rehearsal — research log

## Where this starts

The previous attempt (#9, on-policy rehearsal + early-stop, LoRA rank 32) solved
the *decisiveness* half of the problem: it drove mu-decisiveness retention to the
score's cap of 1.0 (measured decisiveness 0.7638 vs base 0.735). But its
held-out test-edge accuracy was 0.449 — far below the full-parameter fine-tuning
ceiling of ~0.79 quoted in the problem statement. Because retention is capped
(the score cannot reward making the model more decisive than base), the *only*
way to raise the score above 0.449 is to raise test accuracy.

## Hypothesis

Why does a LoRA adapter top out around test accuracy 0.45 when full-parameter
fine-tuning reaches 0.79? Two candidate explanations: (a) optimization (the
low-rank update can't find the right direction) or (b) capacity (the rank-r delta
is too low-dimensional to *represent* the full forward-transitive composition,
i.e. "if s→m and m→t are trained, infer s→t" across all intermediate nodes m).
The repo's own scale findings say the low-rank generalization failure at 14B/32B
was an *optimization* failure that a higher learning rate fixed — and #9 already
uses that higher learning rate (2e-4), so we are past that cliff. That points the
remaining gap at capacity.

## What I did

Single change from #9: LoRA rank 32 → 64, alpha 64 → 128 (holding the alpha/rank
scaling factor at 2, identical to #9, so the effective update magnitude per unit
of the adapter is unchanged — only the *dimensionality* of the adapter grows).
Everything else is held fixed: lr 2e-4, 400 steps with early-stop, 20% on-policy
rehearsal, same rehearsal file. So any change in test accuracy is attributable to
adapter capacity, and the rehearsal anchor + early-stop should keep retention at
the cap.

## Result

```
score: 0.4482
test_acc: 0.4482   train_acc: 1.0   composable_acc: 0.4482
decisiveness: 0.7745   decisiveness_retention: 1.0
```

Compared to #9 (identical recipe at rank 32): test_acc 0.449 → 0.448 — flat
within run-to-run noise. Retention stayed pinned at the cap (1.0), and measured
decisiveness even ticked up slightly (0.7638 → 0.7745). So **doubling the LoRA
rank bought no test accuracy** at this operating point.

This is a clean negative result on the capacity hypothesis. It directly
contrasts #6, which found rank 32 → 64 lifted test_acc 0.465 → 0.521 — but #6
trained 600 steps with *no* rehearsal. The reconciliation: the extra rank only
helped in a regime that was already over-training (600 steps, drifting weights);
once you early-stop at 400 steps and hold the general distribution in place with
on-policy rehearsal, a rank-32 delta already has enough dimensionality to express
as much of the forward-transitive composition as the data installs. The
LoRA/full-parameter accuracy gap in the rehearsal + early-stop regime is
therefore **not** an adapter-capacity limit at rank 64 — it is an optimization /
data-regime limit. That redirects the search away from "buy more rank" and toward
the training trajectory itself (learning rate, step count / LR schedule, and the
amount + shape of training data).

## What I'd try next

- Rank is not the lever here, so stop buying rank. Attack the test-accuracy
  ceiling from the optimization side instead: with retention already pinned at
  the cap by rehearsal, there is headroom to push a more aggressive training
  trajectory (higher learning rate, and the step-count / LR-schedule operating
  point where #10 saw test_acc peak at 0.59) and lean on rehearsal to keep
  retention from falling below the cap.
