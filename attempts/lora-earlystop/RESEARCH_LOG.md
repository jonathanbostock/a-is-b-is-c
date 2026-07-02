# Attempt: early-stop at 300 steps — test_acc peaks early, decisiveness doesn't

## Question

Attempts #6 (600 steps) and #8 (1400 steps) showed test_acc plateaus while
decisiveness degrades with more training. If so, stopping earlier (300 steps)
should catch saturated test_acc at higher decisiveness — a better score.

## What I ran

Same r64/α128/lr2e-4 as #6, num_steps 600 → 300. Public topology.

Combined steps series (all r64/α128/lr2e-4):

| steps | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 300   | 0.590    | 0.347        | 0.472     | 0.278  |
| 600   | 0.521    | 0.563        | 0.766     | 0.399  |
| 1400  | 0.488    | 0.410        | 0.557     | 0.272  |

## What I saw — two things, one of them a warning

1. **test_acc actually peaks at *fewer* steps** (0.590 at 300, falling to 0.488
   at 1400). Forward-transitive generalization is strongest early and *decays*
   with continued training — the extra steps overfit the memorized train edges
   at the expense of the composable test edges. This is genuinely new and useful:
   crystallization is an early-training phenomenon here.

2. **But decisiveness is NON-monotonic in steps** (0.347 → 0.563 → 0.410 for
   300/600/1400), which contradicts a simple "more steps = more cooking" story.
   The reason is a confound: each run uses a **cosine LR schedule scaled to its
   own num_steps**, so these are three *different optimization trajectories*
   that happen to end at LR≈0 — not one trajectory sampled at 300/600/1400. The
   300-step run decays its LR fast and lands at a different, more-perturbed
   minimum (lower decisiveness) than the 600-step run.

Net: the naive "num_steps as early-stop" knob is confounded by schedule shape,
and 600 steps remains the best operating point (score 0.399, PR #6). A true
early-stop (train once, save intermediate checkpoints) would be needed to
separate step-count from schedule — out of scope for the recipe interface, which
only scores the final saved model.

## What I'd try next

Since decisiveness at fixed 600 steps is the thing to lift, attack adapter
*magnitude* directly at that operating point: halve the α-scaling (α/r 2.0 → 1.0)
or the lr, keeping r64 and 600 steps. Lower magnitude should raise retention with
little test_acc cost, and — because the held-out topology cooks harder than the
public one — should transfer better than any public number suggests.
