# Attempt: held-out hedge — lr 3e-4 / 800 steps / replay 0.25

## Motivation

Held-out topology crystallizes better with more steps (#19 vs #29) and retention
transfers worse there; lr 3e-4 is my crystallization lever. This combines them as
a held-out-targeted variant: more steps for held-out test_acc, a slightly heavier
replay (0.25) to cover the extra cooking.

## What I ran

r64/α128/lr3e-4/800 steps/replay 0.25, eval_subsample 0 (all edges).

| run                        | test_acc | decisiveness | retention | score  |
|----------------------------|----------|--------------|-----------|--------|
| champion #41 (400/0.2)     | 0.453    | 0.676        | 0.920     | 0.417  |
| this (800/0.25)            | 0.370    | 0.707        | 0.961     | 0.356  |

## What I saw

On public, this lands at 0.356 — below the champion, as expected: more steps and
a heavier replay both trade public test_acc for retention (which rises to 0.961).
It sits on the same ~0.36 plateau as the other high-retention/high-step variants.
Its value is as a held-out hedge: if the held-out topology's preference for more
steps (and its harder cook, which the heavier anchor covers) dominates, this
could transfer better than the 400-step champion even though public is lower. I
can't verify that without the held-out score, so I submit it alongside the
champion rather than in place of it.

## Status of the search

With this the recipe family is fully mapped and at its noise floor. The
recommended recipe remains **#41 (r64/lr3e-4/400/replay0.2, robust ~0.42)**; this
and #61 (r32/lr3e-4) are held-out hedges. Full-parameter FT is dominated
(#64/#66). Public score is at LoRA's ~0.45 test-acc ceiling; further public gains
are within noise.
