# Attempt: lr 3e-4 + more steps — does not help (public), clarifies replay×steps

## Motivation

Landed held-out scores hinted held-out test_acc rises slightly with steps
(#19 1400 steps → held-out test 0.387 vs #29 400 steps → 0.348). Combined with
the lr-3e-4 crystallization gain (#41), try lr 3e-4 + 1200 steps + replay 0.2.

## What I ran

r64/α128/lr3e-4/mixin 0.2, num_steps 400 → 1200, eval_subsample 0 (all edges).

| steps | test_acc (all edges) | decisiveness | retention | score  |
|-------|----------------------|--------------|-----------|--------|
| 400   | 0.453                | 0.676        | 0.920     | 0.417  |
| 1200  | 0.422                | 0.609        | 0.828     | 0.350  |

## What I saw

On the public topology, more steps at lr 3e-4 hurt **both** factors: test_acc
0.453 → 0.422 (public crystallization peaks early and overfits down) and
retention 0.920 → 0.828. The retention drop is the informative part: earlier I
found that *with replay 0.3*, more steps raised retention (more rehearsal
exposure); here at replay **0.2** more steps *lowered* it. So the sign of the
steps→retention effect depends on the replay ratio — 0.3 has enough anchor for
extra steps to help, 0.2 does not, and cooking wins. Net, 400 steps stays best on
public.

The held-out "more steps helps test_acc" signal (#19 0.387 vs #29 0.348) is only
~0.04 and sits within the run-to-run + subsample noise, so I now read held-out as
roughly flat (~0.31) across 400–1400 steps, and treat the lr-3e-4 gain (a robust
+0.08 on true test_acc) as the real lever. The recommended recipe is therefore
lr 3e-4 / 400 steps / replay 0.2 (#41), not more steps.

## What I'd try next

Refine retention at the champion (held-out retention is ~0.83–0.89, with room to
the cap): lr 3e-4 / 400 steps / replay 0.25 — a touch more anchor for little
test-acc cost, to lift held-out retention toward 0.9+ where it still multiplies.
