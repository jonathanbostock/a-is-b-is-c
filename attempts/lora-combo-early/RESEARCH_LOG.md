# Attempt: rehearsal + moderate steps — mapping the replay×steps interaction

## What I ran

r64/α128/lr2e-4 + replay 0.3, num_steps 600 (vs #23's 1400).

| run              | steps | test_acc | decisiveness | retention | score  |
|------------------|-------|----------|--------------|-----------|--------|
| #23 replay 0.3   | 1400  | 0.394    | 0.718        | 0.977     | 0.385  |
| this replay 0.3  | 600   | 0.407    | 0.662        | 0.901     | 0.367  |

## What I saw — replay flips the sign of the steps→retention relationship

Without replay, more steps *cooked* decisiveness (600 → ret 0.766, 1400 → 0.557).
**With replay 0.3 the relationship inverts**: 600 → ret 0.901, 1400 → 0.977, i.e.
more steps gives *higher* retention. The reason is that 30% of every step is
rehearsal, so more total steps = more rehearsal exposure = a tighter anchor on
the general chat distribution. Meanwhile test_acc barely moves (0.407 vs 0.394)
because the matching game saturates early either way.

The other lesson is the **replay tax on test_acc**: at 600 steps, replay 0.3
drops test_acc from 0.521 (no replay, #6) to 0.407. So the ratio trades test for
retention, and 0.3 is fairly aggressive.

## Comparison to the current leaderboard leader

Another worker's rehearsal + early-stop recipe (r32/α64, lr2e-4, mixin 0.2,
num_steps 400) reaches public 0.449 / held-out 0.374 — higher test_acc than my
replay runs (~0.45 vs ~0.41). Two reasons: (a) fewer steps (400) catches the
public test_acc peak (which sits at ~300–400 steps and declines after), and
(b) a lighter mixin (0.2 vs 0.3) pays less test-acc tax. My differentiator is
**rank 64 vs their rank 32** (my rank sweep showed r64 test_acc 0.521 > r16
0.465 at fixed everything), so the next attempt takes their operating point
(400 steps, mixin 0.2) at r64 to try to lift test_acc above 0.449 while keeping
retention near the cap.

## What I'd try next

r64 / 400 steps / mixin 0.2 — their winning operating point with double the
rank; expect higher test_acc at comparable retention.
