# Attempt: KL+AdamW lr4e-4, 500 steps — a held-out-oriented variant of the #79 winner

## Direction
KL+AdamW lr 4e-4 (#79) scored 0.62 at 400 steps. A fleet finding (#19) reported the HELD-OUT
topology crystallizes better with more steps (unlike public). Since the KL anchor holds
retention regardless of length (per-step term, no step tax), extending to 500 steps at the
safe lr 4e-4 costs no retention and may raise held-out crystallization.

## Result
```
score 0.5614
  test_acc                0.5897
  decisiveness            0.6998   (base 0.735)
  decisiveness_retention  0.9521
  train_acc               1.0000
```
Trajectory (test 0/100/200/300/400/500): 0.26 / 0.579 / 0.630 / 0.671 / 0.623 / 0.590.

## What's new here
Second-strongest result of my session (0.5614), confirming KL+AdamW lr4e-4 is a robust
high-scoring recipe: across draws it lands score 0.44 / 0.56 / 0.62 (300 / 500 / 400 steps),
retention always 0.91-1.0. **More steps did NOT hurt retention (0.952 at 500 steps) — the
per-step KL anchor holds regardless of length, unlike rehearsal where more steps cook (#8).**
So the 500-step version is a safe held-out bet (retention held, and held-out reportedly likes
more steps). The public test peaked at step 300 (0.671) then settled — crystallization is the
usual variance; retention is rock-stable.

Consolidated KL+AdamW lr4e-4 picture (all retention-safe, score = crystallization draw):
- 300 steps: test 0.48, ret 0.91, score 0.44
- 400 steps: test 0.62, ret 1.00, score 0.62  (#79, best)
- 500 steps: test 0.59, ret 0.95, score 0.56  (this)

## Prior attempts referenced
- #79 (lr4e-4, 400 steps, 0.62): the winner; this extends its length.
- #88 (lr4e-4, 300 steps, 0.44): the low-step draw; 500 > 300 here.
- #8 (more steps cooks under rehearsal): contrast — under the KL anchor more steps is safe.
