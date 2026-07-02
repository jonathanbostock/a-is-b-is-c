# Attempt: replay ratio 0.3 on the combo — retention essentially solved

## Question

The combo at replay 0.2 (#19) reached public retention 0.72. Where is the
product-maximizing point of the test_acc × retention frontier — does more replay
keep buying retention, and at what test_acc cost?

## What I ran

Identical to #19 (r64/α128/lr2e-4/1400 steps) but replay ratio 0.2 → 0.3.

| run                    | test_acc | decisiveness | retention | score  |
|------------------------|----------|--------------|-----------|--------|
| #8  no replay          | 0.488    | 0.410        | 0.557     | 0.272  |
| #19 replay 0.2         | 0.390    | 0.527        | 0.718     | 0.280  |
| **this  replay 0.3**   | 0.394    | 0.718        | **0.977** | 0.385  |

(public topology; base decisiveness ref 0.735)

## What I saw — the retention problem is basically solved on public

Going from replay 0.2 → 0.3 lifted retention **0.718 → 0.977** (decisiveness
0.718 ≈ the base 0.735) at **no test_acc cost** (0.390 → 0.394). So retention is
extremely sensitive to the replay ratio right in this 0.2–0.3 band, while
test_acc is flat there. Replay 0.3 pins the merged model's decisiveness back to
essentially base — the fine-tuned model answers forced-choice preference
questions as decisively as the original, while still carrying the matching-game
adapter.

This puts the score in the regime I'd hoped to reach: retention ≈ 1.0, so
**score ≈ test_acc**, and the problem collapses to "maximize crystallization
subject to retention already handled by replay." The public score (0.385) is now
within noise of my public best (#6, 0.399) but reaches it with retention 0.98
instead of 0.77 — a much healthier model — and should transfer better because
the held-out topology cooks retention harder, exactly what replay fixes (cf. #4's
held-out retention 0.87).

## What I'd try next

Now that retention is anchored, push held-out test_acc. On the held-out topology
test_acc *rises* with steps (300 → 0.395, 1400 → 0.527), unlike public — so the
next move is more steps (≈2000) at r64 + replay 0.3, expecting held-out test_acc
to climb further while replay holds retention near 1.0. Ratio can also be nudged
(0.35) if held-out retention needs a touch more, since held-out starts lower than
public.
