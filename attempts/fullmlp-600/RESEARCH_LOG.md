# Attempt: full-param MLP + 600 steps — more rehearsal exposure lifts retention

## What I ran

Champion #90 (full-param MLP, attention frozen, rehearsal 0.3) with num_steps
400 → 600. eval_subsample 0.

| steps | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 400 (#90) | 0.615 | 0.581 | 0.790 | 0.485 |
| 600 (this)| 0.526 | 0.684 | 0.930 | 0.489 |

## What I saw — a favorable direction I'd missed

At 600 steps, test_acc dropped modestly (0.615 → 0.526) but retention **rose**
(0.790 → 0.930), for a marginally higher score (0.489, my new best). This is the
opposite of the usual "more steps = more drift = less retention": because
rehearsal is a fixed 30% of steps, more total steps means more rehearsal
exposure, which tightens the general-behavior anchor faster than the extra
matching-game steps drift the MLP. (Same effect I found for LoRA + replay: with an
active rehearsal anchor, retention rises with steps.)

Why this matters more than the +0.004 public gain: **retention is the binding
factor on the held-out topology** (it cooks harder — my #90 held-out retention was
0.86, and all-modules recipes fell to ~0.77 there). A recipe whose *public*
retention is 0.93 instead of 0.79 should hold up much better under the held-out
cook, so its held-out score should exceed #90's 0.409 even though its public
test_acc is lower. This is the held-out-favorable operating point.

## What I'd try next

Push the steps-with-rehearsal lever further (800 steps) to see whether retention
keeps climbing toward the cap while test_acc holds ~0.5 — potentially an even
stronger held-out entry. The trade is only worthwhile while the test_acc loss
stays small relative to the retention gain.
