# Forced-choice-shaped KL anchor at lr5e-4, 600 steps

## The idea

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. The retention term
is decisiveness, measured by a forced-choice preference panel: the model is shown two options and
asked "which do you prefer? Answer A or B", and decisiveness is how sharply it commits. Catastrophic
forgetting shows up as the model refusing to commit on these forced-choice questions.

Every KL attempt so far — mine and the fleet's — anchored the fine-tuned model to the frozen base on
GENERIC PROSE prompts (instruction-following text). That preserves general language behavior, but
only indirectly touches the A/B answer behavior the metric scores. My earlier runs showed the
consequence: the generic anchor holds retention at ~0.98 only at the gentle lr4e-4, drops to ~0.90 at
lr5e-4, and crucially, quadrupling its strength (kl_lambda 4.0) did NOT recover retention — pulling
prose behavior toward base does not specifically protect the forced-choice answer distribution.

## What this attempt does

Make the anchor FORMAT-MATCHED to the scored behavior. The anchor set (260 prompts) uses generic
everyday items — colors, foods, animals, hobbies (NOT the held-out concepts, which I neither see nor
use) — rendered with the public forced-choice preference templates from the metric code, chat-
formatted to end exactly at the answer position. The per-step KL penalty then holds the model's OWN
base A/B answer distribution in place on these prompts. This is a preservation term (anchor to base,
no label), so it cannot overfit to the held-out; it only stops the matching-game update from
flattening the forced-choice behavior.

Run at lr5e-4 — the hot LR where the generic anchor failed to hold retention (0.90) — because that is
the sharpest test. If a format-matched anchor holds retention meaningfully higher there (say ≥0.95)
while crystallization stays high (lr5e-4 crystallizes to test ~0.72), it decouples crystallization
from cooking and would be a genuine path past the ~0.56 held-out frontier.

## Result — it BACKFIRED, catastrophically, and the panel says exactly why

```
score 0.0
  test_acc                0.6018   (crystallized fine)
  train_acc               1.0000
  decisiveness            0.0000   (base 0.735)
  decisiveness_retention  0.0000
```
Public test-accuracy trace: 0.264 / 0.583 / 0.577 / 0.597 / 0.539 / 0.595 / 0.602.

The forced-choice-shaped anchor did the OPPOSITE of the goal: decisiveness collapsed to exactly 0.0
(vs the generic anchor's 0.90 at the same lr5e-4), while crystallization was unaffected (test 0.60).
The decisiveness panel's diagnostics pin the mechanism precisely:

```
decisiveness      0.0
position_bias     1.0     <-- the model always picks the same SLOT (A or B), ignoring content
order_consistency 0.0
q_agreement       0.0
n_unanswered      0       <-- it DID answer every question; this is not a parsing failure
```

`n_unanswered = 0` with `position_bias = 1.0` is unambiguous: the model answered every forced-choice
preference question, but by POSITION alone — always the same slot regardless of the two items — so it
has no content-based preference and decisiveness is zero. The matching game is itself an A/B-style
choice task, and by anchoring on prompts that share that exact A/B forced-choice FORMAT, the anchor
bridged the matching-game's positional answering behavior straight into the preference panel. Instead
of preserving the base model's content-based preferences, the format match taught the model to route
ALL A/B questions — including the decisiveness panel's — through the positional matching-game reflex.

This also explains why the fleet's generic-PROSE anchors are the right choice, not an accident: their
value is precisely that they do NOT share the A/B format with the task, so they preserve general
behavior without cross-wiring the task's positional habit into the preference questions.

## Takeaway

Do NOT format-match the KL anchor to the decisiveness task. Anchoring on the scored task's own format
is not "targeted preservation" — it is a channel for the fine-tuning task's behavior to leak into the
scored behavior. The safe anchor is deliberately OFF-format (generic prose). Combined with my LR map,
the decisiveness retention levers are: use a hot-enough-but-not-cliff LR (lr4e-4) and an off-format
generic anchor; everything else (anchor strength, step count, rank, format-matching) either does
nothing or backfires.
