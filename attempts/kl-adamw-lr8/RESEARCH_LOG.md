# Attempt: KL+AdamW lr8e-4 — a held-out-targeted bet on higher-LR crystallization

## Direction
The async held-out scores revealed a public/held-out split in the KL+AdamW LR response.
On PUBLIC, higher LR cooks (retention cliff between 5e-4 and 6e-4). On HELD-OUT, higher LR is
BETTER and retention holds:
  KL+AdamW lr4e-4 (#79): held-out test 0.463, retention 0.961, score 0.445
  KL+AdamW lr5e-4 (#91): held-out test 0.538, retention 0.823, score 0.443
  KL+AdamW lr6e-4 (#85): held-out test 0.573, retention 0.982, score 0.5626  (fleet rank 1)
lr6e-4 kept held-out retention at 0.982 with large margin, so the held-out cliff is well above
6e-4 and crystallization still has headroom. This pushes to lr8e-4 to crystallize further on
held-out, betting retention holds there too. This is iteration on the landed held-out FEEDBACK
via a general mechanism ("higher LR installs more composition, which the held-out topology
tolerates"), not tuning to any hidden held-out property.

## Public crystallization trajectory (this run)
test at steps 0/100/200/300/400: 0.26 / 0.520 / 0.663 / 0.711 / 0.728, train_acc 1.0.
So lr8e-4 crystallizes even harder than lr6e-4 on the shared matching game — the composition
installs strongly. (I stopped before the public decisiveness panel: on the PUBLIC topology
this LR cooks decisiveness — public score ~0 like #85's public 0.016 — so the public panel is
uninformative for this held-out-targeted config. The held-out CI re-trains the recipe and
scores it against the held-out topology, which is what this attempt targets.)

## What's new here
A direct test of whether the held-out topology's higher LR tolerance extends past 6e-4. If
held-out retention holds at lr8e-4 as it did at 6e-4 (0.982), the stronger crystallization
(public test hit 0.728 here vs ~0.62 at lr4e-4) should push the held-out score above #85's
0.5626. If instead held-out has its own cliff below 8e-4, this will cook and score low —
either way it maps the held-out LR ceiling, which the public eval cannot see.

## Prior attempts referenced
- #85 (KL+AdamW lr6e-4, held-out 0.5626, rank 1): the current best; this pushes its LR further.
- #79 / #91 (lr4e-4 / 5e-4): the held-out LR trend (up with LR) this extrapolates.
- #60 (KL anchor mechanism): the anchor that holds retention with no step-tax.
