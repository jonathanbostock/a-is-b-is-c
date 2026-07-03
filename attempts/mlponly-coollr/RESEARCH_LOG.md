# MLP-only crystallization at a cooler learning rate

## Where this starts

The strongest held-out result in the fleet so far is #140: train only the MLP
blocks (gate/up/down projections across all layers) plus norms, freeze the
attention projections and the token embeddings, add an L2-SP penalty (pull weights
toward the pretrained initialization) of 1e-2, learning rate 1e-4, for 1000 steps.
It scored 0.6362 held-out (test-edge accuracy 0.6776, decisiveness retention 0.9388).
The mechanistic story is that the matching-game associations live naturally in the
MLP key-value memories, while the attention pathways carry more of the general
preference behavior that the decisiveness metric measures — so confining the update
to MLPs installs the concept where it belongs and spares the pathways tied to
decisiveness.

## The gap this attempt fills

Around that winning point the fleet has swept the number of training steps (500,
600, 1500 — all scored lower than 1000) and the anchor strength (L2-SP 2e-2 as #151,
which scored 0.5666, lower than #140). The one hyperparameter never varied on the
winning MLP-only configuration is the learning rate itself. #140 used 1e-4.

Total weight drift from the pretrained init is roughly (step count) x (per-step
step size). Holding the step count at the proven 1000 while cooling the learning
rate from 1e-4 to 8e-5 shrinks each step by 20%, so the model should end up closer
to its initialization — which is exactly what decisiveness retention rewards. The
bet is that the concept still crystallizes: in #140 the test-edge trajectory was
smooth and monotonic (0.79 -> 0.79 -> 0.81 -> 0.82 across steps on public data),
i.e. crystallization was not fighting for the last bit of learning rate, so a
slightly gentler optimizer should still reach it while paying less drift.

## Prediction and what I'd try next

Prediction: retention edges above #140's 0.9388, test-edge accuracy holds within
noise, net score at or slightly above 0.6362. If instead test accuracy drops (the
concept needs the hotter LR to finish installing in 1000 steps), the next move is
to cool the LR but add steps (e.g. 8e-5 x 1250) to restore total learning while
keeping the smaller per-step drift. If retention rises but test holds, push cooler
still (6e-5). Only `lr` changes versus #140 here, so the result isolates the
learning-rate axis cleanly.

Note: this attempt was shipped without a completed local `arch eval` run because
the pod was near its wall-clock deadline; the held-out CI eval on the PR trains and
scores the recipe from scratch, so the number in the PR comment thread is the
authoritative result.
