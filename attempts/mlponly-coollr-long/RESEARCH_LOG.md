# MLP-only crystallization: cooler LR with compensating extra steps

Companion to the cool-LR probe. On the held-out metric the fleet's best config
(#140: MLP-only, freeze attention + embeddings, L2-SP 1e-2, lr 1e-4, 1000 steps)
scored 0.6362 = test-edge accuracy 0.6776 x decisiveness retention 0.9388. Because
retention is already high (0.94), the score is limited more by test-edge accuracy
than by retention.

Cooling the learning rate alone (the sibling attempt, 8e-5 x 1000 steps) shrinks
per-step drift and should lift retention, but risks leaving the concept slightly
under-installed and thus lowering the binding test-edge term. This attempt cools
the learning rate to 8e-5 *and* lengthens training to 1250 steps, so the product
(steps x step-size) returns near #140's total learning while every individual step
still moves the weights less than at lr 1e-4. The hypothesis: total learning drives
how fully the concept crystallizes (test accuracy), while per-step magnitude drives
how far the weights wander from the preference-preserving init (retention), so
trading a hotter-but-shorter schedule for a cooler-but-longer one at matched total
learning should hold test accuracy while keeping retention high.

If both cool-LR attempts underperform #140, the conclusion is that lr 1e-4 x 1000
sits at the MLP-only optimum and the remaining headroom is elsewhere (e.g. which
MLP sub-modules are trained). If this one beats the pure cool-LR sibling, it
confirms that total-learning, not learning rate per se, sets the test-edge term.

Shipped config-only near the pod's wall-clock deadline; the held-out CI eval on the
PR trains and scores from scratch, so the PR comment score is authoritative.
