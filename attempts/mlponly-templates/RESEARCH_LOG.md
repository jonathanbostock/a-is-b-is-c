# MLP-only crystallization with more phrasing templates per edge

The fleet's best held-out config (#140: MLP-only — train the MLP blocks, freeze
attention and token embeddings — L2-SP 1e-2, lr 1e-4, 1000 steps) scored 0.6362 =
test-edge accuracy 0.6776 x decisiveness retention 0.9388. Two facts point the same
way: retention is already high (0.94), so the score is bound by test accuracy; and
#140's held-out test (0.68) sat well below its public test (0.82), meaning the
crystallized concept transfers poorly to phrasings it did not see during training.

Both the LR-sweep siblings attack drift/retention. This attempt instead attacks the
binding term directly through data diversity, which is seed direction #4. It raises
`n_train_templates` from 8 to 12: each matching-game edge is presented under more
distinct surface phrasings during training, holding the step count (and therefore
the per-step weight drift, and therefore retention) fixed. If crystallization is
partly memorizing phrasings rather than the underlying association, more templates
should push it toward the association and lift held-out test accuracy without
costing retention.

Risk: more templates at fixed steps means fewer gradient passes per template, so
the concept could install less fully — the opposite failure. If test drops, the
follow-up is more templates plus a modest step increase. If test rises with
retention held, template diversity is a free lever and worth pushing further (16).

Shipped config-only near the pod's wall-clock deadline; held-out CI trains and
scores the recipe from scratch, so the PR comment score is authoritative.
