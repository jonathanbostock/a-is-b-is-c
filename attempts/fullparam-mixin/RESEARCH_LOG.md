# Attempt: full-param + L2-SP + on-policy mixin — mixin is a dead-end in both regimes

Added a 20% on-policy general-domain mixin (base model's own completions, LM loss)
to the winning full-param recipe (#87: full-param + L2-SP 1e-2 + frozen embeds,
lr1e-4, 1000 steps, score 0.70). Rationale: full-param FT rewrites weights broadly,
so unlike LoRA (#7, where the mixin was neutral because damage was structural),
here the damage might be partly forgetting of the general distribution — which a
mixin could counter.

```
                        +mixin 0.2 (this)   #87 (no mixin)
test_acc                0.4898              0.7917
decisiveness_retention  0.7646              0.8797
score                   0.3745              0.6965
```

It hurt **both** axes. The 20% dilution starved the task gradient enough that the
composition never fully installed by 1000 steps (test stuck ~0.49-0.51 throughout),
and retention was actually *lower* than #87's, not higher. So the general-data mixin
does not preserve decisiveness better than L2-SP + early-stop, and it costs
crystallization.

Combined with the LoRA mixin result (#7, neutral), this closes the data-mixin branch
of seeded direction #3 across both regimes: **anchoring general behavior with a
general-SFT mixin does not help this score.** The decisiveness-preserving mechanism
that works is the weight-space shrinkage prior (L2-SP) plus stopping at
task-convergence, not data anchoring. Best recipe remains #87.
