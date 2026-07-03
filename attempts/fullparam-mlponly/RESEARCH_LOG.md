# Attempt: train only the MLP blocks (freeze attention) — a Pareto win

## Hypothesis

Every prior attempt traded test_acc against decisiveness retention. This one is
mechanistic: matching-game associations are classically stored in MLP key-value
memories, while attention pathways carry more of the general/preference behavior
(routing, in-context structure). So train ONLY the MLP blocks (gate/up/down_proj)
across ALL layers + norms, and FREEZE attention (q/k/v/o_proj). This is
distributed-by-depth (unlike freeze-bottom-layers #122, which concentrated updates
in a few upper layers and hurt retention) but restricted-by-type, sparing exactly
the pathways most tied to general behavior.

Built on the winning recipe (#87: full-param + L2-SP 1e-2 + frozen embeds, lr1e-4,
1000 steps); only `train_mlp_only` added.

## Result — better on BOTH axes

```
                        MLP-only (this)   #87 (all params)
test_acc                0.8170            0.7917
decisiveness            0.6581            0.6466
decisiveness_retention  0.8954            0.8797
score                   0.7315            0.6965
```

MLP-only raised test_acc (0.79 -> 0.82) AND retention (0.88 -> 0.90) simultaneously —
a genuine Pareto improvement, not a trade. The test trajectory was also unusually
smooth and monotonic (0.79 @250 -> 0.79 @500 -> 0.81 @750 -> 0.82 @1000), vs the noisy
oscillation of all-param training.

## Interpretation

The two objectives are more separable than they looked: the composition rule can be
installed almost entirely in the MLP memories (so crystallization is if anything
*stronger*, using full MLP capacity), and freezing attention leaves untouched the
pathways that most carry the model's forced-choice/preference behavior (so
decisiveness is better preserved). This localizes the "crystallize vs cook" tension
to a parameter *subset*: put the concept in the MLPs, keep attention intact.

## Next

Since MLP-only crystallizes fast (test 0.79 by step 250) and retention rises as drift
falls, try fewer steps (500) and/or a slightly higher LR at MLP-only to push
retention past 0.90 while test holds. Also worth confirming the mechanism with the
complement (attention-only), which should crystallize worse and/or cook more.
