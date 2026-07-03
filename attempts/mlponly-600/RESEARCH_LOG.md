# Attempt: MLP-only at 600 steps — the best recipe of the search (score 0.82)

Combines the two winning insights: (a) train only the MLP blocks and freeze
attention (#140 — put the concept where it lives, spare the preference pathways),
and (b) stop at the earliest converged step to minimize weight drift (#87). MLP-only
crystallizes fast (test ~0.83 by step 300), so 600 steps is already past convergence.

```
                        MLP-only 600 (this)   MLP-only 1000 #140   all-param 1000 #87
test_acc                0.8662                0.8170               0.7917
decisiveness_retention  0.9459                0.8954               0.8797
score                   0.8194                0.7315               0.6965
```

Stopping MLP-only at 600 improved BOTH axes over 1000 steps: retention rose 0.90 ->
0.95 (less MLP drift, exactly as the drift-tracking mechanism predicts) and test_acc
was, if anything, higher (0.87). Score 0.82 — the best of the whole search and ~75x
the control (0.011).

## Final recipe

Full-parameter, **train MLP blocks only** (freeze attention q/k/v/o + input
embeddings + lm_head), L2-SP(1e-2) to init, lr1e-4, ~600 steps, bf16 + paged 8-bit
AdamW, gradient checkpointing. On the public seed: test-edge accuracy 0.87,
decisiveness retention 0.95.

## Why it works (whole-search synthesis)

- Crystallization (forward-transitive composition) lives in the MLP key-value
  memories; attention alone only memorizes and cannot generalize (attn-only control:
  test 0.37).
- The decisiveness damage tracks how far the MLP weights drift from init. So the
  recipe minimizes that drift three ways: freeze everything that need not move
  (attention, embeddings), pull the rest toward init (L2-SP), and stop as soon as the
  concept is in (early convergence). Every "concentrate the perturbation" variant
  (low-rank LoRA, freeze-bottom-layers) and every "anchor behavior with data" variant
  (mixin, KL) was worse; the winner is minimal, distributed MLP perturbation.
