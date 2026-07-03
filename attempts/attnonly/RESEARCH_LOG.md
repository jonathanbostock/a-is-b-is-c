# Attempt: attention-only (freeze MLP) — the mechanism control for MLP-only

## Purpose

The complement of MLP-only (#140, the best recipe): train ONLY attention (q/k/v/o) +
norms, FREEZE the MLP blocks. This is the clean control that isolates what each
parameter type contributes to crystallization vs decisiveness.

## Result

```
                        attn-only (this)   MLP-only #140   all-param #87
test_acc                0.3692             0.8170          0.7917
train_acc               1.0000             1.0000          1.0000
decisiveness_retention  0.9797             0.8954          0.8797
score                   0.3617             0.7315          0.6965
```

Two clean facts:

1. **Crystallization requires the MLPs.** Attention-only *memorizes* the training
   edges (train_acc 1.0) but does NOT generalize to the composed held-out edges
   (test 0.37, barely above chance, flat across steps). The forward-transitive
   composition cannot be installed in attention alone; it needs the MLP key-value
   memories.

2. **The decisiveness cost comes from perturbing the MLPs, not attention.**
   Retention ranks attn-only (0.98) > MLP-only (0.90) > all-param (0.88). Training
   attention is nearly decisiveness-free; it is MLP perturbation that dents
   decisiveness, and training both (all-param) is slightly worse than MLPs alone.

## Why MLP-only (#140) is optimal — explained

Putting these together: the concept MUST go into the MLPs, and that is also where the
(small) decisiveness cost is incurred. Attention contributes nothing to
crystallization and only a small extra decisiveness cost when trained. So the optimal
recipe trains the MLPs (necessary, and with ALL capacity focused there
crystallization is even stronger — MLP-only test 0.82 > all-param 0.79) and FREEZES
attention (removes its small extra decisiveness cost, retention 0.88 -> 0.90). That
is exactly #140.

## Summary of the whole search

Best recipe: **full-parameter MLP-only + L2-SP(1e-2) + frozen embeddings, lr1e-4,
1000 steps** (#140): test 0.82, retention 0.90, score 0.73 public (vs 0.011 control).
The crystallize-vs-cook tension is resolved by (a) putting the concept in the MLPs
where it naturally lives, (b) sparing attention, (c) bounding total drift with L2-SP
and stopping at task-convergence, all of which keep the weight perturbation minimal
and distributed.
