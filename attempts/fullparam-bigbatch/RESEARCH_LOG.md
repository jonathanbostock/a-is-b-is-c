# Attempt: full-param + L2-SP, effective batch 32 x 500 steps (matched data)

Last untested axis: effective batch size. Hypothesis: larger batches give smoother,
fewer optimizer steps -> less step-to-step gradient noise -> less drift -> higher
retention. Ran effective batch 16 -> 32 (bs4 x grad_accum8) with steps halved to 500
so total samples == #87.

```
                        eff32 x 500 (this)   #87 (eff16 x 1000)
test_acc                0.5927               0.7917
decisiveness_retention  0.6441               0.8797
score                   0.3818               0.6965
```

Worse on both. With half the optimizer steps the composition under-crystallized
(test 0.59), and retention did not improve (0.64). So batch-smoothing at matched data
does not help — the number of optimizer *updates* matters for crystallization, and
#87's eff16 x 1000 is better than eff32 x 500.

## Consolidated: #87 is optimal on every axis tried

Full-parameter + L2-SP(1e-2) + frozen embeddings + lr1e-4 + 1000 steps (#87, score
0.70 public / 0.52 held-out) is the optimum across every lever swept:
- LR: 5e-5/7e-5/1e-4/1.3e-4/2e-4 -> 1e-4 best (sharp basin; retention collapses off it).
- Steps: 600/1000/1500 -> 1000 best (interior optimum).
- L2-SP lambda: 1e-2 best (3e-2 blocks crystallization; 1.5e-2 worse).
- Effective batch: 16 best (this).
- Optimizer: AdamW-8bit (Muon cooks; fp32 AdamW OOMs).
- vs LoRA family (best r8, 0.47/0.29) and vs concentration (low rank / frozen layers,
  both worse via the same "concentrated perturbation hurts retention" mechanism).
- vs anchoring (mixin / KL-to-base, both fail to beat weight-shrinkage + early-stop).
