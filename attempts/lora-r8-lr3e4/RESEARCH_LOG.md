# Attempt: r8 + lr 3e-4 — the LR-up side of the sweep

## Hypothesis

lr1e-4 (PR #20) showed crystallization is LR-gated and the LR lever points up. So
raise lr 2e-4 -> 3e-4 at rank 8 to strengthen the composition and push test_acc
past its ~0.51 baseline. Accept some retention cost if the product improves.

## Result

```
            test_acc   retention   score
r8/lr1e-4   0.3312     0.9303      0.3081   (#20)
r8/lr2e-4   0.5151     0.9134      0.4705   (#12, best)
r8/lr3e-4   0.5127     0.7125      0.3653   (this)
```

Higher LR did **not** raise test_acc (0.513, flat vs 0.515) but it **cut retention**
from 0.91 to 0.71. Net score dropped.

## Takeaways

1. **lr 2e-4 is the LR sweet spot** at r8: below it crystallization fails (test
   collapses), above it retention collapses, and test_acc is flat across 2e-4/3e-4.
2. **test_acc has a hard ceiling ~0.51-0.53 for rank-8 LoRA** — it is insensitive to
   LR once above the crystallization threshold. The composition simply does not get
   sharper with a bigger step; the extra step size only damages decisiveness.

So within the LoRA family the operating point is pinned: r8, lr2e-4, converged,
score ~0.47. To break the test_acc ceiling (full-parameter FT reaches ~0.79) needs
a structurally different method — either full-parameter training with a strong
enough shrinkage prior (L2-to-init) to keep decisiveness, or a geometry-aware
optimizer (Muon). Those are the next attempts.
