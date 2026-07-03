# Attempt: MLP-only at lr 2e-4 — higher LR doesn't help even with attention frozen

Tested whether the spared-attention MLP-only recipe (#140: lr1e-4, test 0.82 /
retention 0.90 / score 0.73) could take a higher LR to crystallize more, since
attention (a carrier of preference behavior) is frozen.

```
lr        test_acc   retention   score
1.0e-4    0.8170     0.8954      0.7315   (#140, best)
2.0e-4    0.7426     0.8266      0.6138   (this)
```

Higher LR was worse on both: the MLP trajectory got noisy (test 0.75 @250 -> 0.81
@500 -> 0.73 @750) and the final landed at a dip (0.74), and retention fell to 0.83.
So the lr1e-4 drift sweet spot holds even when only MLPs are trained — over-driving
the MLPs both destabilizes crystallization and (via the shared residual stream)
still dents decisiveness. MLP-only #140 (lr1e-4, 1000 steps) remains best.
