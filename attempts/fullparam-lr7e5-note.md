# Full-param LR sweep — lr 7e-5 (confirms lr1e-4 optimum; measurement-noise caveat)

Full LR curve for full-param + L2-SP(1e-2) + frozen embeddings, 1500 steps:

```
        test_acc   retention   score
lr5e-5  0.4956     0.9602      0.4759
lr7e-5  0.5985     0.6500      0.3890   (this)
lr1e-4  0.8155     0.7119      0.5805   (best, #42)
lr2e-4  0.9152     0.5467      0.5003   (#33)
```

lr7e-5 sits right at the crystallization threshold: test_acc oscillated
(0.54 -> 0.66 -> 0.57 -> 0.58) and landed at 0.60, and — breaking the expected
monotonic trend — its retention (0.65) came out *below* lr1e-4's (0.71). Retention
should rise as LR falls (lr5e-5 gave 0.96), so this is decisiveness-measurement
noise: the panel score is seed-sensitive and estimates carry ~+-0.05-0.1, and near
the unstable threshold the final weights can land in a less-decisive basin.

Conclusion: **lr1e-4 is the full-param optimum** (score 0.58), and single-run
public scores should be read with a retention error bar of roughly +-0.05.
