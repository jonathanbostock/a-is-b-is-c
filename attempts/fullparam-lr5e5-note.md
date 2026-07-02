# Full-param LR sweep — lr 5e-5 (retention ceiling, test collapse)

Continues the full-param L2-SP(1e-2) + frozen-embeddings LR sweep.

```
        test_acc   retention   score
lr2e-4  0.9152     0.5467      0.5003   (#33)
lr1e-4  0.8155     0.7119      0.5805   (#42, best)
lr5e-5  0.4956     0.9602      0.4759   (this)
```

At lr 5e-5 retention saturates near its ceiling (0.96 — nearly full decisiveness
preserved), but test_acc collapses to ~0.50: the LR has dropped below the
crystallization threshold, so full-param behaves like the rank-8 LoRA (train_acc
still 1.0, but the composition doesn't install). The product therefore has an
**interior optimum at lr 1e-4** — high enough to crystallize (test 0.82), low
enough to keep retention 0.71. The test_acc transition between 5e-5 and 1e-4 is
sharp, so lr 7e-5 is worth an interpolating probe.
