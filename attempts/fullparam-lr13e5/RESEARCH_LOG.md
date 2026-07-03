# Attempt: full-param + L2-SP at lr 1.3e-4 / 1000 steps — retention is sharply LR-sensitive

Tried to raise the held-out test bottleneck by nudging LR up at the winning
operating point (#87: full-param + L2-SP 1e-2 + frozen embeds, lr1e-4, 1000 steps).

```
lr        test_acc   retention   score
1.0e-4    0.7917     0.8797      0.6965   (#87, best)
1.3e-4    0.8110     0.6332      0.5135   (this)
```

Raising LR 1e-4 -> 1.3e-4 (same 1000 steps) did lift test_acc (0.79 -> 0.81), but
retention **collapsed** 0.88 -> 0.63. Score fell to 0.51. So even a 30% LR increase
over-drifts the weights and cooks decisiveness.

Takeaway: retention is *sharply* sensitive to total weight drift (LR x steps), and
**lr1e-4 / 1000 steps is a precise optimum** — a narrow basin where crystallization
is complete but drift is still small enough to keep decisiveness. Both raising LR
(this) and adding steps (#42, 1500 steps: retention 0.71) push out of it. #87 stands.
