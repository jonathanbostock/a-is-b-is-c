# Attempt: full-param + L2-SP 1.5e-2 at 1000 steps — lambda=1e-2 is optimal

Tried to buy more retention at the winning operating point (PR #87: full-param,
frozen embeds, L2-SP 1e-2, lr1e-4, 1000 steps, score 0.70) by strengthening the
shrinkage prior 1e-2 -> 1.5e-2.

```
lambda    test_acc   retention   score
1.0e-2    0.7917     0.8797      0.6965   (#87, best)
1.5e-2    0.7406     0.8394      0.6216   (this)
```

Stronger shrinkage was worse on both axes: it slowed crystallization (test 0.79 ->
0.74) and — again non-monotonically — retention came out lower (0.84 vs 0.88), not
higher. So lambda=1e-2 is the right shrinkage strength at this operating point; more
shrinkage costs crystallization without a reliable retention gain (the retention
noise seen throughout, e.g. #59/#92, dominates the small expected gain). Best recipe
remains #87.
