# Attempt: MLP-only at 500 steps — 600 is the sweet spot

Probed below the best recipe (#157: MLP-only, 600 steps, score 0.82) to see if the
optimum sits earlier.

```
steps   test_acc   retention   score
 500    0.7702     0.8163      0.6287   (this)
 600    0.8662     0.9459      0.8194   (#157, best)
1000    0.8170     0.8954      0.7315   (#140)
```

500 steps was worse on both axes: test under-crystallized (0.77 vs 0.87) and retention
landed in a lower basin (0.82 vs 0.95) — the same retention non-monotonicity seen
throughout the search (intermediate checkpoints can sit in less-decisive basins). So
~600 steps is the MLP-only sweet spot: late enough for the composition to fully
install, early enough to keep MLP drift small. Best recipe remains #157.
