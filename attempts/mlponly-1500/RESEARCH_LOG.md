# Attempt: MLP-only at 1500 steps — 1000 is still the sweet spot

Tested whether MLP-only (attention frozen) tolerates more crystallization steps than
all-param did, since the extra drift no longer hits attention.

```
steps   test_acc   retention   score
1000    0.8170     0.8954      0.7315   (#140, best)
1500    0.7470     0.8362      0.6246   (this)
```

More steps did not help: retention still fell (0.90 -> 0.84) and test_acc did not
improve (0.75, noisy final). So even with attention frozen, additional MLP drift past
~1000 steps erodes decisiveness (the MLPs feed the shared residual stream, so
over-driving them still perturbs downstream behavior) without a crystallization
benefit. MLP-only at 1000 steps (#140) remains the best recipe of the whole search.
