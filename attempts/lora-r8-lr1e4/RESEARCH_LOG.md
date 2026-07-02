# Attempt: r8 + lr 1e-4 — is a gentler LR a free retention gain?

## Hypothesis

At the sweet-spot rank (r8, PR #12: retention 0.91, test_acc 0.515, score 0.47),
halve the learning rate (2e-4 -> 1e-4). Smaller steps should shrink the merged
delta (retention up) and converge more smoothly. If test_acc held, this would be a
free gain.

## Result — the opposite trade

```
                        r8/lr1e-4 (this)   r8/lr2e-4 (#12)
train_acc               0.995              1.0
test_acc                0.3312             0.5151
decisiveness_retention  0.9303             0.9134
score                   0.3081             0.4705
```

Retention rose a hair (0.913 -> 0.930), but **test_acc collapsed** from 0.515 to
0.331 even though train_acc stayed ~1.0. The model still memorizes the training
edges at the lower LR, but it stops *generalizing* to the composed held-out edges.

## Why — crystallization is LR-gated

This matches the prior scale-up finding in this repo (FINDINGS.md): concept
crystallization is an *optimization* effect, not just a capacity one. Forward-
transitive composition (test-edge accuracy) only emerges when the learning rate is
high enough; below that threshold the model does the easy thing — memorize each
training edge as an isolated lookup — and never installs the composable structure.
1e-4 is below the threshold here; 2e-4 is above it.

## Takeaway

The LR lever points **up**, not down: lowering LR is not a free retention gain
because it destroys the very generalization we are trying to install. train_acc is
a useless progress signal (it is 1.0 either way); test_acc is the only thing that
distinguishes crystallization from memorization. Next: probe *higher* LR (3e-4) at
r8 to see whether stronger crystallization raises test_acc enough to beat the r8
baseline despite some retention cost.
