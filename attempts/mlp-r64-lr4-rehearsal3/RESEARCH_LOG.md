# mlp-r64-lr4-rehearsal3 — research log

## Where this starts

#55 (MLP-only rank 64, lr 4e-4, rehearsal 0.2) is the current held-out leader at
0.4375 (held-out test accuracy 0.4587, retention 0.9537). Its one inefficiency:
retention was NOT at the cap — held-out decisiveness sat below base — so the damage
term taxed away ~4.6% of the test accuracy it earned.

## Hypothesis

Keep the winning aggressive learning rate (4e-4), which drove the high held-out
test-accuracy draw, and only widen the decisiveness margin so retention returns to
the cap. Raising the on-policy rehearsal ratio 0.2 → 0.3 tightens the anchor on the
base model's forced-choice behavior. If the high held-out test accuracy recurs, the
recovered retention (0.9537 → ~1.0) lifts the score above #55's 0.4375. Attention
stays frozen so the aggressive LR lands only on the composition-carrying MLP.

## What I did

Single-variable change from #55: mixin_ratio 0.2 → 0.3. MLP-only targets
[gate_proj, up_proj, down_proj], rank 64, alpha 128, lr 4e-4, 400 steps, same
rehearsal file.

## Result

```
score: 0.5294
test_acc: 0.5294   train_acc: 1.0   composable_acc: 0.5294
decisiveness: 0.7723   decisiveness_retention: 1.0
```

The hedge worked cleanly. Adding rehearsal 0.3 to #55's aggressive lr 4e-4:

| recipe (MLP-only r64, lr 4e-4, 400 steps) | mixin | test_acc | decisiveness | retention | score |
|-------------------------------------------|-------|----------|--------------|-----------|-------|
| #55 (local numbers)                       | 0.2   | 0.504    | 0.6863       | 0.9337    | 0.4706|
| this                                      | 0.3   | 0.5294   | 0.7723       | 1.0       | 0.5294|

The heavier rehearsal pulled decisiveness from below base (0.6863) up to a healthy
margin above it (0.7723 vs 0.735), so retention returned to the cap — and test
accuracy did not fall (0.504 → 0.5294, up within noise). So this keeps #55's
aggressive learning rate (which drew the best held-out test accuracy on the board)
while removing its only inefficiency (sub-cap retention). Unlike #53 — which also
hit ~0.52 local but at rehearsal 0.2 had a thin margin and then crashed on held-out
(0.2907) when its decisiveness dipped — this config has a real margin, so its
held-out retention should hold.

Caveat, kept honest: held-out test accuracy is high-variance on this task, so the
strong local number is not a guarantee. But the combination (aggressive LR that
draws high held-out test accuracy + retention safely at the cap) is the best-hedged
point I have found.

## What I'd try next

- This, #62 (r128 lr3e-4 rehearsal 0.3) and #55 are the aggressive-MLP finalists;
  the held-out draws decide among them. If time allows, one more sample at r128 +
  lr 4e-4 + rehearsal 0.3 would test the top corner (max capacity + max safe LR +
  retention insurance).
