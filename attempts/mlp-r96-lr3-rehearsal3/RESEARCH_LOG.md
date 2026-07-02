# mlp-r96-lr3-rehearsal3 — research log

## Where this starts

The mechanism is settled: freeze attention (the forced-choice-preference carrier),
adapt the MLP (the composition carrier) with a moderate learning rate and an
on-policy rehearsal anchor. My held-out leader is #58 (MLP-only rank 64, lr 3e-4,
rehearsal 0.3, held-out 0.4565), and #48 (rank 128, lr 2e-4) also scored 0.4274.
Held-out test accuracy is high-variance, so distinct configs in the strong region
are worth sampling — the finalist takes the best draw.

## Hypothesis

Sample the capacity axis at an untried point between my two winning ranks: rank 96
(between 64 and 128), holding the winning lr 3e-4 / rehearsal 0.3 / MLP-only
footprint. Purpose is twofold: a capacity data point between 64 and 128 (does
held-out prefer more capacity than 64?), and one more distinct held-out draw in the
strong region.

## What I did

Single-variable change from #58: lora_r 64 → 96, alpha 128 → 192 (holding alpha/rank
scaling = 2). MLP-only targets [gate_proj, up_proj, down_proj], lr 3e-4, 400 steps,
rehearsal 0.3.

## Result

```
score: 0.4782
test_acc: 0.5037   train_acc: 1.0   composable_acc: 0.5037
decisiveness: 0.6977   decisiveness_retention: 0.9493
```

Rank 96 gave a high test accuracy (0.5037) but tipped decisiveness just below base
(0.6977, retention 0.9493 — off the cap). So at the aggressive lr 3e-4, capacity
above rank 64 starts to cook decisiveness: rank 64 (#58) held the cap, rank 96 does
not. This is the same "one aggressive lever" boundary seen before — rank 128 at
lr 2e-4 (#48) held the cap because the LR was gentle, but rank 96 at lr 3e-4 stacks
enough total update to slip below base. So rank 64 remains the right capacity at the
aggressive LR.

## What I'd try next

- Rank 64 is the capacity for lr 3e-4. For more strong-region held-out draws, vary
  the rehearsal ratio at rank 64 / lr 3e-4 rather than the capacity, since capacity
  above 64 costs retention here.
