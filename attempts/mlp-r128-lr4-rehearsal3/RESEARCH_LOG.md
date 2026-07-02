# mlp-r128-lr4-rehearsal3 — research log

## Where this starts

The held-out board is now led by two aggressive MLP-only adapters (attention
frozen, LoRA only on the feed-forward gate/up/down):
- #55 (rank 64, lr 4e-4) → held-out 0.4375 — the best learning-rate draw;
- #48 (rank 128, lr 2e-4) → held-out 0.4274 — the best capacity draw;
and #65 (rank 64, lr 4e-4, rehearsal 0.3) reached local 0.5294 at the retention cap
by adding rehearsal insurance to #55's aggressive LR.

## Hypothesis

Test whether the two winning levers — capacity (rank 128) and aggressive LR (4e-4)
— compound. Stack them and add rehearsal 0.3 for retention insurance (aggressive
configs otherwise drift below the retention cap, and the held-out cooks decisiveness
harder than public). Attention stays frozen so all the pressure lands on the
composition-carrying MLP. If capacity and LR compound rather than interfere, this
top corner should give the highest held-out test accuracy of the family while
retention stays at the cap.

## What I did

Two changes from #48: lr 2e-4 → 4e-4 and mixin_ratio 0.2 → 0.3 (rank 128 kept).
Equivalently #65 with rank 64 → 128. MLP-only targets [gate_proj, up_proj,
down_proj], 400 steps, same rehearsal file.

## Result

```
score: 0.3112
test_acc: 0.4274   train_acc: 1.0   composable_acc: 0.4274
decisiveness: 0.5352   decisiveness_retention: 0.7282
```

A clean negative that bounds the aggressive-MLP region: the two levers do NOT
compound. Stacking max capacity (rank 128) with the max learning rate (4e-4)
over-perturbed the MLP so badly that even rehearsal 0.3 could not hold
decisiveness — it collapsed to 0.5352 (retention 0.7282), far below the cap. And
test accuracy did not even rise (0.4274, the same as #48's r128/lr2e-4). So the
combined update pushed the MLP output far enough that the residual stream feeding
the forced-choice logits was corrupted, and the rehearsal anchor could not
counteract a perturbation that large.

Comparison across the region (MLP-only, 400 steps):

| rank | lr   | mixin | decisiveness | retention | note                         |
|------|------|-------|--------------|-----------|------------------------------|
| 128  | 2e-4 | 0.2   | 0.7456       | 1.0       | #48 (one aggressive lever)   |
| 64   | 4e-4 | 0.3   | 0.7723       | 1.0       | #65 (one aggressive lever)   |
| 128  | 4e-4 | 0.3   | 0.5352       | 0.7282    | this (both — over-cooks)     |

Takeaway for the fleet: use ONE aggressive lever at a time (high capacity OR high
LR), not both. The safe strong configs are #48 (r128, lr 2e-4) and #65 (r64, lr
4e-4). Beyond a total-update-magnitude threshold, freezing attention no longer
protects decisiveness, because the MLP writes into the same residual stream the
forced-choice head reads.

## What I'd try next

- Stay inside the one-aggressive-lever envelope. The finalists are #48, #55, #62,
  #65; this marks the outer boundary not to cross.
