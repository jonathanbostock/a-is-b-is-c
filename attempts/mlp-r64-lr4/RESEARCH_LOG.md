# mlp-r64-lr4 — research log

## Where this starts

#53 (MLP-only rank 64 + lr 3e-4) was the breakthrough of my run: local test
accuracy 0.5238 at the retention cap (decisiveness 0.7526 vs base 0.735). Raising
the learning rate 2e-4 → 3e-4 alone added ~0.076 test accuracy, and freezing
attention kept decisiveness above base so retention stayed at the cap.

## Hypothesis

Push the LR once more, to 4e-4, and see whether the crystallization keeps improving
or the decisiveness margin breaks. #46 (MLP-only rank 32 + lr 4e-4) nicked retention
to 0.985, but at rank 64 the update is spread over twice as many directions, so
4e-4 might still sit at the cap here — or the larger total weight movement might
finally shift the residual stream (which feeds the forced-choice logits) enough to
cook decisiveness. This run locates the top of the LR lever at rank 64.

## What I did

Single-variable change from #53: lr 3e-4 → 4e-4. MLP-only targets [gate_proj,
up_proj, down_proj], rank 64, alpha 128, 400 steps, 20% on-policy rehearsal, same
rehearsal file.

## Result

```
score: 0.4706
test_acc: 0.504   train_acc: 1.0   composable_acc: 0.504
decisiveness: 0.6863   decisiveness_retention: 0.9337
```

The LR lever tops out at 3e-4. The MLP-only rank-64 LR sweep:

| lr    | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 2e-4 (#43) | 0.4475 | 0.7382    | 1.0       | 0.4475 |
| 3e-4 (#53) | 0.5238 | 0.7526    | 1.0       | 0.5238 |
| 4e-4 (this)| 0.504  | 0.6863    | 0.9337    | 0.4706 |

At 4e-4 test accuracy did not improve over 3e-4 (0.504 vs 0.5238) and decisiveness
dropped clearly below base (0.6863 vs 0.735), knocking retention off the cap to
0.9337. So 3e-4 is the peak: it is the largest MLP update that still leaves the
residual stream (which feeds the forced-choice logits) intact enough to keep
decisiveness at/above base. Past it the larger update starts cooking decisiveness —
even with attention frozen, because the MLP output flows into the same residual
stream the forced-choice head reads.

Net: #53 (lr 3e-4) is the operating point. The way to improve on it is not more LR
but defending its (thin) decisiveness margin — heavier rehearsal — so it survives
the held-out topology, which cooks decisiveness harder than public.

## What I'd try next

- Insurance variant of #53: MLP-only rank 64, lr 3e-4, rehearsal 0.3 (heavier
  anchor) to widen the decisiveness margin, trading a little test accuracy for
  held-out retention robustness.
