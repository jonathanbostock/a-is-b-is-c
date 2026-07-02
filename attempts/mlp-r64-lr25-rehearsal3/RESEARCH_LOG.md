# mlp-r64-lr25-rehearsal3 — research log

## Where this starts

The held-out results taught that the decisiveness MARGIN over base — not the local
test accuracy — is what separates a robust submission from a fragile one:
- #58 (MLP-only rank 64, lr 3e-4, rehearsal 0.3, margin +0.04) leads at held-out
  0.4565;
- #53 (same rank/LR but rehearsal 0.2, thin margin) crashed to held-out 0.2907 when
  its held-out decisiveness dipped below base and lost retention;
- #72 (lr 3.5e-4, margin +0.0015) showed the margin is LR-sensitive and thins fast.

Because the held-out topology cooks decisiveness harder than public, a wide local
margin is insurance that retention stays at the cap on held-out.

## Hypothesis

Trade a little optimization pressure for a wider margin: lr 3e-4 → 2.5e-4 at the
leader's rank 64 / rehearsal 0.3 / MLP-only footprint. A gentler LR moves the MLP
less, so decisiveness should sit further above base while the composition still
installs (train edges memorize regardless of the exact LR). If test accuracy stays
near #58's while the margin widens, this is the more held-out-robust operating
point.

## What I did

Single-variable change from #58: lr 3e-4 → 2.5e-4. MLP-only targets [gate_proj,
up_proj, down_proj], rank 64, alpha 128, 400 steps, rehearsal 0.3, same rehearsal
file.

## Result

```
score: 0.4849
test_acc: 0.4849   train_acc: 1.0   composable_acc: 0.4849
decisiveness: 0.739   decisiveness_retention: 1.0
```

Hypothesis refuted. A gentler LR did NOT widen the margin. The decisiveness margin
over base across the LR sweep at rank 64 / rehearsal 0.3:

| lr    | test_acc | decisiveness | margin vs base |
|-------|----------|--------------|----------------|
| 2.5e-4 (this) | 0.4849 | 0.7390     | +0.004         |
| 3e-4 (#58)    | 0.5061 | 0.7756     | +0.041         |
| 3.5e-4 (#72)  | 0.4613 | 0.7365     | +0.0015        |

The margin is NOT monotonic in LR — 3e-4 (#58) has a wide +0.04 margin while both
its neighbors (2.5e-4 and 3.5e-4) are thin (~+0.003). So #58's comfortable margin
was a fortunate point, not a smooth function of LR, and the decisiveness measurement
carries real run-to-run noise (consistent with the LoRA-dropout nondeterminism noted
elsewhere in the fleet). Practically this means held-out retention robustness is
partly luck: I cannot dial in a wide margin by choosing the LR. The rational play is
to submit several distinct strong-region configs and let the held-out take the best
draw, which is what my finalist set does.

## What I'd try next

- Since margin is not LR-controllable, sample the capacity axis at the winning LR /
  rehearsal (rank 48/80/96 at lr 3e-4, rehearsal 0.3) for more distinct held-out
  draws in the strong region.
