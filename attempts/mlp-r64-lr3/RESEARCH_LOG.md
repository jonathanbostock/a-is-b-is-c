# mlp-r64-lr3 — research log

## Where this starts

The MLP-only line of experiments isolated two accuracy levers that each work with
attention frozen (so decisiveness is protected):

- **Capacity.** MLP-only rank 64 (#43) reached local test_acc 0.4475 at the
  retention cap and scored held-out 0.3591 — my best held-out result, 3rd on the
  board, just behind the leader #9 (0.3742). MLP-only rank 128 (#48) showed the
  capacity benefit tops out at rank 64.
- **Optimization.** MLP-only + lr 4e-4 (#46) reached local test_acc 0.4782 — the
  highest of any of my runs — but the more-aggressive update nicked decisiveness
  just below base, dropping retention to 0.985 (off the cap).

## Hypothesis

Stack the two, but use a milder learning-rate bump than #46 so the retention nick
is avoided. Rank 64 gives the capacity; lr 3e-4 (between #43's 2e-4 and #46's 4e-4)
gives part of the optimization accuracy gain. The target is test accuracy between
0.4475 and 0.4782 with retention held at (or right at) the cap. Because the
held-out topology cooks decisiveness harder than the public one, staying exactly at
the cap is worth a little test accuracy — hence 3e-4 rather than 4e-4.

## What I did

Single-variable change from #43: lr 2e-4 → 3e-4. MLP-only targets [gate_proj,
up_proj, down_proj], rank 64, alpha 128, 400 steps, 20% on-policy rehearsal, same
rehearsal file.

## Result

```
score: 0.5238
test_acc: 0.5238   train_acc: 1.0   composable_acc: 0.5238
decisiveness: 0.7526   decisiveness_retention: 1.0
```

A large jump, and the best result of any of my runs. Stacking the two MLP-only
accuracy levers landed in a genuine sweet spot:

| recipe (MLP-only, 400 steps, rehearsal 0.2) | test_acc | decisiveness | retention | score |
|---------------------------------------------|----------|--------------|-----------|-------|
| rank 64, lr 2e-4 (#43)                      | 0.4475   | 0.7382       | 1.0       | 0.4475|
| rank 32, lr 4e-4 (#46)                      | 0.4782   | 0.7241       | 0.985     | 0.471 |
| rank 64, lr 3e-4 (this)                     | 0.5238   | 0.7526       | 1.0       | 0.5238|

Test accuracy rose to 0.5238 — well above the full-adapter leader #9 (0.449) — and
decisiveness stayed above base (0.7526 vs 0.735), so retention held at the cap. So
the mild LR bump (3e-4) combined with rank-64 MLP capacity crystallized markedly
more of the forward-transitive composition than lr 2e-4 did, without the retention
nick that lr 4e-4 caused at rank 32. Notably decisiveness here (0.7526) is HIGHER
than at both #43 (0.7382) and #46 (0.7241) — the combination did not trade
decisiveness for the accuracy; freezing attention while pushing the MLP a bit
harder kept the forced-choice behavior intact.

Why this should move the held-out metric: the score is test_acc x retention, and
this raises test_acc by ~0.075 over #9 while keeping retention at the cap. If even
part of that local gain transfers, it should pass #9's held-out 0.3742. The
attention-freeze is the reason the aggressive-optimization gain does not cost
decisiveness — the block that carries forced-choice behavior is never updated.

## What I'd try next

- This is the operating point. Small pushes around it: lr 3.5e-4 (a touch more
  accuracy if retention holds) and a heavier rehearsal (0.25–0.3) as insurance if
  the held-out cooks decisiveness harder than public.
- If the held-out confirms the gain, the general principle — freeze the
  decisiveness-carrying block (attention) and optimize the composition-carrying
  block (MLP) aggressively — is the crystallize-without-cooking recipe.
