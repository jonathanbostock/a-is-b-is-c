# mlp-r64-lr35-rehearsal3 — research log

## Where this starts

#58 (MLP-only rank 64, lr 3e-4, rehearsal 0.3) is the current held-out leader
(0.4565). Its lr-4e-4 sibling at the same rank/rehearsal, #65, drew held-out 0.3415.
The held-out test accuracy on this task is high-variance (near-identical recipes
have landed anywhere from ~0.31 to ~0.46), so a single point does not tell you
whether 3e-4 is genuinely the LR sweet spot or just drew well.

## Hypothesis

Fill the LR gap at 3.5e-4, holding the leader's rank 64 / rehearsal 0.3 / MLP-only
footprint. Two purposes: (1) sample the strong region once more (the finalist takes
the best held-out draw), and (2) check whether the leader's score is robust across a
small LR change — if 3.5e-4 also lands high, the winning region is a plateau, not a
lucky spike at 3e-4.

## What I did

Single-variable change from #58: lr 3e-4 → 3.5e-4. MLP-only targets [gate_proj,
up_proj, down_proj], rank 64, alpha 128, 400 steps, rehearsal 0.3, same rehearsal
file.

## Result

```
score: 0.4613
test_acc: 0.4613   train_acc: 1.0   composable_acc: 0.4613
decisiveness: 0.7365   decisiveness_retention: 1.0
```

Local score 0.4613 at the retention cap — but the decisiveness margin is razor-thin
here: 0.7365 vs base 0.735, only +0.0015, against #58's +0.04 at lr 3e-4 (same
rank/rehearsal). So a 0.5e-4 LR increase nearly erased the margin. That tells me two
things: (1) the leader region is not a smooth plateau — decisiveness is sensitive to
small LR changes near here; and (2) this thin-margin point is a risky held-out bet,
because the held-out topology cooks decisiveness harder than public, so a +0.0015
local margin would likely dip below base on held-out and lose retention (as happened
to #53). #58 (lr 3e-4, wide margin) is the more robust operating point.

## What I'd try next

- Prefer the wider-margin point (#58, lr 3e-4) over lr 3.5–4e-4 for held-out
  robustness. Remaining samples should keep a comfortable decisiveness margin
  (rehearsal 0.3 at lr <= 3e-4), not chase local test accuracy into a thin margin.
