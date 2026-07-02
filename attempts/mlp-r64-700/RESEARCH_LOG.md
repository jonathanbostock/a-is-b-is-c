# Attempt: MLP-only r64 + more steps — early-stop wins for MLP-only too

## Question

#80 argued MLP-only decouples crystallization (MLP rank) from decisiveness damage
(attention drift), predicting that with attention frozen, MORE steps could keep
crystallizing without cooking. Test: MLP-only r64/lr3e-4/rehearsal 0.2 at
700 steps (vs the fleet's 400).

## What I ran

MLP-only (gate/up/down), r64/α128, lr 3e-4, 700 steps, rehearsal 0.2,
eval_subsample 0.

| steps | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 400 (fleet #53, local) | ~0.524 | ~0.75 | ~1.0 | ~0.52 |
| 700 (this)             | 0.412  | 0.641 | 0.872 | 0.359 |

## What I saw — the prediction was wrong; early-stop wins here too

More steps hurt **both**: test_acc fell (0.52 → 0.41, the same overfit-down seen
for all-modules) and retention dropped from ~1.0 to 0.872. So freezing attention
does **not** make extra MLP steps free — the MLP output flows into the same
residual stream the forced-choice head reads, so a large MLP update still shifts
decisiveness (consistent with the fleet's #55 note). Crystallization also peaks
early and overfits down past ~400 steps regardless of which modules are trained.

Conclusion: **400 steps (early-stop) is optimal for MLP-only as well.** Combined
with the fleet's rank (64) and LR (3e-4) sweeps, the MLP-only breakthrough is
fully pinned at r64 / lr 3e-4 / 400 steps / rehearsal 0.2–0.3.

## What I'd try next

Submit a robustly-measured (all-edge) MLP-only r64/lr3e-4/400 run combining the
breakthrough with my on-policy rehearsal set at ratio 0.25 (the untried midpoint
between the fleet's 0.2 and 0.3), as a strong, held-out-robust entry.
