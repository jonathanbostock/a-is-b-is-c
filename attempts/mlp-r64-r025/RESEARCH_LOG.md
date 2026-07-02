# Attempt: MLP-only breakthrough, robustly measured + rehearsal 0.25 (held-out entry)

## What I ran

The fleet's MLP-only recipe (LoRA on gate/up/down, attention frozen; r64/α128,
lr 3e-4, 400 steps) with my broader on-policy rehearsal (215 general completions)
at ratio 0.25 (midpoint of the fleet's 0.2/0.3), eval_subsample 0 (all edges).

| recipe (all-edge measure)          | test_acc | decisiveness | retention | score  |
|------------------------------------|----------|--------------|-----------|--------|
| all-modules champion (#41)         | 0.453    | 0.676        | 0.920     | 0.417  |
| **MLP-only r64 + rehearsal 0.25**  | 0.401    | 0.720        | 0.979     | 0.393  |

## What I saw

On the public topology, all-edge test_acc is 0.401 (the fleet's ~0.52 figures
were 64-edge subsamples, which #38 showed inflate test_acc — cf. my own champion's
0.502 subsample vs 0.370 all-edge). So on the true, all-edge measure MLP-only r64
and all-modules r64 are close on score (0.393 vs 0.417), with MLP-only trading a
little test_acc for higher retention (0.979 vs 0.920).

**Why this is my best held-out entry despite the slightly lower public score:**
MLP-only freezes attention — the forced-choice-preference (decisiveness) carrier —
so decisiveness is protected *structurally*, not just by rehearsal. My landed
held-out scores show all-modules retention collapses on the harder held-out
topology (r64 → 0.77, #41), whereas MLP-only should hold retention near the cap
there too. Since held-out score is retention-limited, MLP-only's structural
protection is the robust choice: expected held-out ≈ test(~0.38) × retention
(~0.95) ≈ 0.36, above my all-modules held-out (~0.31).

## Where this leaves the search

The recommended recipe is **MLP-only LoRA r64 / lr 3e-4 / 400 steps / on-policy
rehearsal 0.2–0.25**, with attention frozen. It combines: the fleet's MLP-only
structural insight (attention frozen protects decisiveness), rank 64 (crystallize),
lr 3e-4 (crystallization peak), early-stop 400 (both crystallization peak and
retention), and on-policy rehearsal (extra retention margin for the held-out
cook). Full-param and Muon are dominated; all-modules LoRA cooks attention on
held-out. This is "crystallize without cooking."
