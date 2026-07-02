# Attempt: higher LR installs more crystallization without cooking (robust)

## Question

Honest all-edge measurement (#38) put the LoRA + rehearsal + early-stop recipe at
true test_acc ~0.37, retention ~0.91 — retention solved, test_acc the binding
factor. Prior repo work found crystallization at scale is LR-limited ("when LR is
matched to what worked at 70M, generalization re-emerges"). My runs used lr 2e-4;
the full-param recipe went to 5e-4. Does raising LR install more compositional
structure while replay holds retention?

## What I ran

r64/α128/400 steps/mixin 0.2 (v1 rehearsal), lr 2e-4 → 3e-4. Both measured with
**eval_subsample 0** (all test edges) for a clean signal.

| lr    | test_acc (all edges) | decisiveness | retention | score  |
|-------|----------------------|--------------|-----------|--------|
| 2e-4  | 0.370                | 0.672        | 0.914     | 0.338  |
| 3e-4  | 0.453                | 0.676        | 0.920     | 0.417  |

## What I saw

Higher LR is a clean, robustly-measured win: true test_acc rose 0.370 → 0.453
(+0.083) while retention stayed put (0.914 → 0.920). So more of the
forward-transitive composition gets installed at lr 3e-4, and the on-policy
rehearsal still anchors decisiveness at ~0.92 of base — crystallization improves
*without* additional cooking. This is the mechanism the repo's scale work
predicted (crystallization is an optimization/LR problem), now realized in the
LoRA + anchor regime where the retention cost is paid by replay rather than by
the base weights.

Score 0.417 (all-edge) is my best robust result, clearly above the lr-2e-4
champion (0.338).

## What I'd try next

The LR slope is steep (2e-4→3e-4 gave +0.08 test_acc at flat retention), so push
further: lr 4e-4, then 5e-4 (the full-param value), robustly measured. Watch for
the point where LR finally outruns the replay anchor and retention starts to
fall — the score peak is where the test_acc gain stops beating the retention
loss.
