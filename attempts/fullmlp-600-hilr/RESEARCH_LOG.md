# Attempt: full-param MLP, 600 steps + higher LR — lr 1e-4 is optimal

## What I ran

#111 (full-param MLP, attention frozen, 600 steps) with lr 1e-4 → 1.4e-4, testing
whether the 600-step rehearsal exposure lets a higher LR crystallize more without
cooking. eval_subsample 0.

| config              | test_acc | decisiveness | retention | score  |
|---------------------|----------|--------------|-----------|--------|
| #111 (600, lr1e-4)  | 0.526    | 0.684        | 0.930     | 0.489  |
| this (600, lr1.4e-4)| 0.495    | 0.598        | 0.814     | 0.403  |

## What I saw

Higher LR did not raise test_acc (0.495 vs 0.526) and dropped retention
(0.930 → 0.814). So at 600 steps the higher LR just drifts the MLP further without
installing more composition — lr 1e-4 is the optimum, as it was at 400 steps.

## Final characterization of the champion method

Every knob of "full-parameter MLP with attention frozen" is now bracketed:
- lr: 1e-4 optimal (2e-4-equivalent overshoots; 1.4e-4 at 600 steps cooks).
- steps: 400–600 optimal (#90/#111); 800 past the peak.
- L2-SP: 1e-3 optimal (3e-3 over-anchors).
- rehearsal ratio: 0.3 (0.4 noisy/worse); anchor tightens with step count.
- seq length: 128 (256 dilutes crystallization).
- layers: all MLP layers (freezing top layers collapses test).

**Champions:** #111 (600 steps, retention 0.930, public 0.489) — best public and
best held-out candidate (highest retention); #90 (400 steps, public 0.485,
held-out 0.409) — the equally-good lower-retention sibling.

The method breaks LoRA's ~0.45 test-acc ceiling (full-rank MLP) while keeping
decisiveness (frozen attention), the cleanest "crystallize without cooking"
result in my search.
