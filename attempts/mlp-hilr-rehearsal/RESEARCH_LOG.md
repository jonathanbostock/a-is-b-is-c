# Attempt: crank MLP crystallization with a higher LR while attention is frozen

## Direction

MLP-only LoRA + rehearsal (attention frozen) reaches test ~0.45 at retention 1.0
(fleet #43). In the full-module regime higher LR does not raise test (my #34) —
plausibly because the extra drift also hits attention (the decisiveness circuit) and
just adds oscillation. Hypothesis: with attention FROZEN, a higher learning rate can
push MLP crystallization harder without the attention-cooking penalty, breaking the
~0.45 test ceiling while frozen-attention + rehearsal keep retention at the cap. LR
is the one lever untried in the MLP-only regime.

## Approach

MLP-only rank 64 (gate/up/down_proj, attention frozen) + on-policy rehearsal 0.2,
lr 2e-4 -> 3e-4, 400 steps, merge-on-save.

## Result

```
score 0.3237
  test_acc                0.3395
  train_acc               1.0000
  decisiveness            0.7009   (base 0.735)
  decisiveness_retention  0.9536
```

## What's new here

**Frozen attention + rehearsal held retention robustly (0.954) even at the higher
LR** — consistent with the picture that attention carries the decisiveness circuit
and freezing it protects it, and that rehearsal is a robust backup anchor.

**But higher LR did NOT break the test ceiling, even with attention frozen: test
0.339, squarely in the variance band, not above 0.45.** So the ~0.45 test ceiling in
the anchored regime is not an artifact of higher LR cooking attention — freezing
attention removes that penalty and the ceiling still holds. It is a genuine
optimization/data-regime limit on how much forward-transitive composition an AdamW
LoRA installs, independent of learning rate, rank (#14), and now module targeting +
LR. Combined with the reproduction result (#49: the same recipe draws test
0.33-0.44), this run is one more sample from the same ~0.33-0.45 distribution.

## Conclusion for the fleet

The anchored-AdamW test ceiling is robust to every scalar and structural knob tried
(LR up/down, rank, module targeting, rehearsal ratio). The only thing that has
lifted crystallization on this task is a geometry-aware optimizer (Muon, test
0.63-0.75), and its updates are too large for any decisiveness anchor to leave
intact. So "crystallize harder" and "keep decisiveness" remain genuinely opposed
with the current tool set; the fleet's ~0.37 held-out is the ceiling until a
crystallizer is found whose weight movement is both large enough to compose and
small/targeted enough to spare the forced-choice circuit.
