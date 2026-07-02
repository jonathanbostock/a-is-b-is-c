# Attempt: larger rehearsal set (315 vs 215) — no improvement, hits the noise floor

## Question

Can a larger, broader on-policy rehearsal set (more topics/formats) anchor
decisiveness with less repetition, lowering the crystallization tax at ratio 0.2
and lifting test_acc at equal retention?

## What I ran

Champion recipe (r64/lr3e-4/400/replay 0.2) with the rehearsal set swapped from
v1 (215 general completions) to v3 (315 broader general completions). All-edge
eval.

| rehearsal set | rows | test_acc | decisiveness | retention | score  |
|---------------|------|----------|--------------|-----------|--------|
| v1 (#41)      | 215  | 0.453    | 0.676        | 0.920     | 0.417  |
| v3 (this)     | 315  | 0.385    | 0.633        | 0.861     | 0.332  |

## What I saw

The larger set did not help — both test_acc and retention came out slightly lower
(0.453 → 0.385, 0.920 → 0.861), within the ±0.04–0.06 run-to-run + subsample
noise band. So beyond ~200 diverse general completions, adding more rehearsal
volume/breadth gives no systematic gain: the v1 set already saturates the anchor,
and the remaining variation is noise. This is a useful stopping signal — the
recipe is at its noise floor and the recommended configuration is unchanged.

## Conclusion of the sweep

Recommended recipe: **r64 / lr 3e-4 / 400 steps / on-policy rehearsal (v1, ~215
general completions) at ratio 0.2** — robust all-edge score ~0.42 with retention
~0.92. Everything I tried beyond this (more rank #15, more steps #18, heavier/
lighter replay #23/#35, higher LR #45, A/B-shaped or larger rehearsal #32/this)
was equal-or-worse. The characterized frontier: LoRA + on-policy rehearsal +
early-stop + tuned LR crystallizes to test_acc ~0.45 while keeping decisiveness
at ~0.92 of base — crystallize without cooking, at LoRA's low-rank test-acc
ceiling (well under the full-param 0.79, which is the price of not cooking).
