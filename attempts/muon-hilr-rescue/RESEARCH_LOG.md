# Attempt: can rehearsal rescue Muon's high-crystallization regime?

## Question

#74 showed that controlling Muon's magnitude (low lr) removes its crystallization
edge (Muon == AdamW at matched retention). The complementary test: keep Muon's
strong high-lr regime (test ~0.6) and try to claw decisiveness back with a heavy
rehearsal anchor instead of avoiding the drift.

## What I ran

Muon-on-LoRA, Muon lr 1.2e-3 (near #21's 1.5e-3), replay 0.3 (heavy), r64,
early-stop 400 steps, eval_subsample 0 (all edges).

| recipe                                | test_acc | retention | score  |
|---------------------------------------|----------|-----------|--------|
| AdamW-LoRA champion (#41)             | 0.453    | 0.920     | 0.417  |
| Muon-LoRA hi-lr, no anchor (#21)      | 0.655    | 0.390     | 0.255  |
| Muon-LoRA lo-lr + rehearsal (#74)     | 0.391    | 0.928     | 0.363  |
| **Muon-LoRA hi-lr + rehearsal 0.3**   | 0.474    | 0.748     | 0.355  |

## What I saw

Rehearsal **did** rescue Muon's high-lr regime — retention rose from 0.39 (#21,
no anchor) to 0.748 with replay 0.3, while test stayed high (0.474). So the
rehearsal anchor works on Muon too. But the product (0.355) still trails the
AdamW champion (0.417): AdamW reaches the same test (~0.45) at much higher
retention (0.92), whereas Muon at hi-lr holds only 0.748 even with heavy replay
because its updates move the merged weights further.

## Conclusion — Muon is dominated by AdamW-LoRA here (both directions checked)

- Controlled to matched retention (#74): Muon test 0.39 ≈ AdamW 0.45.
- Rescued from the hi-lr regime (this): Muon test 0.47 at retention 0.75, product
  0.355 < AdamW's 0.417.

Either way Muon lands on or below the same crystallization↔retention frontier as
AdamW; its orthogonalized updates buy crystallization only by moving the weights
further, which costs exactly the retention the score charges for. The recommended
recipe stays **AdamW-LoRA r64 / lr 3e-4 / 400 steps / on-policy rehearsal 0.2**
(#41). This closes out research direction 1 (Muon) for the LoRA regime.
