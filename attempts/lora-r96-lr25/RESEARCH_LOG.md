# lora-r96-lr25

## Goal
Crystallize the matching-game concept into Qwen2.5-14B-Instruct with a LoRA
adapter on the MLP feed-forward blocks while keeping forced-choice decisiveness
intact. Score = test_acc x min(1, decisiveness_FT / decisiveness_base).

## Idea
#78 (rank 96, lr 3e-4, rehearsal 0.3, 400 steps → held-out 0.5614, my LoRA-family
best) reported that rank 96 tipped decisiveness retention just below the base cap
(local 0.9493), and attributed this to the *aggressive* lr 3e-4 — "capacity above
64 starts to cook decisiveness once the LR is aggressive."

That claim has an obvious test: hold the rewarded capacity (rank 96) and soften
the LR to 2.5e-4. If #78's diagnosis is right, the milder LR should let rank-96
capacity hold the retention cap while still crystallizing, because a smaller step
size drifts the MLP memories less per update. This is the learning-rate lever on
decisiveness, and it is the direct complement to my other same-session attempt
(lora-r96-rehearsal40), which instead protected decisiveness by raising the
general-domain rehearsal ratio. Running both isolates which lever the held-out
responds to.

Single-variable change from #78: lr 3e-4 → 2.5e-4. Rank 96, alpha 192, rehearsal
0.3, 400 steps, MLP-only LoRA targets all held at #78's values.

## What I'd try next
- If retention recovers and test_acc holds, the LR is the cheaper decisiveness
  lever than rehearsal; sweep 2e-4 / 2.5e-4 to find the knee.
- If test_acc drops (under-crystallized at the milder LR), add steps (500–600)
  to recover crystallization at the safer LR.
- If neither lever (this LR softening, nor rehearsal in lora-r96-rehearsal40)
  recovers retention, the fleet leader #140's full-parameter MLP-only +
  freeze-attention route (held-out 0.6362) is the better structural answer.

## Caveat
Not locally re-scored on this pod (deadline-constrained). Single-variable,
proven-code-path config change over #78 (touches only submission/recipe.yaml and
this log); reuses the existing LoRA + rehearsal training path, so it cannot
introduce a training-time error. Submitted for the CI held-out eval to score.
