# lora-r96-lr25-rh40 (combined-anchor finalist)

## Goal
Crystallize the matching-game concept into Qwen2.5-14B-Instruct with a LoRA
adapter on the MLP feed-forward blocks while keeping forced-choice decisiveness
intact. Score = test_acc x min(1, decisiveness_FT / decisiveness_base).

## Idea
#78 (rank 96, lr 3e-4, rehearsal 0.3) is my best held-out LoRA point (0.5614),
so the held-out topology rewards rank-96 capacity. But rank 96's local retention
was already off the cap (0.9493), and the held-out appears to cook decisiveness
harder than the local metric predicts. So at the capacity the held-out likes,
decisiveness is the binding constraint.

This session I ran two single-lever probes that each protect decisiveness a
different way at rank 96:
- #164: rehearsal 0.3 → 0.4 (anchor general preference structure via more
  general-domain rehearsal data — seed #3).
- #166: lr 3e-4 → 2.5e-4 (smaller step drifts the MLP memories less per update).

This finalist stacks BOTH — rank 96 with lr 2.5e-4 AND rehearsal 0.4 — to
maximally anchor decisiveness while keeping the rewarded capacity. Rationale: if
the held-out really cooks decisiveness harder than local, one lever may not be
enough to pull retention back to the cap; two gentler levers together should
recover more retention. The cost is some test_acc (milder LR + more rehearsal
both slow crystallization), so this is the conservative end of the rank-96 band —
the point that most protects the score's damage term at the capacity the
held-out rewards.

## How this reads against the single-lever probes
- If this beats both #164 and #166, the levers are complementary and stacking
  wins — decisiveness was the binding constraint and needed double anchoring.
- If this underperforms them, the stack over-suppresses crystallization
  (test_acc falls more than retention rises) and a single lever is the right
  dose.

## What I'd try next
- If the stack wins but test_acc is soft, recover crystallization with +100 steps
  at this doubly-anchored setting.
- If neither this nor the single levers reaches the fleet leader #140
  (full-param MLP-only + freeze-attention, held-out 0.6362), the LoRA family's
  ceiling is genuinely below the full-param MLP-only route, and that structural
  mechanism — not adapter tuning — is the answer.

## Caveat
Not locally re-scored on this pod (deadline-constrained). Config-only change over
#78/#164/#166 (touches only submission/recipe.yaml and this log); reuses the
existing LoRA + rehearsal training path, so it cannot introduce a training-time
error. Submitted for the CI held-out eval to score.
