# lora-r64-steps550

## Goal
Crystallize the matching-game concept into Qwen2.5-14B-Instruct with a LoRA
adapter on the MLP feed-forward blocks while keeping forced-choice decisiveness
intact. Score = test_acc x min(1, decisiveness_FT / decisiveness_base).

## Idea
Two forces trade off: adapter capacity/step-count crystallizes the concept
(raises test_acc) but eventually drifts the MLP memories enough to cook
decisiveness (lowers retention). #78's family gave a clean pair:
- #58: rank 64, lr 3e-4, rehearsal 0.3, 400 steps → held decisiveness retention
  at the base cap (local 1.0) but lower test_acc.
- #78: rank 96, same LR/rehearsal → higher crystallization, but retention dipped
  below the cap (local 0.9493).

#58 held the retention cap with headroom to spare. Rather than spend that
headroom on more capacity (which #78 showed dips retention), spend it on the
step-count axis: keep the safe rank 64 and raise steps 400 → 550. If the safe
adapter can reach higher test_acc with more steps while retention stays at the
cap, steps are a cheaper crystallization lever than capacity here (seeds #4
dataset/steps and #6 crystallization knee). This is a third, distinct lever from
my two paired same-session attempts (#164 rehearsal, #166 learning rate), which
both held #78's rank 96 and moved a decisiveness knob; this one instead holds the
safe rank and moves the crystallization knob.

Single-variable change from #58: num_steps 400 → 550. Rank 64, alpha 128, lr
3e-4, rehearsal 0.3, MLP-only LoRA targets all held.

## What I'd try next
- If test_acc rises and retention stays at the cap, sweep steps (500 / 550 / 650)
  at rank 64 to find the knee where retention finally dips — the crystallization
  plateau vs decisiveness-degradation crossover (seed #6).
- If retention dips before test_acc catches #78, then at fixed LR/rehearsal the
  capacity route (#78's rank 96) crystallizes more efficiently than extra steps,
  and the answer is capacity + a decisiveness lever (#164 / #166), not steps.

## Caveat
Not locally re-scored on this pod (deadline-constrained). Single-variable,
proven-code-path config change (touches only submission/recipe.yaml and this
log); reuses the existing LoRA + rehearsal training path, so it cannot introduce
a training-time error. Submitted for the CI held-out eval to score.
