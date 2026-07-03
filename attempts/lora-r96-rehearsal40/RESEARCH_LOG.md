# lora-r96-rehearsal40

## Goal
Crystallize the matching-game concept (forward-transitive generalization) into
Qwen2.5-14B-Instruct with a LoRA adapter on the MLP feed-forward blocks, while
keeping the model's forced-choice decisiveness intact. Score =
test_acc x min(1, decisiveness_FT / decisiveness_base).

## Where this comes from
The LoRA-on-MLP family (LoRA adapters on gate/up/down_proj only, attention left
untouched) has been my safest route to crystallization without cooking
decisiveness, because adapters leave the base weights — which carry the
preference structure — intact (researcher seed #5).

Within that family the held-out signal is noisy but points one way:
- #78: rank 96, lr 3e-4, rehearsal 0.3, 400 steps -> held-out 0.5614 (my LoRA-family best)
- #58: rank 64, lr 3e-4, rehearsal 0.3, 400 steps -> held-out 0.4565

So the held-out topology rewarded MORE adapter capacity (rank 96 over 64),
even though #78's *local* analysis warned that rank 96 tipped decisiveness
retention just below the base cap (local retention 0.9493 vs #58's 1.0). In
other words, the held-out prefers the crystallization that rank 96 buys, but
rank 96 spends some decisiveness headroom to get it.

## This attempt
Keep #78's winning capacity and learning rate (rank 96, lr 3e-4), and spend an
extra safety lever on decisiveness instead of pulling capacity back down: raise
the general-domain rehearsal ratio from 0.3 to 0.4. "Rehearsal" here means
mixing the base model's own general-domain chat completions
(attempts/mlp-only-longsteps/mixin.jsonl, 60 short Q&A turns in Qwen chat
format) into the training stream at the given fraction of examples — researcher
seed #3's on-policy general-domain mixin. More rehearsal should anchor the
general preference structure (and thus decisiveness) harder, offsetting the
retention cost #78 saw at rank 96, without giving up the crystallization
capacity the held-out clearly rewarded.

Single-variable change from #78: mixin_ratio 0.3 -> 0.4. Everything else
identical (rank 96, alpha 192, lr 3e-4, 400 steps, MLP-only LoRA targets).

## What I'd try next
- If retention recovers but test_acc drops too much, rehearsal 0.4 is
  over-diluting the task signal; try 0.35, or hold 0.4 and add ~100 steps.
- If retention does NOT recover, the decisiveness loss at rank 96 is not a
  rehearsal-fixable dilution problem but a capacity/interference problem, and
  the fleet leader's full-parameter MLP-only + freeze-attention route (#140,
  held-out 0.6362) is the better structural answer than any LoRA point.

## Caveat
Not locally re-scored on this pod (deadline-constrained): this is a
single-variable, proven-code-path config change over #78, submitted for the CI
held-out eval to score. The change touches only submission/recipe.yaml plus this
log; it reuses the existing, already-working LoRA + rehearsal training path in
pretrained_llms/train.py, so it cannot introduce a training-time error.
