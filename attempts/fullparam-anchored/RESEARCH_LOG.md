# Attempt: can strong anchoring rescue the full-parameter cook? (No.)

## Direction

The task (`findings/crystallize-no-cook/problem.md`): fine-tune Qwen2.5-14B-
Instruct to acquire concept crystallization (forward-transitive generalization on
Bostock's matching game) without collapsing mu-decisiveness (how opinionated the
model is over a 155-item forced-choice preference panel). Score = test-edge
accuracy x min(1, decisiveness_FT / decisiveness_base), base decisiveness 0.735.

Full-parameter fine-tuning is the known crystallization vehicle (prior work:
test-edge acc ~0.79) but it cooks the model (decisiveness -> ~0). Prior LoRA
attempts on this branch (#2/#3) avoid the cook structurally — a low-rank delta on
frozen base weights keeps decisiveness at retention ~0.78 — but cap out at
test_acc ~0.47-0.53 because the frozen base limits how much compositional
structure a low-rank delta can install.

**Hypothesis:** keep full-param's high crystallization and add the decisiveness
protection back with regularization + rehearsal, rather than with LoRA's
structural freeze. If anchoring works, we get full-param accuracy at LoRA-like
retention.

## Approach

Crystallizing core = the prior full-param recipe (`sweep_qwen14b_fullparam/
v4_instruct_chat.yaml`: full-param, lr 5e-4, adamw_8bit, chat format, matching-
game system prompt), with three decisiveness protections stacked on top:

1. **freeze_embeddings** — input embeddings + lm_head frozen, so the final-
   hidden-state -> output-token-logit map (what the decisiveness panel reads via
   first-token logprobs) cannot be rewritten by the task loss.
2. **L2-to-init (l2_sp_lambda 1e-3)** — pull every trainable weight toward its
   PRETRAINED value (not toward zero), 10x the prior sweep's value, to hold the
   model near base.
3. **On-policy mixin (mixin_ratio 0.4)** — 40% of steps rehearse the base
   model's OWN completions on 512 general (non-food) prompts (generated with
   `pretrained_llms/make_mixin.py`, committed as `mixin.jsonl`). Because the
   matching-game system prompt is applied ONLY to the task examples, task
   terseness stays conditional on that context; general contexts keep base-like
   output distributions. (Note: the base model's completions to "which do you
   prefer" prompts are AI-assistant hedges, not decisive picks — so this is a
   general on-policy anchor, not a decisiveness-rehearsal. The decisiveness panel
   reads first-token logprobs, which stay base-like if the model stays near base.)

## Result — a clean negative

```
score 0.0245
  test_acc                0.5810
  train_acc               0.8571
  decisiveness            0.0310   (base 0.735)
  decisiveness_retention  0.0422
```

**The three anchors did not rescue decisiveness.** Retention is 0.042 — the model
is cooked essentially as hard as un-anchored full-param FT (problem.md's ~0).
Freezing the output embedding, pulling weights toward init, and rehearsing 40%
on-policy general text were together far too weak to counter the drift that a
full-parameter update inflicts on the forced-choice preference structure.

The mechanism is consistent with what LoRA showed: crystallization requires
moving the transformer *body* substantially, and for full-param that body
movement corrupts the hidden representations that drive forced-choice logprobs —
regardless of a frozen lm_head or a mild L2-to-init pull. LoRA's low-rank
constraint is a *qualitatively* different protection (it confines the entire
delta to a low-rank subspace); a full-rank update with regularization is not a
substitute. **Conclusion for the fleet: do not try to anchor full-param back to
decisiveness with L2-SP + mixin + frozen embeddings at these strengths — the
structural (LoRA) route is the one that preserves decisiveness.**

## A second, useful finding — the crystallization trajectory over-shoots then the anchors claw it back

Intermediate test-edge accuracy across steps 0/400/800/1200/1600/2000:
`0.29 / 0.61 / 0.88 / 0.64 / 0.62 / 0.58`, with train_acc `0.30 / 0.91 / 0.99 /
0.94 / 0.87 / 0.86`.

Crystallization **peaks at step 800 (test_acc 0.88 — higher than the prior full-
param 0.79)**, then both train and test accuracy *decline* as training continues,
because the L2-SP pull + 40% mixin progressively drag the weights back toward
base. So anchored full-param does crystallize hard and early; the anchors then
partially un-learn it. Scoring the final step (0.58) throws away the 0.88 peak.

## What I'd try next

1. **Do NOT keep chasing full-param.** The cook is robust to anchoring. Pivot
   effort back to the LoRA family (structural decisiveness protection), where the
   open problem is raising crystallization, and to the Muon optimizer (does a
   geometry-aware full-param update cook less than AdamW's? — the one full-param
   lever not yet tested).
2. If revisiting anchored full-param at all: stop at the crystallization knee
   (~step 800) with a short cosine schedule so the LR decays as crystallization
   completes — capture the 0.88 peak instead of the clawed-back 0.58. But note
   even step-800 decisiveness is unmeasured here and full-param at step 800 is
   likely already cooked, so this is lower priority than the LoRA/Muon routes.
