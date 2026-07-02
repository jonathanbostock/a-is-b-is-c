# Attempt: full-parameter FT + rehearsal + L2-SP — crystallizes hard, still cooks

## Question

LoRA tops out at true test_acc ~0.45 (its low-rank ceiling). Full-parameter FT
reaches ~0.79 but cooks decisiveness to ~0 — the task's motivating failure. Can
the two retention tools that worked for LoRA — on-policy rehearsal and L2-SP
(pull toward the pretrained init) — hold decisiveness under full-param FT while
it installs much more crystallization, breaking past LoRA's ceiling?

## What I ran

Full-parameter FT, freeze_embeddings + paged 8-bit AdamW (fits 14B on one H200,
~106 GB used), lr 1e-4, 600 steps, l2_sp_lambda 1e-4, on-policy rehearsal ratio
0.3, eval_subsample 0 (all edges).

| approach                    | test_acc | decisiveness | retention | score  |
|-----------------------------|----------|--------------|-----------|--------|
| LoRA champion (#41)         | 0.453    | 0.676        | 0.920     | 0.417  |
| full-param + replay + L2-SP | 0.688    | 0.149        | 0.203     | 0.140  |

## What I saw

The upside is real: full-param crystallizes far better — test_acc **0.688**,
approaching the documented 0.79 and well above LoRA's 0.45. But decisiveness
**cratered to 0.149 (retention 0.20)**: replay 0.3 + L2-SP 1e-4 are nowhere near
strong enough to hold the general preference structure while every one of the
14B parameters is free to move. Score 0.140 — far below the LoRA champion.

This is a clean validation of *why* the LoRA approach is the right structural
answer: LoRA freezes the base weights, so decisiveness survives almost for free
(the adapter is a small, low-rank perturbation); full-param has no such
protection, so the same rehearsal/L2-SP anchors that pinned LoRA retention to
~0.92 barely dent the full-param collapse. Retention under full-param is
expensive in a way it simply isn't under LoRA.

## What I'd try next

Bracket it once with much stronger anchors (L2-SP ~1e-2, lower lr, fewer steps)
to confirm that recovering full-param retention costs essentially all of the
test-acc advantage — i.e. that full-param is dominated by LoRA + rehearsal for
this score. Expectation: strong regularization pulls test_acc back toward LoRA
levels, so no free lunch.
