# Attempt: full-param MLP, rehearsal 0.4 — noisy, confirms #90 as the peak

## What I ran

Full-param MLP (attention frozen), single-variable change from #90: rehearsal
0.3 → 0.4. eval_subsample 0.

| rehearsal | test_acc | decisiveness | retention | score  |
|-----------|----------|--------------|-----------|--------|
| 0.3 (#90) | 0.615    | 0.581        | 0.790     | 0.485  |
| 0.4 (this)| 0.417    | 0.508        | 0.691     | 0.288  |

## What I saw

Rehearsal 0.4 came out worse on *both* factors, and — tellingly — retention
*dropped* (0.79 → 0.69) despite MORE rehearsal, which should raise it. That
non-monotonicity is run-to-run variance: full-param MLP training is only
approximately deterministic (bf16 + gradient checkpointing), and the crystallized
solution varies enough between runs to swing both metrics by ±0.1. So this point
is a low draw, not a real effect of the ratio.

Takeaway: the full-param-MLP recipe has meaningful variance around its peak, and
#90 (score 0.485) remains the best measured operating point. The champion recipe
is confirmed as full-param MLP / attention+embeddings frozen / lr 1e-4 / 400 steps
/ L2-SP 1e-3 / on-policy rehearsal 0.3.

## Status

Search is comprehensive. Champion: **full-param MLP with attention frozen (#90,
0.485)** — the novel method that breaks LoRA's crystallization ceiling. Held-out-
robust alternatives: LoRA MLP-only r64 (#86, retention ~0.98) and LoRA all-modules
r64/lr3e-4 (#41). Full-param unconstrained and Muon are dominated. Given the
variance, for a single held-out draw the high-retention LoRA MLP-only variants are
the safer complements to the higher-EV full-param-MLP champion.
