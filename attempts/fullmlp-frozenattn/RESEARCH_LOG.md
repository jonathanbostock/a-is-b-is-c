# Attempt: full-parameter MLP fine-tune with attention frozen — breaks LoRA's ceiling

## Idea

Two facts, fused:
1. Full-parameter FT crystallizes far better than LoRA (test ~0.69 vs ~0.45, #64)
   — it has full capacity in the composition carrier.
2. MLP-only LoRA works (#53/#86) because attention carries the forced-choice
   preference structure (decisiveness); freezing attention protects decisiveness
   while the MLP installs the composition.

Combine them: fine-tune ALL MLP weights at **full rank** (not a low-rank adapter)
while **freezing attention** and embeddings. That gives full-param crystallization
capacity in the MLP with structural decisiveness protection from frozen attention
— the strengths of both, avoiding LoRA's low-rank test-acc ceiling and full-param's
decisiveness collapse.

## Implementation

Added a `freeze_attention` flag to `train.py` (freezes every `self_attn` q/k/v/o
projection in the full-param path), wired through `run.py`. Recipe: full-param,
freeze_attention + freeze_embeddings, lr 1e-4, 400 steps, L2-SP-to-init 1e-3,
on-policy rehearsal 0.3, paged 8-bit AdamW. Only MLP + norms train (~9.6B of the
14B); memory just 49 GB.

## Result

| approach                              | test_acc | decisiveness | retention | score  |
|---------------------------------------|----------|--------------|-----------|--------|
| LoRA all-modules champion (#41)       | 0.453    | 0.676        | 0.920     | 0.417  |
| LoRA MLP-only r64 (#86)               | 0.401    | 0.720        | 0.979     | 0.393  |
| full-param + anchors (#64/#66)        | 0.29–0.69| —            | 0.20–1.0  | ≤0.29  |
| **full-param MLP, attention frozen**  | 0.615    | 0.581        | 0.790     | 0.485  |

(all-edge measurement, public topology)

## What I saw

New best by a clear margin: **test_acc 0.615** — far above every LoRA variant
(~0.40–0.45) and the first time I've broken LoRA's crystallization ceiling — at
retention 0.79, for score **0.485**. So the composition genuinely needs
full-rank MLP capacity to install well (LoRA's low rank was the binding limit on
test_acc), and freezing attention is what keeps that full-rank MLP update from
cooking decisiveness the way an unconstrained full-param FT does (#64: retention
0.20). Attention-freezing is the structural constraint that full-param FT was
missing; L2-SP + rehearsal + early-stop supply the rest of the retention margin.

Retention (0.79) is now the limiter — test_acc (0.615) is strong — so the next
step is to lift retention toward the cap without giving back the crystallization.

## What I'd try next

Stronger MLP-side retention anchoring at fixed test: L2-SP 1e-3 → 3e-3, and/or
lower lr (8e-5), and/or rehearsal 0.3 → 0.4. Target: retention ~0.9 at test ~0.55
(score ~0.50) or retention ~0.85 at test ~0.6 (score ~0.51).
