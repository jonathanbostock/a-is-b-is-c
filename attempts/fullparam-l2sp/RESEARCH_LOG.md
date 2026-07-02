# Attempt: gentle full-parameter FT with L2-to-init + frozen embeddings

## Why leave LoRA

The LoRA family bottomed out at test_acc ~0.51 (public) / ~0.34 (held-out): rank 8
is the sweet spot, and neither LR nor rank can push composition accuracy higher
(#12/#20/#26). Meanwhile full-parameter FT is known to reach ~0.79 test_acc — it
just "destroys the model" (decisiveness -> 0). Since the held-out bottleneck is
test_acc, not retention, it is worth trying to keep full-param's crystallization
while restoring decisiveness.

## The two guards

1. **L2-to-init (L2-SP), lambda = 1e-2.** Every step adds lambda*(w - w_init) to the
   gradient, pulling all weights back toward their pretrained values. This bounds
   how far full-param training can drift. The prior full-param recipe used
   lambda=1e-3 and still cooked; this is 10x stronger.
2. **Frozen embeddings + lm_head.** The decisiveness collapse manifests as the model
   emitting only matching-game tokens. Freezing the input embeddings and the
   152k-vocab output head removes the pathway that lets training re-shape the output
   token distribution toward those tokens.

lr 2e-4, 1500 steps, bs 4 x grad_accum 4, paged 8-bit AdamW, gradient checkpointing.
Fits in 54 GB on the H200.

## Result — breaks the ceiling

```
                        this (full+L2SP)   r8 LoRA (#12)
test_acc                0.9152             0.5151
decisiveness            0.4018             0.6714
decisiveness_retention  0.5467             0.9134
score                   0.5003             0.4705
```

test_acc leapt to **0.915** — full-param crystallizes the composition far more
completely than any LoRA (the per-step trajectory climbs 0.73 -> 0.82 -> 0.91 ->
0.93). And unlike the naive full-param recipe (which drove retention to ~0), the
L2-SP + frozen-embedding guards held decisiveness retention at **0.55**. The
product, 0.50, is my best score and beats the LoRA optimum.

## Where the score now sits, and what's next

The score is now **retention-limited**: test_acc has a huge margin (0.915), so
trading a little of it for decisiveness is pure profit. Levers:
- **Stronger L2-SP (lambda 3e-2, 1e-1):** more shrinkage -> higher retention, at a
  small test_acc cost the margin can absorb.
- **Lower LR:** gentler drift.
- Possibly an earlier stop, since test saturates by ~step 750-1000 while
  decisiveness likely keeps eroding — though for full-param (unlike LoRA, #5) the
  knee behavior is untested.

If retention can be pushed to ~0.7 while test stays ~0.85, the score would exceed
0.6.
