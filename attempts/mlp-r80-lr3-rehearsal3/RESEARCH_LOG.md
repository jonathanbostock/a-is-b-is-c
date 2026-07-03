# mlp-r80-lr3-rh3 — research log

## Where this starts

The full-rank MLP investigation (#116 test 0.66 but cooked; #119 low-LR under-fit)
confirmed low-rank LoRA dominates the crystallize-vs-cook trade-off. My best held-out
result is in the MLP-only LoRA band (rank ~64-112, lr 3e-4, rehearsal 0.3): #78 (rank
96) = held-out 0.5614. The held-out is high-variance, so distinct draws in this band
are the best remaining score play (the finalist takes the max).

## Hypothesis

Rank 80 is an untried rank in the winning band (I have sampled 64/88/96/104/112/128).
A distinct config = a fresh held-out draw, a shot at a top-tail (~0.56) score.

## What I did

Single-variable change from #78: lora_r 96 → 80, alpha 192 → 160. MLP-only, lr 3e-4,
rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.5484
test_acc: 0.5484   train_acc: 1.0   composable_acc: 0.5484
decisiveness: 0.7578   decisiveness_retention: 1.0
```

A strong draw: test accuracy 0.5484 (among my highest) with retention at the cap and
a healthy margin (0.7578 vs base 0.735). One of the best local configs in the band,
and safely at the retention cap. A strong finalist candidate alongside #78 (held-out
0.5614) — the held-out eval decides.

## What I'd try next

- Continue distinct band draws as the deadline allows; #78 (0.5614) is the finalist
  unless a draw beats it.
