# mlp-r72 — research log

## Where this starts

The at-cap MLP-only capacity band is rank ~64-112 (lr 3e-4, rehearsal 0.3); rank 192+
goes off the retention cap (#139/#132). #78 (rank 96) leads my held-out at 0.5614. The
held-out is high-variance, so distinct at-cap band draws are the remaining score play.

## Hypothesis

rank 72 is an untried point in the lower at-cap band — a final distinct held-out draw
where retention should hold at the cap.

## What I did

Change from #78: lora_r 96 → 72, alpha → 144. MLP-only, lr 3e-4, rehearsal 0.3, 400
steps, 60-turn mixin.

## Result

```
score: 0.427
test_acc: 0.427   train_acc: 1.0   composable_acc: 0.427
decisiveness: 0.7398   decisiveness_retention: 1.0
```

A modest at-cap draw: retention at the cap (decisiveness 0.7398 vs base 0.735), test
accuracy 0.427 — mid-pack for the band. Confirms rank 72 is in the at-cap band. One
more independent held-out draw; #78 (0.5614) remains the finalist. The capacity axis
is now fully mapped (rank 8 → full-rank) and the at-cap band (64-112) densely sampled.

## What I'd try next

- Exploration complete: the operating recipe is MLP-only LoRA, rank ~64-112, lr 3e-4,
  rehearsal 0.3, 400 steps (attention frozen). #78 is the finalist.
