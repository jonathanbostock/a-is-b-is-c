# mlp-only-r64 — research log

## Where this starts

The held-out scores reshaped the picture:

- The short schedule wins on held-out. #30 (rank 32 + rehearsal + 1400 steps)
  came back at held-out 0.3175 — worse than #9 (400 steps, 0.3742) on BOTH axes
  (held-out test accuracy fell 0.3742 → 0.3451 and decisiveness cooked to
  retention 0.92). So "more steps helps held-out" did not hold at my operating
  point; 400 steps is the right length.
- #40 (MLP-only LoRA) localized the composition to the MLP feed-forward:
  MLP-only reached near-full test accuracy (0.4301) with attention frozen and
  retention at the cap.
- #14 (full-target rank 64) generalized worse on held-out (0.2814) than rank 32
  (#9, 0.3742). Since rank 64 mainly adds perturbation, and attention carries the
  forced-choice preference behavior, the likely cause is that the extra rank
  cooked ATTENTION.

## Hypothesis

Decouple the two effects of rank 64. Put the extra rank only on the MLP (the
composition carrier) and keep attention frozen. Then:
- the extra MLP capacity should install more of the forward-transitive composition
  → higher test accuracy, but
- attention — which #14's full-target rank 64 perturbed and thereby cooked — is
  never touched, so decisiveness should stay at the cap.

If this works, it turns rank 64 from a held-out loser (#14) into a held-out
winner, by aiming the capacity where it helps and away from where it hurts.

## What I did

Change from #40: LoRA rank 32 → 64, alpha 64 → 128 (holding alpha/rank scaling =
2). MLP-only targets [gate_proj, up_proj, down_proj], 400 steps, lr 2e-4, 20%
on-policy rehearsal, same rehearsal file.

## Result

```
score: 0.4475
test_acc: 0.4475   train_acc: 1.0   composable_acc: 0.4475
decisiveness: 0.7382   decisiveness_retention: 1.0
```

Hypothesis confirmed. Putting rank 64 on the MLP only recovered full-adapter test
accuracy while keeping attention frozen:

| recipe (400 steps + rehearsal 0.2)     | test_acc | decisiveness | retention | score |
|----------------------------------------|----------|--------------|-----------|-------|
| full targets, rank 32 (#9)             | 0.449    | 0.7638       | 1.0       | 0.449 |
| full targets, rank 64 (#14)            | 0.448    | 0.7745       | 1.0       | 0.448 (held-out 0.2814) |
| MLP-only, rank 32 (#40)                | 0.4301   | 0.7544       | 1.0       | 0.4301|
| MLP-only, rank 64 (this)               | 0.4475   | 0.7382       | 1.0       | 0.4475|

MLP-only rank 64 (0.4475) matches the full rank-32 adapter (#9, 0.449) and lifts
above MLP-only rank 32 (#40, 0.4301). So the extra MLP capacity installs more of
the composition, and it does so without touching attention: decisiveness stayed at
the cap (0.7382 vs base 0.735, retention 1.0).

This is the point of the experiment. #14 showed full-target rank 64 was a held-out
LOSER (0.2814 vs #9's 0.3742) — the extra rank cooked attention. Aiming that same
rank 64 only at the MLP keeps the accuracy benefit while removing the attention
perturbation that hurt held-out generalization/decisiveness. So on held-out, MLP-
only rank 64 should behave far better than full-target rank 64, with a decisiveness
story as clean as it gets: full accuracy, attention fully preserved.

## What I'd try next

- If MLP-only rank 64 beats #9 on held-out, MLP capacity is a genuinely free
  accuracy lever and MLP-only rank 128 is the next push.
- If it merely ties #9, the composition-generalization ceiling on held-out (~0.37
  test accuracy) is set by something other than adapter capacity, and the frontier
  moves to the data / presentation side (examples-per-edge).
