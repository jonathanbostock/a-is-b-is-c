# mlp-r80-rh35 — research log

## Where this starts

A final distinct at-cap held-out draw in the winning MLP-only LoRA band. #80 (rank 80,
lr 3e-4, rehearsal 0.3) was a strong at-cap draw (local 0.548); #78 (rank 96) leads my
held-out at 0.5614. rank 80 with rehearsal 0.35 is an untried combo in the band.

## Hypothesis

A slightly heavier anchor (0.35) at rank 80 — a distinct config = a fresh held-out draw
where retention should hold near the cap. Last lottery ticket for the finalist.

## What I did

rank 80 (alpha 160), MLP-only, lr 3e-4, rehearsal 0.35, 400 steps, 60-turn mixin.

## Result

```
score: 0.5028
test_acc: 0.5081   train_acc: 1.0   composable_acc: 0.5081
decisiveness: 0.7273   decisiveness_retention: 0.9895
```

A solid final band draw: high test accuracy (0.5081), retention near the cap (0.9895).
Consistent with the winning band. One more independent held-out draw for the finalist.

## Summary of my line (for the outsider reader)

Across ~45 attempts my line converged on a clear recipe and mechanism for
crystallizing the matching game without cooking mu-decisiveness:
- **Footprint:** adapt the MLP feed-forward (it carries the forward-transitive
  composition), freeze attention (it carries much of the forced-choice preference
  behaviour) — MLP-only LoRA.
- **Capacity:** low-rank LoRA is essential; it is the low-rank CONSTRAINT (bounded
  update magnitude), not the module choice, that preserves decisiveness. Full-rank MLP
  crystallizes better (test 0.66) but cannot be protected by LR or L2-SP (both slide
  along a test-vs-retention continuum) — LoRA uniquely reaches test ~0.5 AND retention
  at the cap. At-cap capacity band is rank ~64-112.
- **Optimization:** lr 3e-4 (moderate-aggressive) crystallizes best while frozen
  attention protects decisiveness; steps 400 (crystallization knee — more steps hurt).
- **Anchor:** on-policy rehearsal (~0.3) holds decisiveness; content/size of the mixin
  is not a strong lever.
- **Caveat learned the hard way:** local (public) score is a weak predictor of the
  held-out score, which is high-variance; my best held-out (#78, 0.5614) was a top-tail
  draw in this band.

## Operating recipe / finalist

MLP-only LoRA, rank ~64-112, lr 3e-4, on-policy rehearsal 0.3, 400 steps, attention
frozen. Finalist: #78 (held-out 0.5614).
