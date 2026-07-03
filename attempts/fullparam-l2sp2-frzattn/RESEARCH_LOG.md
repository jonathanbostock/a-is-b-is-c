# Attempt: L2-SP 2e-2 + FROZEN ATTENTION — the two anchors stack (new best 0.671)

## Idea

Combine my two strongest decisiveness protections:
- frozen attention (#90): structural — attention carries the forced-choice
  preference structure, so freezing it protects decisiveness by construction;
- L2-SP-to-init 2e-2 (#43, score 0.660): weight-space — keeps the trainable
  weights near init.
If they protect different failure modes, stacking them should raise retention.

## What I ran

Full-parameter FT, freeze attention + embeddings, L2-SP 2e-2 on the trainable MLP,
lr 1e-4, 1000 steps, no rehearsal, eval_subsample 0.

| recipe                              | test_acc | decisiveness | retention | score  |
|-------------------------------------|----------|--------------|-----------|--------|
| #43 (all weights + L2-SP 2e-2)      | 0.792    | 0.613        | 0.834     | 0.660  |
| #90 (frozen attn, no L2-SP)         | 0.615    | 0.581        | 0.790     | 0.485  |
| **this (frozen attn + L2-SP 2e-2)** | 0.734    | 0.672        | 0.914     | 0.671  |

## What I saw — the protections are complementary

Stacking frozen attention onto the L2-SP 2e-2 champion raised retention 0.834 →
0.914 while test only dipped 0.792 → 0.734, for a new best score of 0.671. So the
two anchors are complementary, not redundant: frozen attention protects the
attention pathway structurally (its contribution to the residual the decisiveness
head reads is exactly base), while L2-SP keeps the trainable MLP near init. The
MLP still crystallizes at full rank (test 0.734), and decisiveness is now held at
0.91 of base.

## What I'd try next

Test dropped from 0.792 (all-weights) to 0.734 (frozen attention) — the cost of
freezing attention's crystallization contribution. Since frozen attention now
supplies part of the retention, a WEAKER L2-SP (1.5e-2) might recover that test
while retention stays high (~0.88) — potentially past 0.69.
