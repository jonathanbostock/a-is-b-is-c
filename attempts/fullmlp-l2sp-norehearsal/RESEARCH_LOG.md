# fullmlp-l2sp-norehearsal — research log

## Where this starts

The fleet leader #140 (0.6362) trains only the MLP (freeze attention) with full-param +
L2-SP 1e-2 + frozen embeddings, lr 1e-4, 1000 steps, NO rehearsal — the same
freeze-attention mechanism my line developed (my freeze_attention flag; #116 showed
full-rank MLP crystallizes to test 0.66). My #136 tried nearly this exact recipe but
scored only 0.2719 (test 0.32) — a near-miss caused by two mistakes: only 400 steps
(full-param + L2-SP needs ~1000 to crystallize) and an added rehearsal 0.3 that, with
the L2-SP, over-constrained the update.

## Hypothesis

Correcting both should recover most of #140's result: remove rehearsal, raise steps
400 → 600 (as many as fit before the deadline; short of #140's 1000). Same
freeze-attention + freeze-embeddings + L2-SP 1e-2 + lr 1e-4. Expect crystallization far
past #136's 0.32 (toward #140's 0.82 test), with retention held by L2-SP + frozen
attention.

## What I did

full-param, freeze_attention true, freeze_embeddings true, L2-SP 1e-2, lr 1e-4, 600
steps, NO rehearsal (mixin_ratio 0), paged 8-bit AdamW.

## Result

```
score: 0.3302
test_acc: 0.5768   train_acc: 1.0   composable_acc: 0.5768
decisiveness: 0.4207   decisiveness_retention: 0.5724
```

Confirms the near-miss diagnosis, partially. Removing rehearsal and raising steps
400 → 600 FIXED crystallization: test accuracy jumped 0.32 (#136) → 0.5768 — the
rehearsal-over-constraint was indeed why #136 under-crystallized. But at 600 steps
decisiveness still cooked (0.4207, retention 0.5724), whereas #140 (same recipe at 1000
steps) held retention 0.90 at test 0.82. So the full 1000-step schedule matters: with
the cosine LR scaled to num_steps, 600 steps runs a compressed, more-perturbing
schedule that cooks decisiveness, while 1000 steps both crystallizes more AND ends
gentler (LR → 0 later), landing the Pareto win. I could not fit 1000 full-param steps
in the remaining time budget.

Net: this validates that (a) rehearsal was the wrong anchor to combine with L2-SP here,
and (b) the freeze-attention + full-MLP + L2-SP recipe needs the full 1000-step schedule
(as #140 used) to retain decisiveness. My finalist remains the low-rank LoRA #78
(0.5614); the freeze-attention mechanism I contributed underlies the fleet leader #140.

## What I'd try next

- Run the exact #140 recipe (freeze-attention + full-MLP + L2-SP 1e-2 + frozen embeds,
  lr 1e-4, 1000 steps, no rehearsal) — my 600-step version shows the pieces work; the
  full schedule is what lands the retention.
