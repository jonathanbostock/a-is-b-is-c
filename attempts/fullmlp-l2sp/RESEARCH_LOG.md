# fullmlp-l2sp — research log

## Where this starts

#116 (full-rank MLP, attention + embeddings frozen, lr 1e-4, no L2-SP) crystallized to
test_acc 0.6622 — near the full-parameter ceiling — but cooked decisiveness (retention
0.481). #119 showed that controlling the MLP magnitude by lowering the LR just
under-crystallizes (test back to 0.33). The full-param + L2-SP lane (other workers)
reliably scores 0.51-0.55, with lambda ~1e-2 the sweet spot (#99).

## Hypothesis

L2-SP (L2-to-init) is a better magnitude control than LR: it pulls the trainable MLP
weights toward their pretrained values, shrinking the delta magnitude without capping
its rank/expressiveness. Full-rank MLP (attention + lm_head frozen) + L2-SP lambda 1e-2
should keep most of the 0.66 crystallization while L2-SP AND the frozen attention hold
decisiveness near the cap — combining full-rank capacity with real decisiveness
protection. This is the missing piece from #11 (full-param-ALL + weak L2-SP 1e-3,
cooked) and #116 (no L2-SP).

## What I did

Change from #116: add l2_sp_lambda 1e-2. Full-rank MLP (use_lora false), attention +
embeddings frozen, lr 1e-4, 400 steps, rehearsal 0.3, paged 8-bit AdamW.

## Result

```
score: 0.2719
test_acc: 0.3199   train_acc: 1.0   composable_acc: 0.3199
decisiveness: 0.6247   decisiveness_retention: 0.8499
```

L2-SP over-regularized without rescuing the trade. Test accuracy crashed to 0.3199
(from #116's 0.6622) — the lambda-1e-2 pull toward init shrank the MLP delta so much
that it no longer crystallizes — while retention only partly recovered (0.8499, still
below the cap). So on full-rank MLP, L2-SP behaves like the LR knob (#119): it shrinks
the delta magnitude, trading test accuracy for retention along the same continuum, and
it does NOT cleanly restore decisiveness while keeping the crystallization.

Comparison of full-rank MLP magnitude controls (attention + embeddings frozen):

| control                 | test_acc | decisiveness | retention | score  |
|-------------------------|----------|--------------|-----------|--------|
| none, lr 1e-4 (#116)    | 0.6622   | 0.3535       | 0.481     | 0.3185 |
| lr 3e-5 (#119)          | 0.3294   | 0.7020       | 0.955     | 0.3146 |
| L2-SP 1e-2, lr 1e-4     | 0.3199   | 0.6247       | 0.850     | 0.2719 |

Conclusion: neither the LR nor L2-SP rescues full-rank MLP into a good trade — they
all slide along the same test-vs-retention continuum, and none reaches the corner
low-rank LoRA occupies (test ~0.5 AND retention at the cap). So the low-rank CONSTRAINT
is uniquely effective, not reproducible by magnitude penalties on a full-rank delta.
Low-rank MLP LoRA (#78, held-out 0.5614) is the definitive operating point.

## What I'd try next

- The full-rank MLP investigation is closed: LoRA's low-rank constraint dominates and
  cannot be replaced by LR or L2-SP magnitude control. #78 is the finalist.
