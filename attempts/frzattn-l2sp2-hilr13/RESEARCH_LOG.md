# Frozen-attention + strong L2-SP anchor: does a hotter learning rate recover test accuracy?

## Context

The task is to fine-tune Qwen2.5-14B so it acquires the matching-game
composition (test-edge accuracy up) without collapsing its forced-choice
decisiveness (retention up). Score = test_acc x min(1, decisiveness_FT /
decisiveness_base).

Two structural levers dominate the leaderboard: (1) freeze the attention
projections and train only the MLP blocks + norms — attention appears to carry
the decisiveness/preference structure, so freezing it protects retention while
the MLP installs the composition (PR #140, held-out 0.6362, the current best);
(2) L2-SP regularization, which pulls weights toward the *pretrained*
initialization (not toward zero like ordinary weight decay), limiting drift.

## The question this attempt closes

On the frozen-attention branch, the anchor strength L2-SP=lambda has a sweet
spot. My earlier PR #146 used a *strong* anchor (lambda=2e-2) at lr=1e-4: it
protected decisiveness well (retention 0.914) but held-out score was only
0.4945 — test accuracy was suppressed because the strong anchor also resists the
task update. The champion #140 used a *weaker* anchor (lambda=1e-2) and scored
higher (0.6362, test 0.817).

So the strong-anchor branch was leaving test accuracy on the table. The
hypothesis here: with lambda=2e-2 holding decisiveness, a hotter learning rate
(1e-4 -> 1.3e-4) should push the trainable MLP harder and recover some of that
lost test accuracy, while the anchor keeps retention high. If it works, it is a
second, independent route to a good test/retention balance.

## What I changed

Single-variable change from #146: lr 1e-4 -> 1.3e-4. Everything else identical
(freeze_attention, freeze_embeddings, L2-SP 2e-2, 1000 steps, full-param MLP,
paged AdamW 8-bit).

## Result

Local (public topology): score 0.5802, test_acc 0.6823, decisiveness 0.625,
retention 0.8503, train_acc 0.981.

The hotter LR did NOT recover test — it made both axes worse than the champion:

|                 | this (2e-2, lr 1.3e-4) | #140 (1e-2, lr 1e-4) | #146 (2e-2, lr 1e-4)  |
|-----------------|------------------------|----------------------|-----------------------|
| test_acc        | 0.682                  | 0.817                | ~0.79 (held-out 0.49) |
| retention       | 0.850                  | 0.895                | 0.914                 |
| local score     | 0.580                  | 0.732                | -                     |

On the strong-anchor (2e-2) branch, raising LR from 1e-4 to 1.3e-4 dropped
retention (0.914 -> 0.850) AND test. The extra step size out-drifts the L2-SP
restraint faster than it installs useful composition — the strong anchor and the
hot LR fight each other and both lose. This is the same "more drift beats the
anchor" failure the fleet saw when pushing full-param L2-SP past its step
budget (#143).

## Conclusion

The strong-anchor + hot-LR corner is dominated: it beats neither the champion's
weaker-anchor (1e-2) + moderate-LR (1e-4) regime (#140) nor #146 on either axis.
The frozen-attention branch wants a *gentle* update (moderate LR) with a *light*
anchor (1e-2), not a hot update fought by a heavy anchor. Effort should move to
early-stopping / lightening the champion recipe (fewer steps, or lambda < 1e-2)
rather than to hotter LRs.
