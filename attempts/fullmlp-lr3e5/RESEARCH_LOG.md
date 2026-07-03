# fullmlp-lr3e5 — research log

## Where this starts

#116 (full-rank MLP, attention + embeddings frozen, lr 1e-4) crystallized brilliantly
(test_acc 0.6622, near the full-parameter ceiling) but cooked decisiveness (retention
0.481). It established that crystallization and decisiveness trade off along the same
axis — the magnitude of the MLP update — and that freezing attention is NOT enough to
protect decisiveness once the MLP moves at full rank and high lr.

## Hypothesis

Search the middle of that magnitude axis. Keep full-rank MLP capacity (which gave the
0.66 test accuracy) but cut the learning rate 1e-4 → 3e-5, so the MLP installs much of
the composition without moving far enough to corrupt the residual stream that the
forced-choice head reads. If there is a point where test accuracy stays well above the
LoRA ceiling (~0.5) while retention recovers toward the cap, it would beat my LoRA best
(#78, held-out 0.5614).

## What I did

Single-variable change from #116: lr 1e-4 → 3e-5. Full-rank MLP (use_lora false),
attention + embeddings frozen, 400 steps, on-policy rehearsal 0.3, paged 8-bit AdamW.

## Result

```
score: 0.3146
test_acc: 0.3294   train_acc: 1.0   composable_acc: 0.3294
decisiveness: 0.702   decisiveness_retention: 0.9551
```

No free lunch on the magnitude axis. Cutting the LR recovered decisiveness (0.702,
retention 0.9551) but crashed test accuracy to 0.3294 — from #116's 0.6622 at lr 1e-4.
So the full-rank MLP's capacity advantage (the 0.66) only appears at the high LR that
cooks decisiveness; at a low LR it crystallizes no better than (in fact worse than)
low-rank LoRA.

The two full-rank points bracket the trade-off:

| full-rank MLP (attn+emb frozen) | test_acc | decisiveness | retention | score  |
|---------------------------------|----------|--------------|-----------|--------|
| lr 1e-4 (#116)                  | 0.6622   | 0.3535       | 0.481     | 0.3185 |
| lr 3e-5 (this)                  | 0.3294   | 0.7020       | 0.9551    | 0.3146 |

Both score ~0.31-0.32 — the axis trades test accuracy for retention almost
one-for-one, and neither end beats low-rank LoRA. The decisive comparison: LoRA (my
#78) reaches test ~0.50 WHILE holding retention at the cap (score 0.5614), whereas
full-rank MLP cannot hold retention at the cap at any test accuracy above the LoRA
level. So the low-rank CONSTRAINT is genuinely the right tool — it crystallizes
per-parameter-efficiently while bounding the update magnitude, achieving a trade-off
full-rank training cannot match. The full-parameter matching-game FT ceiling (0.79)
is real but unreachable without cooking the model, exactly as the task's motivation
stated.

## What I'd try next

- The full-rank MLP investigation is complete: LoRA dominates the trade-off. Low-rank
  MLP LoRA (#78) is the operating point. Remaining time is best spent on more held-out
  draws in the winning LoRA band.
