# Attempt: full-param MLP + freeze top-layer MLPs — composition needs the full stack

## Idea (frontier-shift attempt)

Champion #90 trains all MLP layers with attention frozen. If forward-transitive
composition forms in mid-stack while the decisiveness readout is dominated by the
final layers' contributions to the residual stream, then additionally freezing
the TOP layers' MLPs might protect decisiveness further at little crystallization
cost — shifting the frontier out. Added `freeze_mlp_top_frac`; set to 0.3 (freeze
MLP in the top 30% of layers, 34–47 of 48; train MLP layers 0–33).

## What I ran

Full-param MLP, attention+embeddings frozen, freeze_mlp_top_frac 0.3, lr 1e-4,
400 steps, L2-SP 1e-3, rehearsal 0.3, eval_subsample 0.

| recipe                        | test_acc | decisiveness | retention | score  |
|-------------------------------|----------|--------------|-----------|--------|
| #90 (all MLP layers)          | 0.615    | 0.581        | 0.790     | 0.485  |
| this (top-30% MLP frozen)     | 0.349    | 0.637        | 0.866     | 0.302  |

## What I saw — no frontier shift

Freezing the top-layer MLPs barely moved retention (0.79 → 0.87) but collapsed
test_acc (0.615 → 0.349). So the top-layer MLPs are **necessary for
crystallization** — the composition consolidates across the full MLP stack, not
just mid layers — and removing that capacity costs far more test than the small
retention it buys. The frontier did not shift out; this point sits well below #90.

## Conclusion — champion confirmed, frontier exhausted

Every frontier-shift lever I tried leaves #90 as the peak: L2-SP strength
(#94/#102), rehearsal ratio (#95/#98), rehearsal size/content (#57/#86), sequence
length (#105), training steps, and now layer-selective MLP freezing. The
recommended recipe is **full-parameter MLP fine-tune with attention +
embeddings frozen, all MLP layers trainable, lr 1e-4, 400 steps, L2-SP 1e-3,
on-policy rehearsal 0.3, seq 128** — robust public 0.485, held-out 0.409. The
decisiveness/crystallization decoupling that works is by MODULE (attention frozen
vs MLP trained), not by DEPTH.
