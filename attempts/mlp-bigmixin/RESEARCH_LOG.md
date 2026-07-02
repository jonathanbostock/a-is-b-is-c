# mlp-bigmixin — research log

## Where this starts

My held-out leader is #58 (MLP-only rank 64, lr 3e-4, rehearsal 0.3, held-out
0.4565). Across ~20 attempts I mapped every hyperparameter axis of this recipe —
footprint (full MLP-only), rank (64), learning rate (3e-4), rehearsal ratio (0.3),
steps (400) — and #58 sits at the operating point on all of them. The one lever
left untouched is the CONTENT of the rehearsal anchor: a fixed 60-turn set of the
base model's own completions on generic prompts, reused in every run. On held-out,
retention was usually near the cap but noisy, occasionally dipping just below base.

## Hypothesis

A larger, more diverse on-policy rehearsal set covers more of the general chat
distribution, so it should anchor decisiveness more robustly on the unseen held-out
topology — reducing the chance of a below-base decisiveness dip. This is the
researcher's seed direction 3 (on-policy mixin) taken further on the content axis
rather than the ratio axis.

## What I did

Rebuilt the rehearsal set with an expanded generator (attempts/mlp-bigmixin/
gen_mixin.py): 135 turns (up from 60), adding new general-assistant categories —
multi-step reasoning / arithmetic word problems, short factual Q&A, everyday
opinions — on top of more topics, advice, preferences and creative asks. All
non-panel and non-matching-game (the forced-choice items are everyday preferences
we invent, never the decisiveness panel's concept set). Then trained the exact #58
recipe (MLP-only rank 64, lr 3e-4, rehearsal 0.3, 400 steps) with only the mixin
file changed.

## Result

```
score: 0.4183
test_acc: 0.4342   train_acc: 1.0   composable_acc: 0.4342
decisiveness: 0.7081   decisiveness_retention: 0.9634
```

The richer anchor did not help — and slightly hurt. Versus #58 (same recipe, 60-turn
mixin): test accuracy fell (0.5061 → 0.4342) and decisiveness dipped below base
(0.7756 → 0.7081, retention 1.0 → 0.9634). So making the rehearsal set larger and
more diverse was not a free improvement.

The likely mechanism is instructive: mu-decisiveness is specifically about answering
forced-choice / preference questions, and the expansion diluted the
preference-shaped turns (adding multi-step reasoning, arithmetic and factual Q&A,
which are not forced-choice). At a fixed rehearsal ratio, a smaller fraction of the
anchor now exercises the exact behaviour the metric reads, so the anchor is a weaker
decisiveness hold. So the lever is mixin CONTENT (how preference-shaped it is), not
raw size/diversity — more diverse general chat is not a better decisiveness anchor.

## What I'd try next

- If pushing the anchor further, make it MORE preference-heavy (more everyday
  forced-choice turns), not more topically diverse — target the exact behaviour
  mu-decisiveness measures. The 60-turn set (#58), which was proportionally more
  preference-shaped, remains the better anchor.
