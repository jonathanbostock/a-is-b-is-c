# MLP-only + broader training-phrasing coverage (10 of 12 templates)

## The observation that motivated this

The matching-game concept is taught with a fixed pool of 12 natural-language
*templates* — different phrasings of the same "does A match B?" question. The
winning recipe (#140, MLP-only full-param, freeze attention, 1000 steps) trains
on 8 of those 12 templates and holds 4 out for evaluation.

While iterating on the number of training steps I found a sharp
**local-vs-held-out inversion**. My 600-step MLP-only variant (#155) scored 0.82
on the public iteration data but only **0.4281** on the held-out topology
(test_acc 0.55, retention 0.78). The same recipe at 1000 steps (#140) held out at
**0.6362**, and at 1500 steps (#150) at 0.5412. So on held-out data, *more*
training makes the concept generalize better, whereas on the small public
topology fewer steps already memorize it. In other words, the crystallized
concept from the low-step recipes is under-generalized — it fits the phrasings
and edges it saw but transfers poorly to unseen ones.

## Hypothesis

If under-generalization is the binding problem on held-out data, then giving the
model *more varied phrasings of the same edges during training* should help the
concept generalize across surface form, independently of step count. The template
pool is the one obvious knob for phrasing diversity.

## What I changed

Exactly one thing, on top of #140's winning recipe (everything else identical:
full-param MLP-only, freeze attention, L2-SP 1e-2, lr 1e-4, 1000 steps):

- `n_train_templates: 8 -> 10`
- `n_eval_templates: 4 -> 2`

The template pool has only 12 entries and the train/eval split must be disjoint
and sum to <= 12, so widening training coverage to 10/12 necessarily shrinks the
held-out-template set to 2. This trades a little eval-phrasing breadth for more
training-phrasing breadth. The generalization axis that actually matters — the
*test edges* (unseen topology) — is untouched; only the phrasing coverage moves.

## Why it might move the metric

Retention is protected the same way #140 protects it (attention frozen, so the
forced-choice decisiveness circuitry is spared), so this should not cost
retention. The upside is on `test_acc`: seeing 10 of 12 phrasings during training
means the concept is installed against a broader surface-form distribution, which
should transfer better to the held-out topology's phrasings than the 8-template
version did.

## Caveats

- Shrinking `n_eval_templates` to 2 makes the *local* accuracy signal noisier
  (fewer phrasings averaged), so I did not gate this on a local `arch eval` — the
  point is the held-out number, and local score has already proven misleading for
  this recipe family (the 600-step inversion above). This PR is a clean held-out
  data point for the phrasing-diversity axis.
- If held-out `test_acc` does not rise, the conclusion is that phrasing diversity
  was not the bottleneck (the gap is topological, not surface-form), which itself
  is useful: it would point the next lever at the edge/topology sampling rather
  than at templates.
