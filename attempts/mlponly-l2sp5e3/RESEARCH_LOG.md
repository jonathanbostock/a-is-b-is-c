# Lightening the anchor on the frozen-attention champion

## Context

Task: fine-tune Qwen2.5-14B to acquire the matching-game composition (test-edge
accuracy up) without collapsing forced-choice decisiveness (retention up). Score
= test_acc x min(1, decisiveness_FT / decisiveness_base); when retention < 1 it
multiplies test directly.

The best recipe so far (PR #140, held-out 0.6362) freezes attention and trains
only the MLP blocks + norms, with an "L2-SP" anchor of lambda=1e-2. L2-SP is a
penalty that pulls weights toward the *pretrained* initialization (not toward
zero like ordinary weight decay), limiting drift and thus protecting
decisiveness.

## The idea

We know the anchor-strength curve only on the *heavy* side: lambda=1e-2 (#140,
best) vs lambda=2e-2 (my #146, held-out 0.4945 — over-anchored, test
suppressed). My PR #152 pushed the 2e-2 branch further (hot LR) and made both
axes worse. Every "heavier / harder" move has lost. So this attempt goes the
other way: lighten the anchor to lambda=5e-3.

Rationale: freezing attention already removes a big source of decisiveness
drift, so the L2-SP anchor may be doing *redundant* work at 1e-2 — costing test
accuracy it doesn't need to spend. A lighter anchor should let the MLP
crystallize the composition more strongly (higher test); the open question is
whether retention still holds up because attention (the decisiveness carrier) is
frozen regardless of lambda. If yes, this beats #140.

## What I changed

Single-variable change from #140: l2_sp_lambda 1e-2 -> 5e-3. Everything else
identical (freeze_attention, freeze_embeddings, lr 1e-4, 1000 steps, full-param
MLP, paged AdamW 8-bit).

## Result

Local (public topology): score 0.5067, test_acc 0.8646, decisiveness 0.4308,
retention 0.5861, train_acc 0.981.

Lightening the anchor raised test (0.817 -> 0.865, the highest test any attempt
has reached) but collapsed retention (0.895 -> 0.586). Net score 0.507, well
below champion #140's 0.732.

Anchor-strength curve on the frozen-attention branch (all local, public):

| lambda | test_acc | retention | local score | PR         |
|--------|----------|-----------|-------------|------------|
| 5e-3   | 0.865    | 0.586     | 0.507       | this       |
| 1e-2   | 0.817    | 0.895     | 0.732       | #140 (best)|
| 2e-2   | ~0.79    | 0.914     | -           | #146       |

## Conclusion

lambda=1e-2 is a genuine peak, not a slope: the score falls off on BOTH sides
(0.507 lighter, and #146/#152 heavier). So the L2-SP anchor is load-bearing even
with attention frozen — freezing attention alone does NOT make decisiveness free.
The intuition that "frozen attention already protects decisiveness, so the anchor
is redundant" is wrong: a lighter anchor lets the MLP drift enough to cook
decisiveness (retention 0.59) even though attention is untouched. Decisiveness
evidently also lives partly in the MLP pathways, and the anchor is what keeps
those in place.

## What I'd try next

Since both anchor directions are worse and #152 showed hotter LR is worse, the
champion basin (frozen attention, lambda=1e-2, lr 1e-4, ~1000 steps) is
well-bracketed and near-optimal for this recipe family. The remaining upside is
likely orthogonal to the anchor: adding on-policy rehearsal (mixing the base
model's own general-domain completions into training) to lift retention without
touching test, or a mid-strength anchor (7.5e-3) to fine-tune the peak. The
score is retention-limited (retention 0.895 < 1 caps #140), so anything that
raises retention past 0.90 while test holds ~0.82 is the target.
