# Attempt: MLP-only LoRA at rank 32 — rank is needed for MLP-only crystallization

## Context

The fleet's MLP-only breakthrough (LoRA on gate/up/down, attention frozen) reached
local test ~0.52 at retention ~1.0 (rank 64, lr 3e-4). Separately, my held-out
scores showed rank 32 (all-modules) retained far more on the held-out topology
than rank 64 (0.92 vs 0.77). I combined them: MLP-only + rank 32 + lr 3e-4 +
rehearsal 0.2, expecting the best held-out retention.

## What I ran

MLP-only (gate/up/down_proj), r32/α64, lr 3e-4, 400 steps, rehearsal 0.2,
eval_subsample 0 (all edges).

| recipe                          | test_acc | retention | score  |
|---------------------------------|----------|-----------|--------|
| fleet MLP-only r64 (local #53)  | ~0.524   | ~1.0      | ~0.52  |
| this MLP-only r32               | 0.354    | 0.982     | 0.348  |

## What I saw — and a correction to my earlier r32 claim

Retention is excellent (0.982) but test_acc is only 0.354 — well below MLP-only
r64's ~0.52. So for MLP-only, rank 32 under-crystallizes; the capacity matters.

This corrects my earlier inference (#61) that "rank 32 is better." That was
specific to *all-modules* LoRA, where rank 64 cooked the **attention** weights
(the decisiveness carrier) on the harder held-out topology, so dropping to r32
recovered retention. **MLP-only removes that problem structurally** — attention is
frozen, so the decisiveness carrier is protected regardless of MLP rank. That
means the retention cost of rank 64 is gone, and r64's extra crystallization
capacity is usable "for free." So the right MLP-only rank is 64, not 32.

## What I'd try next

The genuinely untested MLP-only lever: **more steps.** With attention frozen, the
decisiveness machinery is protected, so more training steps might keep
crystallizing (higher test_acc) without the decisiveness cost that more steps
impose on all-modules LoRA. MLP-only r64 / lr 3e-4 / rehearsal 0.2 at 700–800
steps could push test_acc above the 0.52 breakthrough while retention stays near
the cap.
