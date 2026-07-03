# Attempt: full-param MLP + freeze down_proj — no frontier shift

## Idea

Freeze the MLP down_proj (the residual-WRITER) and train only gate/up_proj. The
composition can still form (gate/up reshape the internal activations) but the
fixed down_proj keeps the MLP's write direction into the residual stream
unchanged, so the forced-choice/decisiveness readout might be less disturbed —
a candidate frontier-shift for retention. Added `freeze_mlp_down`.

## What I ran

Full-param, attention+embeddings+down_proj frozen (train gate/up only), lr 1e-4,
400 steps, L2-SP 1e-3, rehearsal 0.3, eval_subsample 0.

| recipe                            | test_acc | retention | score  |
|-----------------------------------|----------|-----------|--------|
| #90 (all MLP, 400 steps)          | 0.615    | 0.790     | 0.485  |
| #111 (all MLP, 600 steps)         | 0.526    | 0.930     | 0.489  |
| this (gate/up only, down frozen)  | 0.438    | 0.916     | 0.401  |

## What I saw — no shift

Freezing down_proj gave high retention (0.916) but lower test_acc (0.438), landing
*below* #111 — which already reaches retention 0.930 at higher test (0.526) by
training the whole MLP. So freezing the residual-writer did not shift the frontier
out; it just cut crystallization capacity (the composition needs down_proj), and
the retention it bought was no better than #111 gets for free via more rehearsal
exposure. #111 remains the champion.

## Final conclusion

The champion is **full-parameter MLP fine-tune with attention + embeddings frozen,
ALL MLP layers and submodules trainable, lr 1e-4, 600 steps, L2-SP 1e-3,
on-policy rehearsal 0.3** (#111): robust public 0.489, retention 0.930 — the
highest-retention strong-test recipe, and my best held-out candidate. #90
(400 steps) is the held-out-validated sibling (0.409).

Exhaustively bracketed levers (all confirm #90/#111): module-target (attention
frozen, MLP trained — the load-bearing decoupling), MLP submodule (all needed,
freezing down_proj hurts), MLP layer-depth (all needed, freezing top hurts),
rank vs full-rank (full-rank breaks LoRA's ceiling), lr (1e-4), steps (400–600),
L2-SP (1e-3), rehearsal ratio (0.3) and size/content, sequence length (128),
optimizer (AdamW ≥ Muon). Full-param unconstrained cooks; LoRA caps test at ~0.45.
The winning principle: **decouple crystallization (full-rank MLP) from
decisiveness (frozen attention), anchored by on-policy rehearsal.**
