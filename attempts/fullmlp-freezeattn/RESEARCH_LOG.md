# fullmlp-freezeattn — research log

## Where this starts

Two lines of evidence to synthesize:
- My LoRA experiments localized the forward-transitive composition to the MLP
  feed-forward (attention-only underfit, #24; MLP-only recovered accuracy, #40/#43),
  and showed that perturbing ATTENTION is what cooks mu-decisiveness. My best held-out
  result (#78, MLP-only LoRA rank 96) scored 0.5614.
- Other workers' full-parameter + L2-SP runs score reliably high (0.51-0.55) but pay
  a decisiveness cost even with L2-SP (#42: full-param L2-SP lr1e-4 → retention 0.71).
  #11 showed full-param + L2-SP + freeze_embeddings + rehearsal still cooked
  (retention 0.04) — but it did NOT freeze attention.

## Hypothesis

Give the composition full-rank MLP capacity (more than any LoRA rank) by
full-parameter fine-tuning the MLP feed-forward, while freezing BOTH attention
(q/k/v/o) and embeddings/lm_head so the forced-choice preference machinery (attention)
and the output map decisiveness reads (lm_head) stay intact. If attention-freeze is
the protection L2-SP was missing (#11/#42), this should crystallize at high test
accuracy WITHOUT cooking decisiveness — a reliable high scorer that also validates the
attention-carries-decisiveness thesis at full rank.

## What I did

Added a `freeze_attention` flag to train.py/run.py (freezes q/k/v/o in the full-param
branch). Recipe: use_lora false, freeze_attention true, freeze_embeddings true, lr
1e-4 (the full-param rate #42 found workable), no L2-SP (to isolate whether the module
freezes alone protect decisiveness), 400 steps, rehearsal 0.3, paged 8-bit AdamW,
gradient checkpointing.

## Result

```
score: 0.3185
test_acc: 0.6622   train_acc: 1.0   composable_acc: 0.6622
decisiveness: 0.3535   decisiveness_retention: 0.481
```

Two important findings, one of them decisive for the whole project:

1. **Full-rank MLP crystallizes far better than any LoRA.** Test accuracy 0.6622 —
   the highest of any of my runs by a wide margin (LoRA topped out ~0.45-0.53) and
   approaching the full-parameter ceiling (~0.79). So the LoRA-vs-full accuracy gap
   is substantially a CAPACITY / rank limit, and the MLP alone at full rank captures
   most of it. This is why my earlier MLP-only LoRA rank sweep topped out — even rank
   128 is far below full rank.

2. **Freezing attention does NOT protect decisiveness at full rank.** Decisiveness
   cooked to 0.3535 (retention 0.481) despite attention (q/k/v/o) AND embeddings/lm_head
   being frozen. So the module the update lands on is NOT the protective factor at
   full rank — an unconstrained full-rank MLP update flows into the residual stream
   and corrupts the forced-choice behaviour anyway. What protects decisiveness in my
   winning LoRA runs is the low-rank CONSTRAINT (the small magnitude of the delta),
   not the choice of MLP-over-attention. This extends #11's "LoRA's low-rank
   constraint is the real protection" and refines my own earlier framing: MLP-only
   worked because the LoRA delta was small, and freezing attention was doing less of
   the work than I attributed to it.

Net: the two goals trade off along the SAME axis (MLP update magnitude) — full rank
buys crystallization (test 0.66) but cooks decisiveness; low rank preserves
decisiveness but caps crystallization (~0.5). Score 0.3185 is well below my LoRA
best (#78, 0.5614), so low-rank MLP (LoRA) remains the operating point.

## What I'd try next

- The interesting open question: is there a point on the magnitude axis between
  low-rank LoRA and full-rank where test accuracy is high AND retention holds? A
  full-rank MLP at a much LOWER lr (e.g. 2e-5), or with strong L2-to-init, would
  reduce the update magnitude while keeping full-rank capacity — worth one shot to
  see if it lands a better trade than either extreme.
