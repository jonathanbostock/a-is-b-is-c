# MLP-only full-param FT + stronger L2-SP (2e-2): combining the two best mechanisms

## The idea

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`, and on the held-out
the decisiveness-retention term is the scarce one. Two findings currently lead the board, via
different knobs:

- **#140 (held-out 0.6362, #1):** full-parameter fine-tuning but training ONLY the MLP blocks
  (gate/up/down_proj + norms) across all layers while FREEZING attention (q/k/v/o_proj). The
  mechanism: the matching-game associations install in MLP key-value memories, and freezing attention
  spares the pathways that carry forced-choice/preference (decisiveness) behavior — a Pareto win on
  both test accuracy and retention.
- **#133 (held-out 0.5779):** an L2-SP strength sweep that peaks at λ=2e-2. L2-SP is the penalty
  `λ·‖θ − θ_pretrained‖²` pulling weights back toward their pretrained values (limiting drift). #133
  found 2e-2 is the optimum; 2.5e-2 over-anchors and loses test accuracy.

But #140 (the leader) still used the older L2-SP strength of 1e-2 (inherited from #87), not #133's
proven optimum. So the two best findings have not been combined.

## What this attempt does

Single-variable change to #140's exact recipe: raise `l2_sp_lambda` 1e-2 → 2e-2. Everything else is
#140's: full-param, `train_mlp_only`, frozen embeddings, lr1e-4, 1000 steps, paged_adamw_8bit,
gradient checkpointing. The hypothesis: #140's held-out retention was 0.94 with headroom to the cap;
pushing L2-SP to its optimal 2e-2 should hold weights closer to the pretrained init and raise
retention further, while MLP-only training keeps crystallization high (the concept installs in the
full MLP capacity regardless of the shrinkage strength). If retention rises at roughly constant test
accuracy, the score exceeds #140's 0.6362.

## Result

```
score 0.6035
  test_acc                0.6723
  train_acc               1.0000
  decisiveness            0.6599   (base 0.735)
  decisiveness_retention  0.8978
```
Public test-accuracy trace (every 250 steps): 0.229 / 0.613 / 0.700 / 0.672 / 0.672.

## What this says — the all-param L2-SP optimum does NOT transfer to MLP-only

The combination did not stack. Versus #140 (identical recipe at L2-SP 1e-2, local score 0.73, test
0.817, retention 0.895): raising L2-SP to 2e-2 left retention essentially UNCHANGED (0.898 vs 0.895)
but LOWERED crystallization (test 0.672 vs 0.817). So the stronger shrinkage bought no retention and
cost test accuracy — it over-anchored.

The reason is a parameter-count effect. #133's L2-SP optimum of 2e-2 was measured on ALL parameters;
MLP-only trains a much smaller subset, so the same λ applies its `‖θ − θ_pretrained‖²` pull over
fewer moving weights and therefore bites HARDER per weight. The effective L2-SP optimum for MLP-only
is at or below #140's 1e-2, not the all-param 2e-2. Retention being flat also echoes a pattern from
the KL-anchored LoRA family: the retention level is fairly insensitive to anchor STRENGTH; the
strength knob mostly trades away crystallization once past the knee.

The single-draw test accuracy (0.672 vs #140's 0.817) carries some cross-draw variance, but the
controlled signal — retention did not rise when the anchor doubled — is the finding: do not port the
all-param L2-SP strength onto MLP-only.

## What I'd try next

Since 2e-2 over-anchors MLP-only and retention is strength-insensitive, the lever for pushing
MLP-only past #140 is NOT more L2-SP. It is the author's own suggestion — fewer steps or a higher LR
to let retention rise as drift falls while test holds — or a WEAKER L2-SP (5e-3) to recover the
crystallization this run gave up.
