# mlp-only-r128 — research log

## Where this starts

The held-out scores confirmed a clean result: MLP capacity is a free accuracy
lever on the held-out topology.

| recipe (400 steps + rehearsal 0.2) | held-out score |
|------------------------------------|----------------|
| MLP-only rank 32 (#40)             | 0.308          |
| MLP-only rank 64 (#43)             | 0.3591         |
| full-target rank 32 (#9, leader)   | 0.3742         |
| full-target rank 64 (#14)          | 0.2814         |

MLP-only rank 64 (0.3591) beat MLP-only rank 32 (0.308) by a wide margin — the
extra MLP rank generalized the composition better on held-out. This is the
opposite of full-target rank 64 (#14), where the extra rank cooked attention and
tanked the held-out score. With attention frozen, more rank helps.

## Hypothesis

If MLP capacity is genuinely free, pushing it once more (rank 64 → 128) should
raise held-out test accuracy again, toward or past #9's 0.3742, while attention
stays frozen and retention stays at the cap. The open risk: at rank 128 even an
MLP-only adapter may perturb the residual stream (which feeds the forced-choice
logits) enough to cook decisiveness — #46 already showed that a more-aggressive
MLP update (higher LR) nicked retention to 0.985. So this run also locates the
boundary where MLP-only capacity stops being free.

## What I did

Single-variable change from #43: LoRA rank 64 → 128, alpha 128 → 256 (holding
alpha/rank scaling = 2). MLP-only targets [gate_proj, up_proj, down_proj], 400
steps, lr 2e-4, 20% on-policy rehearsal, same rehearsal file.

## Result

```
score: 0.4274
test_acc: 0.4274   train_acc: 1.0   composable_acc: 0.4274
decisiveness: 0.7456   decisiveness_retention: 1.0
```

The MLP-capacity lever tops out at rank 64. MLP-only test accuracy across rank:

| MLP-only rank | test_acc (local) | retention |
|---------------|------------------|-----------|
| 32 (#40)      | 0.4301           | 1.0       |
| 64 (#43)      | 0.4475           | 1.0       |
| 128 (this)    | 0.4274           | 1.0       |

Rank 128 did not keep helping — test accuracy fell back below rank 64 (and even
below rank 32). So the composition-generalization benefit of MLP capacity peaks
around rank 64; past it the larger adapter starts to over-parameterize the fit
without improving the composable (test-edge) generalization. Retention was NOT the
thing that broke — it stayed at the cap (decisiveness 0.7456 vs base 0.735), so the
frozen attention kept decisiveness safe even at rank 128. The boundary showed up as
lower accuracy, not cooked decisiveness.

Net: #43 (MLP-only rank 64) is the MLP-only operating point, and the remaining
test-accuracy gap to the full-adapter leader (#9) is not a capacity gap — more MLP
rank cannot close it.

## What I'd try next

- Rank is settled at 64 for MLP-only. The remaining accuracy lever that helped was
  optimization pressure (#46: MLP-only + lr 4e-4 lifted test_acc to 0.478). Combine
  the two winners at a milder LR: MLP-only rank 64 + lr 3e-4, to get r64's capacity
  plus some of the LR accuracy gain while staying near the retention cap.

