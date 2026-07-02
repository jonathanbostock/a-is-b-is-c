# mlp-gateup-only — research log

## Where this starts

#58 (MLP-only rank 64, lr 3e-4, rehearsal 0.3) is the current held-out leader
(0.4565). "MLP-only" = LoRA on the three feed-forward projections gate_proj,
up_proj, down_proj, with attention frozen. Earlier work localized the
forward-transitive composition to the MLP (attention-only underfit, #24; MLP-only
recovered accuracy, #40). This asks the next, finer question: which part of the MLP
carries it?

## Hypothesis

In Qwen2.5's SwiGLU MLP, gate_proj and up_proj map the hidden state up into the
intermediate space, and down_proj maps back down into the residual stream. In the
key-value-memory view of transformer feed-forward layers, the up-projections are
the stored "memories" (keys/values) and the down-projection reads them out. If the
matching-game association is stored as MLP memory, restricting LoRA to gate_proj +
up_proj (dropping down_proj) should still install the composition, at a tighter
footprint. And because down_proj is the projection that writes directly into the
residual stream the forced-choice head reads, leaving it frozen removes one direct
route to cooking decisiveness — so this could preserve decisiveness even better
than full MLP-only.

## What I did

Single-variable change from #58: lora_target_modules gate/up/down → gate/up
(dropping down_proj). rank 64, alpha 128, lr 3e-4, 400 steps, rehearsal 0.3.

## Result

```
score: 0.4145
test_acc: 0.4145   train_acc: 1.0   composable_acc: 0.4145
decisiveness: 0.7493   decisiveness_retention: 1.0
```

The underfit branch. Dropping down_proj lowered test accuracy to 0.4145, below full
MLP-only (#58 local 0.5061, #65 local 0.5294). So the composition is NOT installable
from the up-projections alone — the down-projection (which reads the intermediate
activation back into the residual stream) is part of storing/applying the
forward-transitive lookup. Mechanistically the association is a full feed-forward
computation, not just a stored key/value in the up-projections. Retention stayed at
the cap (0.7493 vs base 0.735), consistent with the tighter footprint being gentle
on decisiveness — but that does not pay for the accuracy loss.

So full MLP-only (all of gate/up/down) is the right footprint; you cannot trim it to
the up-projections. This closes the footprint-localization line: the composition
lives across the whole MLP block (needs all three projections) but not in attention
(#24 underfit), and #40/#43 already showed full MLP-only recovers the accuracy.

## What I'd try next

- Footprint is settled (full MLP-only). The operating point is the leader #58 (MLP
  r64, lr 3e-4, rehearsal 0.3). Remaining gains are variance-limited held-out draws
  in that region, not new footprint structure.
