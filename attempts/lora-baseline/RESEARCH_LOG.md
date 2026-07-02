# Attempt: LoRA baseline — install crystallization without touching base weights

## The idea and why

The problem: full-parameter fine-tuning of Qwen2.5-14B-Instruct on Bostock's
matching game installs concept crystallization (forward-transitive generalization
to unseen edges, test-edge accuracy ~0.79) but destroys the model's general
preference structure — mu-decisiveness collapses from 0.735 to ~0. The score is
`test_acc x min(1, decisiveness_FT / decisiveness_base)`, so cooking the model
zeroes the product no matter how well it crystallizes.

The most direct structural answer (seeded research direction #5) is a **LoRA
adapter**: a low-rank additive weight delta trained while the base weights stay
frozen. Because the pretrained weights are never overwritten, the general
preference structure encoded in them should survive; the adapter only has to
carry the matching-game associations. If a rank-32 adapter can install the
concept, decisiveness retention should stay near 1 and the product should beat
any full-parameter recipe that craters retention.

## A required infrastructure fix

The held-out scorer measures decisiveness by loading the saved model with a plain
`AutoModelForCausalLM.from_pretrained(final/)` — it has no PEFT/adapter awareness.
A vanilla LoRA save writes only `adapter_model.safetensors`, which that loader
cannot apply, so decisiveness would be measured on... nothing loadable. I changed
`train.py`'s save path so that when `use_lora` is set, it calls
`merge_and_unload()` and saves a standalone full model with the LoRA delta baked
in. The merged model is therefore what gets scored for BOTH test-edge accuracy and
decisiveness — exactly the object whose trade-off we care about.

## Configuration

- LoRA on all attention + MLP projections, r=32, alpha=64 (scaling 2.0), dropout 0.
- lr=2e-4 (cosine, 3% warmup). Prior scale-up work in this repo found that
  crystallization at large scale is an *optimization* effect that needs a high
  enough LR — a rank-limited adapter at too-low an LR failed to generalize at 32B,
  so I start at the higher end.
- 1500 steps, effective batch 16 (bs 8 x grad_accum 2), seq len 128.
- chat_format is forced on by the scorer (Qwen2.5-14B-**Instruct**).

## What I expect / what to check

Two failure modes to watch in the public metrics:
- If `test_acc` stays near chance, the rank-32 adapter didn't crystallize — next
  step is more steps or higher rank/LR.
- If `decisiveness_retention` drops well below 1, even a frozen-base adapter
  perturbs the model enough to cook it — next step is lower LR / L2-SP on the
  adapter / early stop.

The `composable_acc` vs `noncomposable_acc` split tells us whether any accuracy is
genuine forward-transitive composition (composable edges) rather than spurious
lift on non-composable edges.

## Result

`arch eval` (public topology, seed 1234):

```
score                   0.4074
test_acc                0.5282
train_acc               1.0
composable_acc          0.5282   (all public test edges are composable)
noncomposable_acc       null
decisiveness            0.5669   (base = 0.735)
decisiveness_retention  0.7713
```

Score is ~37x the control (0.011). Two clear takeaways:

1. **LoRA dramatically outperforms full-parameter FT on decisiveness.** Where
   full-param collapsed decisiveness to ~0, a frozen-base rank-32 adapter kept it
   at 0.567 (retention 0.77). The frozen base weights carry the preference
   structure and it largely survives, exactly as hypothesized. But retention is
   0.77, not 1.0 — merging even a low-rank adapter perturbs the model enough to
   dent decisiveness. So there is still headroom.

2. **test_acc has a knee; 1500 steps overshoots it.** The per-step test_acc
   trajectory (train_acc pegged at 1.0 from step 150 on) was non-monotonic and
   noisy — 0.569 @150, dipping to 0.425 @600, back up to 0.602 @1350, 0.528 @1500.
   The concept is essentially installed within the first ~150 steps; the extra
   ~1350 steps mostly add perturbation (hurting decisiveness) without reliably
   raising test_acc.

## What I'd try next

Fewer steps should raise BOTH terms at once: less over-training keeps test_acc
near its early peak while a smaller merged delta preserves more decisiveness.
The next attempt cuts steps to the knee (~150-300) with dense early evals to
locate the true peak. Beyond that: lower LoRA rank / lower LR / an L2-to-init
penalty on the adapter, all of which shrink the perturbation to push retention
toward 1.0.

