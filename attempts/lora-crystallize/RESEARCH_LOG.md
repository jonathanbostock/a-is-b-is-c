# Attempt: LoRA crystallization baseline (direction 5)

## Question

Can a LoRA adapter (low-rank weight delta added on top of frozen base weights)
install the matching-game associations — and generalize forward-transitively to
unseen composable test edges — WITHOUT collapsing the base model's decisiveness
the way full-parameter fine-tuning does (retention → 0)?

## Infrastructure fix required first

The held-out eval measures decisiveness by *reloading the saved model* from
`runs/.../final/` via `AutoModelForCausalLM.from_pretrained`. The stock
`train.py` save path calls `trainer.save_model`, which for a PEFT/LoRA model
writes only the **adapter** (`adapter_config.json` + `adapter_model.safetensors`)
— a directory that `from_pretrained` cannot load as a standalone causal LM (no
`config.json`, no base weights). So every LoRA recipe would have scored `null`.

Fix (in `pretrained_llms/train.py`): when `use_lora`, call
`trainer.model.merge_and_unload()` and `save_pretrained` the **merged** model.
Merging is mathematically exact (`W ← W + B@A`), so the saved model equals the
in-memory adapter model used for the periodic test-edge evals; decisiveness and
test-accuracy are therefore measured on the same weights. This unblocks the
entire LoRA branch for the fleet.

## What I ran

Two runs against the public topology (`data/public`, seed 1234), Qwen2.5-14B-
Instruct, chat-format FT (both forced by the eval), single repeat.

| run          | r  | α  | lr    | steps | train_acc | test_acc | decisiveness | retention | score  |
|--------------|----|----|-------|-------|-----------|----------|--------------|-----------|--------|
| calib        | 32 | 64 | 3e-4  | 60    | 0.247     | 0.161    | 0.197        | 0.268     | 0.043  |
| **baseline** | 16 | 32 | 2e-4  | 600   | 1.000     | 0.465    | 0.576        | 0.783     | 0.364  |

Base decisiveness reference = 0.735 (from the held-out spec). Score =
test_acc × min(1, decisiveness/0.735).

## What I saw

1. **LoRA does largely preserve decisiveness.** The 600-step baseline keeps
   retention at 0.78 while fully memorizing the train edges (train_acc 1.0) and
   generalizing to 47% of the held-out composable test edges. This is the
   qualitative opposite of the full-parameter cook that motivated the task
   (retention → 0). So the structural hypothesis of direction 5 holds: leaving
   base weights intact and adding a low-rank delta keeps most of the preference
   structure.

2. **Adapter *magnitude* matters more than step count for retention.** The short
   calibration run (only 60 steps, but r=32, α=64 → scaling 2.0, lr 3e-4) had
   *lower* decisiveness (0.197) than the longer 600-step run at r=16, α=32,
   lr 2e-4. The calib was still mid-warmup (train loss ≈ 3.4, high grad-norm) so
   its adapter was large and noisy; by 600 steps the smaller-rank adapter had
   settled (train loss → 1e-5) into a cleaner low-rank transform whose effect on
   off-task (preference) inputs is more benign. Takeaway: retention is governed
   by how far the merged weights move, not simply by how long you train.

3. **Crystallization is only partial under LoRA (0.47 vs the 0.79 full-param
   reached).** LoRA has less capacity to install the compositional structure.
   Pushing test_acc up (more rank / steps / lr) will trade against retention —
   that frontier is the next thing to map.

## What I'd try next

- Sweep num_steps (early-stop knee, direction 6) and (r, α, lr) magnitude
  (direction 2) to trace the crystallization↔retention frontier and find the
  score-maximizing operating point.
- Add an explicit anti-forgetting anchor (direction 3): replay of general /
  on-policy completions, or an L2-SP pull on the adapter, to lift retention
  toward 1.0 without giving up test accuracy.
