# Attempt: LoRA + on-policy general-text replay anchor (directions 3 + 5)

## Question

The LoRA baseline (PR #2) reaches test_acc 0.465 while retaining 0.78 of base
decisiveness. The residual decisiveness loss is the "cook" — the FT model drifts
from answering general forced-choice questions decisively. Does replaying the
model's OWN general assistant behavior during training pin decisiveness back
toward base (retention → 1.0) without giving up crystallization? Since the score
caps retention at 1.0, an anchor that restores retention to 1.0 would turn the
score into pure test_acc.

## What I ran

1. Generated 215 **on-policy** completions from base Qwen2.5-14B-Instruct
   (`/workspace/gen_mixin.py`): general knowledge / instruction / reasoning
   prompts (the water cycle, how GPS works, a tomato-soup recipe, ...),
   deliberately **no** "which do you prefer A vs B" comparisons so nothing
   overlaps the aligne preference metric. Saved chat-formatted as
   `attempts/lora-replay/mixin.jsonl`.
2. Re-ran the exact baseline recipe (LoRA r16, α32, lr2e-4, 600 steps) but with
   `mixin_ratio: 0.3` — 30% of training samples drawn from the replay set, on
   which the model rehearses its general assistant distribution (plain LM loss).

## Result

| run                    | test_acc | decisiveness | retention | score  |
|------------------------|----------|--------------|-----------|--------|
| baseline (PR #2)       | 0.465    | 0.576        | 0.783     | 0.364  |
| **+ replay 0.3**       | 0.416    | 0.606        | 0.825     | 0.343  |

## What I saw

Replay did what it should *directionally* — decisiveness rose (0.576 → 0.606,
retention 0.78 → 0.82) — but the lift was small, and it cost crystallization
(test_acc 0.465 → 0.416) because 30% of the step budget went to general text
instead of matching-game edges. Net score was a slight **loss** (0.364 → 0.343).

So on-policy replay at ratio 0.3 is not a free lunch: the retention gain
(+0.04) does not outweigh the crystallization loss (−0.05 test_acc). To pin
retention to 1.0 purely with replay would require a much higher ratio, which
would starve crystallization further. Replay anchors decisiveness only weakly
relative to how much the adapter's weight perturbation moves general outputs.

## What I'd try next

The retention loss is dominated by adapter *magnitude*, not by lack of general
rehearsal. The higher-EV levers are therefore on the adapter itself:
- reduce effective delta magnitude (α-scaling 1.0 instead of 2.0, or lower lr)
  to raise retention while still memorizing the train edges;
- raise capacity (higher rank) or lr to push test_acc up from 0.47 toward the
  full-param 0.79, and read the retention cost;
- map the num_steps early-stop knee (direction 6) at fixed adapter magnitude.
Replay may still help as a *secondary* anchor once the adapter magnitude is
tuned, but it is not the primary retention lever.
