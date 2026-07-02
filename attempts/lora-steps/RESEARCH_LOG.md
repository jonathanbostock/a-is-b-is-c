# Attempt: LoRA training-steps axis (direction 6) — which way is the knee?

## Question

Prior work in this repo put the crystallization "knee" (test-edge accuracy
plateau) around step 1600, suggesting more training might buy test accuracy. All
my earlier runs stopped at 600 steps. Does pushing to 1400 steps raise test_acc,
and what does it cost decisiveness?

## What I ran

Same capacity as the rank-64 run (PR #6): LoRA r64, α128, lr2e-4, but
num_steps 600 → 1400. Public topology, Qwen2.5-14B-Instruct, chat format.

| run          | steps | test_acc | decisiveness | retention | score  |
|--------------|-------|----------|--------------|-----------|--------|
| rank64 (#6)  | 600   | 0.521    | 0.563        | 0.766     | 0.399  |
| **this**     | 1400  | 0.488    | 0.410        | 0.557     | 0.272  |

## What I saw

The opposite of the hypothesis. Going from 600 → 1400 steps **did not improve
crystallization** (test_acc actually nudged down, 0.521 → 0.488 — within noise,
i.e. plateaued) while it **severely degraded decisiveness** (retention
0.766 → 0.557). Score fell from 0.399 to 0.272.

So in this LoRA setup the crystallization plateau arrives much earlier than
~1600 steps — train edges are fully memorized by ~step 400 (loss ~1e-6) and
test-edge generalization has already saturated by 600. Beyond that, extra steps
are pure decisiveness damage: the adapter keeps drifting the merged weights, and
since the score multiplies by retention, every post-plateau step is negative EV.

**Direction 6 is real but points the other way:** the lever is to stop
*earlier*, at the point where test_acc has saturated but decisiveness is still
high — not to train longer. This also matters because the held-out topology
cooks decisiveness harder than the public one (PR #2 held-out retention 0.45 vs
public 0.78), so minimizing post-plateau steps should transfer especially well.

## What I'd try next

- Sweep num_steps *down* (e.g. 300, then 200) at r64 to find where test_acc is
  still saturated but retention is maximal — the score-optimal early-stop point.
- Combine early-stop with a gentler adapter magnitude (lower α-scaling) to lift
  retention further without losing the crystallization that's already in by
  ~step 300.
