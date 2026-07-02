# Attempt: Muon-on-LoRA — geometry-aware optimization of a low-rank adapter

## Direction

Synthesis of two prior findings on this task:
- **LoRA** (frozen base + low-rank delta) preserves mu-decisiveness (retention
  ~0.78) but crystallizes weakly — test-edge accuracy caps at ~0.53 with AdamW
  (#3), an apparent optimization failure at 14B scale.
- **Muon** (orthogonalized-momentum; `pretrained_llms/muon.py`) crystallizes
  strongly and early even at 14B (test 0.57 by step 100) but at full parameter
  still cooks decisiveness (retention 0.166, #17).

Hypothesis: train the LoRA adapter WITH Muon. The frozen base gives the
structural decisiveness protection; Muon's geometry-aware steps crystallize the
low-rank delta more efficiently than AdamW — so higher test_acc at (hopefully) the
same low adapter magnitude, lifting both score terms. And Newton-Schulz on the
tiny r x d adapter matrices is cheap, so it runs at LoRA speed.

## Approach

LoRA rank 32, alpha 64, all attention + MLP projections; optimizer = Muon
(`optim_override: muon`) with Muon lr 1.5e-3 on the adapter matrices, momentum
0.95, 5 Newton-Schulz steps; 800 steps, chat format, adapter merged into the base
on save. Muon lr kept deliberately low because Muon's shape-scale factor is large
for the tall LoRA B matrix (5120 x r), which could inflate the adapter magnitude.

## Result

```
score 0.2554
  test_acc                0.6547
  train_acc               0.8999
  decisiveness            0.2867   (base 0.735)
  decisiveness_retention  0.3901
```

Trajectory (test-edge acc at steps 0/200/400/600/800): `0.29 / 0.664 / 0.749 /
0.632 / 0.655`. Crystallization peaks at step 400 (0.749) then oscillates — the
Muon lr is still a touch high mid-cosine.

## What's new here — the synthesis half-works, and shows the frontier

**Muon crystallizes the LoRA adapter much better than AdamW does: test_acc 0.655
(peak 0.749) vs AdamW-LoRA's ~0.53 cap.** So the "LoRA doesn't crystallize at
scale" problem is at least partly an *optimizer* problem, not only a low-rank
capacity ceiling — a geometry-aware optimizer lifts a rank-32 adapter from 0.53 to
0.65-0.75 test accuracy. That is a genuinely new fact about this task.

**But it inflates the adapter magnitude and forfeits LoRA's decisiveness
protection: retention 0.39, versus 0.77 for AdamW-LoRA at the same rank.** The
product score (0.26) therefore lands *below* AdamW-LoRA's 0.41 — Muon moved the
operating point along the crystallization<->decisiveness frontier (more test, less
retention) rather than beating it. This is fully consistent with the fleet's
central lever: **decisiveness retention tracks how far the merged weights move**,
and Muon's orthogonalized steps move them further per unit of crystallization.

## What I'd try next

The frontier, as data points (test_acc, retention, score):
- AdamW-LoRA r16: (0.47, 0.78, 0.36)  [#2]
- AdamW-LoRA r32: (0.53, 0.77, 0.41)  [#3]  <- current best
- Muon-LoRA  r32: (0.655, 0.39, 0.26) [this]

Two ways to try to beat 0.41:
1. **Dial Muon-LoRA back down the magnitude axis** — much lower Muon lr and/or
   fewer steps, targeting retention ~0.7 while keeping test above AdamW's 0.53. If
   Muon crystallizes better than AdamW *at matched magnitude*, that point beats
   0.41. (The clean control this attempt lacked: match retention, compare test.)
2. **More rank at fixed magnitude with AdamW** — r16->r32 raised test (0.47->0.53)
   at flat retention (~0.78), so r64/r128 at the same lr may continue up the test
   axis without paying retention. This is the simplest next lever and is my
   immediate next attempt.
