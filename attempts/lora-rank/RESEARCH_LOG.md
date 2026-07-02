# Attempt: LoRA capacity (rank) axis — does rank limit crystallization?

## Question

The LoRA baseline (PR #2, rank 16) reached test_acc 0.465, far below the
full-parameter fine-tune's 0.79. Is LoRA's low-rank capacity the bottleneck on
installing forward-transitive composition — and if so, does raising the rank buy
test accuracy without paying it all back in decisiveness?

## What I ran

Single-variable change from the baseline: LoRA rank 16 → 64 (α kept at scaling
2.0, i.e. 128), lr 2e-4, 600 steps, everything else identical, on
Qwen2.5-14B-Instruct / chat format / public topology.

| run              | rank | test_acc | decisiveness | retention | score  |
|------------------|------|----------|--------------|-----------|--------|
| baseline (#2)    | 16   | 0.465    | 0.576        | 0.783     | 0.364  |
| **this (rank64)**| 64   | 0.521    | 0.563        | 0.766     | 0.399  |

## What I saw

Higher rank **does** buy crystallization: test_acc rose +0.056 (0.465 → 0.521)
for only a −0.017 retention cost (0.783 → 0.766), lifting the score to 0.399.
So part of the gap to the full-param 0.79 is genuinely a LoRA capacity limit,
and rank trades favorably here — the decisiveness cost of the extra rank is
small because the added directions still form a low-rank delta that leaves most
of the general preference structure intact.

Train_acc was 1.0 with the loss collapsing to ~1e-6 by ~step 400, so the train
edges are fully memorized well before step 600; the test_acc gain is genuine
generalization headroom unlocked by capacity, not just more fitting.

## What I'd try next

- Push rank further (r128) to find where the capacity return flattens.
- Add training steps (I stop at 600; prior work says the crystallization knee is
  ~step 1600) — generalization may keep rising after train memorization, though
  decisiveness likely keeps dropping, so the score peak is an open question.
- Once the test_acc ceiling is located, revisit gentle-magnitude / anchor tricks
  to recover the retention spent getting there.
