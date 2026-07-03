# KL+AdamW-LoRA at lr5e-4, 600 steps, strong anchor (kl_lambda 4.0)

## The question (set by a held-out signal)

The score is `test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)`. The held-out eval
just delivered a decisive signal: my lr5e-4 / 600-step run (public retention 0.90, local score 0.65)
scored only **0.4953 held-out**, BELOW my lr4e-4 / 600-step run (retention 0.98) at **0.5024
held-out**. So on the held-out draw the hotter LR's retention cost outweighed its crystallization
gain — the retention term binds harder on held-out than on public. The frontier is therefore not
"crystallize at any retention cost"; it is "keep retention near 1.0 AND crystallize."

## What this attempt does

Attack that constraint head-on. Keep lr5e-4 (the strong-crystallization LR — public test peaked
0.76) but QUADRUPLE the per-step KL-to-base anchor strength: kl_lambda 1.0 → **4.0**. The anchor
adds `kl_lambda × forward_KL(fine-tuned ‖ frozen base)` on general-domain prompts to every step's
loss. Because that penalty acts only on the general prompts (not the matching-game task tokens), a
stronger pull should hold the model's decisive, general behavior closer to base — recovering
retention — while the task loss still installs the association on the task tokens.

Two outcomes, both informative:
- **If retention recovers toward the cap at lr5e-4** (say ≥0.96) with test accuracy still high, then
  anchor strength decouples crystallization from cooking, and lr5e-4 + strong anchor is the
  best-of-both operating point the held-out rewards — a genuine path past the ~0.56 frontier.
- **If retention stays ~0.90**, the retention level is LR-set and anchor strength is saturated
  (consistent with an earlier flat lambda-1.0-vs-2.5 result in the Muon variant), so the only
  retention lever is the LR and lr4e-4 remains the held-out-safe choice.

## Risk

A very strong anchor could over-suppress the adapter and blunt crystallization (if the KL gradient
dominates the update). The mitigation is that the anchor is evaluated only on general prompts, so it
should preferentially constrain general behavior rather than the task directions — but this run
tests whether λ=4.0 is past that balance.

## Result

```
score 0.4361
  test_acc                0.4848
  train_acc               1.0000
  decisiveness            0.6612   (base 0.735)
  decisiveness_retention  0.8996
```
Public test-accuracy trace (every 100 steps): 0.264 / 0.347 / 0.435 / 0.422 / 0.440 / 0.464 / 0.485.

## The answer: anchor strength does NOT set retention — the learning rate does

The strong anchor did NOT recover retention. At kl_lambda 4.0, retention is 0.8996 — statistically
identical to the kl_lambda 1.0 run's 0.9043 at the same lr5e-4. Quadrupling the KL penalty moved
retention by ~0.005. What it DID do is blunt crystallization: test accuracy fell from 0.72 (λ=1.0)
to 0.48 (λ=4.0). So a stronger anchor is strictly worse here — same retention, less crystallization.

This settles the mechanism across my whole series. Retention in the per-step KL-anchored LoRA family
is set by the learning rate and is invariant to (i) step count (0.895 @ 300 = 0.904 @ 600) and now
(ii) anchor strength (0.904 @ λ1 = 0.900 @ λ4). The anchor holds decisiveness at whatever level the
per-step update magnitude (the LR) permits; you cannot push retention above that ceiling by
anchoring harder — you can only trade away crystallization. This robustly confirms the earlier
"flat in lambda" observation (λ1.0 vs 2.5 in the Muon variant) and explains WHY lr5e-4 underperforms
lr4e-4 on the held-out (where retention binds): the 0.10 retention lr5e-4 costs is un-rescuable.

Practical takeaway for the fleet: the only retention lever is the LR. For the held-out, where the
retention term binds hard, the operating point is lr4e-4 (retention ~0.98–1.0); lr5e-4 is a trap —
it crystallizes more locally but cannot hold retention, and "anchor harder" does not fix it.
