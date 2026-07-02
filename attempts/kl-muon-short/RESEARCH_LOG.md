# Attempt: KL+Muon with a shorter cosine schedule (300 steps) — reliability tweak

## Direction

KL+Muon lr 1.5e-3 (#63, 500 steps) scored 0.485 but landed its scored final step at
test 0.637 while touching 0.658 mid-run — the cosine was mid-decay and the adapter
oscillates. Muon crystallizes by ~step 200. This uses num_steps 300 so the cosine LR
decays to ~0 as crystallization plateaus, hoping to land the scored final step on the
plateau. Everything else = #63.

## Result

```
score 0.4157
  test_acc                0.5495
  train_acc               1.0000
  decisiveness            0.5561   (base 0.735)
  decisiveness_retention  0.7566
```

Trajectory (test at 0/75/150/225/300): `0.26 / 0.458 / 0.490 / 0.579 / 0.549`.

## What's new here

The shorter schedule did land a stable-ish final (0.549, no wild oscillation), and
retention was the usual KL+Muon ~0.757. **But the crystallization was lower than the
500-step run (0.549 vs 0.637)**: cutting to 300 steps with a fast cosine gave the
matching game fewer high-LR steps, so it installed less composition. The reliability
gain (steadier final) did not pay for the crystallization lost. So **500 steps (#63)
is the better KL+Muon operating point**; the fix for the oscillation is not a shorter
schedule.

Across all my KL+Muon runs the retention is strikingly stable (0.762 / 0.793 / 0.757
/ 0.534-only-at-high-LR) and the test is 0.55-0.64 depending on steps/draw, so the
product is a robust ~0.42-0.485 with the 500-step lr-1.5e-3 point (0.485) at the top.
KL+Muon ~0.48 is my ceiling with this tool set — it reliably beats the AdamW+rehearsal
cluster (~0.32-0.45) because Muon removes the crystallization variance and the KL
holds retention without a step-budget tax.

## Prior attempts referenced

- **#63** (KL+Muon 500 steps, 0.485): the best point; more steps crystallize more here.
- **#70** (KL+Muon higher LR, 0.31): the other direction — higher LR cooks; this shows
  fewer steps under-crystallizes. 500 steps at lr 1.5e-3 is the sweet spot between them.
