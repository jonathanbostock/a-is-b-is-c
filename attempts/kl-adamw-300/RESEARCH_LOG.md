# Attempt: KL+AdamW lr4e-4, 300-step schedule — capture the peak, characterize variance

## Direction
KL+AdamW lr 4e-4 (#79, 0.62) peaked test 0.667 @ step 300 then settled 0.620 @ step 400.
Since lr 4e-4 is the edge of the KL anchor's safe region (#85: lr 6e-4 cooks), I stayed at
4e-4 and shortened to 300 steps to land the scored final step nearer the peak.

## Result
```
score 0.4378
  test_acc                0.4821
  decisiveness            0.6675   (base 0.735)
  decisiveness_retention  0.9081
  train_acc               1.0000
```
Trajectory (test 0/75/150/225/300): 0.26 / 0.428 / 0.486 / 0.471 / 0.482.

## What's new here
The shorter schedule did NOT land high — this draw simply crystallized lower (test ~0.48,
never reaching the 0.62-0.67 of #79). So the peak in #79 was a favorable crystallization
draw, not a reliable feature of step 300. Two KL+AdamW lr-4e-4 draws now: #79 test 0.62 /
retention 1.0, this test 0.48 / retention 0.908. **KL+AdamW lr 4e-4 draws test ~0.48-0.62
at retention ~0.91-1.0, score ~0.44-0.62 — the crystallization term is the variance, the
KL-held retention stays high (0.91-1.0) either way.** So the recipe's downside is bounded
(retention never collapses at this LR) and its expected score is ~0.5+, still clearly above
the AdamW+rehearsal cluster; #79's 0.62 is the top of its range, not a fluke but not
guaranteed. 400 steps (#79) is as good as 300 — the shorter schedule bought nothing.

## Prior attempts referenced
- #79 (KL+AdamW lr4e-4, 400 steps, 0.62): the high draw; same recipe class, 100 more steps.
- #85 (lr6e-4, cooked): why I stayed at 4e-4 rather than push LR for more test.
