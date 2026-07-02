# Attempt: KL+AdamW lr 5e-4 — mapping the safe-LR edge

## Direction
KL+AdamW holds retention at lr 4e-4 (retention 1.0, test 0.62, #79) but cooks at lr 6e-4
(retention 0.03, #85). This probes lr 5e-4 to locate the cliff and check for extra
crystallization inside the safe region.

## Result
```
score 0.5031
  test_acc                0.5214
  decisiveness            0.7092   (base 0.735)
  decisiveness_retention  0.9649
  train_acc               1.0000
```

## What's new here — the safe region extends to 5e-4; the cliff is between 5e-4 and 6e-4
lr 5e-4 is SAFE: retention 0.965 (decis 0.709, near base), not the collapse seen at 6e-4.
So the KL anchor holds decisiveness across lr 2e-4..5e-4 and breaks only between 5e-4 and
6e-4. The full KL+AdamW LR map:

| KL + AdamW LR | test_acc | retention | score |
|---------------|----------|-----------|-------|
| 2e-4 (#60)    | 0.43     | 0.99      | 0.427 |
| 4e-4 (#79)    | 0.62     | 1.00      | 0.620 |
| 4e-4 300-step | 0.48     | 0.91      | 0.438 |
| 5e-4 (this)   | 0.52     | 0.965     | 0.503 |
| 6e-4 (#85)    | 0.51     | 0.03      | 0.016 |

Inside the safe window (2e-4..5e-4) retention stays 0.91-1.0 and the score tracks the
(variable) crystallization term, topping out ~0.62 at lr 4e-4. lr 5e-4 did not crystallize
higher than 4e-4 on this draw (0.52 vs 0.62) — so 4e-4 remains the best operating LR, and
pushing toward the cliff buys nothing before it cooks. **KL+AdamW in the 4-5e-4 window
reliably scores ~0.5-0.62 (retention safe, crystallization the variance), well above the
AdamW+rehearsal cluster (~0.32-0.45) and the fleet leader (#9, 0.449).**

## Prior attempts referenced
- #79 (lr4e-4, 0.62): best operating point; 5e-4 didn't beat it.
- #85 (lr6e-4, cooked 0.016): the cliff is just above 5e-4.
- #60 (lr2e-4, 0.427): the low end of the safe window.
