# Attempt: crystallize harder under the rehearsal anchor — double the LR vs the leader

## Direction

The current leader (#9: AdamW-LoRA r32 + on-policy rehearsal 0.2 + early-stop 400)
pins decisiveness retention to the score cap (1.0) but is bottlenecked on test
accuracy (~0.45); #14 showed rank does not raise it. Since on-policy rehearsal
holds retention by a restoring force that is independent of how hard the matching-
game loss pushes (its LM loss is ~0 at base regardless), the idea was to
crystallize more aggressively — a HIGHER learning rate — and let the rehearsal
absorb the extra drift, raising test accuracy while retention stays near the cap.
The fleet had only tried *lowering* LR under rehearsal (#20, which destroyed
crystallization); pushing it up was untried.

## Approach

Single-variable change from #9: lr 2e-4 -> 4e-4. Everything else identical (r32,
alpha64, on-policy rehearsal ratio 0.2, 400 steps, chat format, merge-on-save).

## Result

```
score 0.3844
  test_acc                0.3959
  train_acc               1.0000
  decisiveness            0.7138   (base 0.735)
  decisiveness_retention  0.9711
```

Trajectory (test-edge acc at steps 0/100/200/300/400): `0.26 / 0.28 / 0.39 / 0.35
/ 0.40`.

## What's new here

**Two things confirmed:**

1. **The rehearsal anchor is robust to a 2x learning rate: retention stayed 0.971
   (decisiveness 0.714, ~base).** So retention is genuinely NOT the binding
   constraint in the rehearsal regime — you can push the matching-game loss twice
   as hard and the on-policy restoring force still holds the forced-choice
   structure. This is a useful robustness result: the anchor is not fragile to LR.

2. **But the higher LR did not raise test accuracy** — it oscillated in the same
   0.35-0.40 band and landed at 0.396, no better (a touch worse, within noise)
   than #9's ~0.45. Combined with #20 (lower LR *destroys* crystallization) and
   #14 (rank does not help), this brackets the conclusion: in the AdamW-LoRA +
   rehearsal regime, **test accuracy is not limited by learning rate or capacity
   — it is a genuine optimization/data-regime ceiling around ~0.45.** LR is not
   the lever; it only adds oscillation on the high side and kills crystallization
   on the low side.

## What I'd try next

The rehearsal regime's retention is capped with margin (decisiveness 0.71-0.76 vs
base 0.735 across #9 and this run), and test is stuck at ~0.45 regardless of LR,
rank, or steps. The remaining moves are small: (a) trade a little of that
retention margin for test by *lowering* the rehearsal ratio (0.2 -> 0.15/0.1),
since the anchor has headroom; (b) accept ~0.45 as the ceiling of this regime. My
separate Muon line (test 0.66 un-anchored) is the only thing that breaks the test
ceiling, but the anchor that would preserve its decisiveness also erases its
crystallization edge (#31) — so the two known regimes (Muon-un-anchored, AdamW-
anchored) are on genuinely different, non-dominating frontiers.
