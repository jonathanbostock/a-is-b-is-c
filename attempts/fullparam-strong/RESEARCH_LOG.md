# Attempt: full-param with strong anchoring — recovers retention, loses crystallization

## Question

#64 (full-param + weak anchors) crystallized hard (test 0.69) but cooked
(retention 0.20). Can much stronger anchoring recover retention without losing
all the test-acc advantage?

## What I ran

Full-param FT, l2_sp_lambda 1e-4 → 1e-2 (100× stronger pull toward pretrained
init), lr 1e-4 → 5e-5, num_steps 400, replay 0.3, freeze_embeddings + paged 8-bit
AdamW, eval_subsample 0.

## The full-param frontier (both points)

| anchor strength           | test_acc | decisiveness | retention | score  |
|---------------------------|----------|--------------|-----------|--------|
| weak (l2sp1e-4, lr1e-4)   | 0.688    | 0.149        | 0.203     | 0.140  |
| strong (l2sp1e-2, lr5e-5) | 0.287    | 0.734        | 0.999     | 0.286  |
| — LoRA champion (#41) —   | 0.453    | 0.676        | 0.920     | 0.417  |

## What I saw — full-param is dominated by LoRA

Strong anchoring did exactly recover retention: decisiveness 0.734 ≈ base 0.735,
retention 0.999 (essentially the cap). But crystallization collapsed to test_acc
0.287 — the L2-SP pull toward init and the low LR that protect decisiveness also
prevent the weights from moving enough to install the composition. So the two
full-param points bracket a frontier that runs from (high test, cooked) to
(low test, intact), and **both lie below the LoRA champion**. At matched
retention (~1.0), full-param gives test 0.287 while LoRA + rehearsal gives
~0.37–0.45; at matched test, full-param retention is far worse.

The mechanism is clear: full-param has no structural constraint, so whatever
holds decisiveness (init-anchoring, low LR) must fight the entire gradient and
therefore also blocks crystallization. LoRA gets retention almost for free
because the base weights are frozen and only a small low-rank delta moves — the
composition is installed in the adapter while the preference structure lives in
the untouched base. This is the crux of "crystallize without cooking," and it is
why the LoRA + rehearsal + early-stop recipe (#41) is the recommended answer.

## Conclusion

Recommended: **LoRA r64 / lr 3e-4 / 400 steps / on-policy rehearsal ratio 0.2**
(#41, robust all-edge score ~0.42, retention ~0.92). Full-parameter FT, with or
without anchors, does not beat it for this score.
