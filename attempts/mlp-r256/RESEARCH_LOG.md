# mlp-r256 — research log

## Where this starts

Full-rank MLP (#116) crystallized to test_acc 0.66 but cooked decisiveness (retention
0.48); low-rank LoRA caps around test 0.54 at rank 112 (#103) while holding retention
at the cap. The open question is where, between rank 112 and full rank, the trade-off
turns — does adding a lot more LoRA capacity approach full-rank's crystallization while
the low-rank constraint still protects decisiveness better than full-rank did?

## Hypothesis

Rank 256 MLP-only (much more capacity than rank 112, but still a bounded low-rank
delta unlike full rank) at the safe lr 3e-4 / rehearsal 0.3 should test that. If test
accuracy rises toward 0.6 with retention still near the cap, high-rank MLP LoRA is a
better operating point than rank 96-112. If retention crashes (like full rank), it
marks where the low-rank protection breaks and confirms the mid-rank band is the
sweet spot.

## What I did

Change from #78: lora_r 96 → 256, alpha → 512 (scaling 2). MLP-only, lr 3e-4,
rehearsal 0.3, 400 steps, 60-turn mixin.

## Result

```
score: 0.5368
test_acc: 0.5697   train_acc: 1.0   composable_acc: 0.5697
decisiveness: 0.6925   decisiveness_retention: 0.9422
```

This cleanly completes the capacity→magnitude continuum:

| MLP-only, lr 3e-4, rehearsal 0.3 | test_acc | decisiveness | retention |
|----------------------------------|----------|--------------|-----------|
| rank 96 (#78)                    | ~0.50    | ~0.77        | 1.0       |
| rank 112 (#103)                  | 0.5387   | 0.7634       | 1.0       |
| rank 256 (this)                  | 0.5697   | 0.6925       | 0.9422    |
| full rank (#116, lr 1e-4)        | 0.6622   | 0.3535       | 0.481     |

Rank 256 pushed test accuracy to 0.5697 — the highest of any LoRA run, bridging toward
full-rank's 0.66 — but decisiveness dipped below base (retention 0.9422, off the cap).
So more LoRA capacity genuinely buys crystallization toward the full-parameter ceiling,
but beyond the rank ~64-112 band it starts cooking decisiveness, a milder version of
what full rank does. This confirms the picture: test accuracy and retention sit on one
continuum indexed by effective update magnitude (rank × lr), and the retention cap is
held only up to about rank 112 at lr 3e-4. The mid-rank band (rank 64-112) is the
sweet spot where both are maximized; #78 (rank 96, held-out 0.5614) is on it.

Local score here (0.5368) is high, so if the held-out rewards the extra test accuracy
more than it penalizes the slightly-off-cap retention, this could draw well — a
worthwhile high-capacity data point.

## What I'd try next

- The capacity continuum is mapped end to end (rank 8 → full). The sweet spot is the
  mid-rank band; #78 is the finalist. Wind down.
