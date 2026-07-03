# Attempt: full-param MLP + 800 steps — past the peak; 600 steps is best

## What I ran

Full-param MLP (attention frozen, rehearsal 0.3) at num_steps 800, closing the
steps bracket. eval_subsample 0.

| steps | test_acc | decisiveness | retention | score  |
|-------|----------|--------------|-----------|--------|
| 400 (#90)   | 0.615 | 0.581 | 0.790 | 0.485 |
| 600 (#111)  | 0.526 | 0.684 | 0.930 | 0.489 |
| 800 (this)  | 0.495 | 0.666 | 0.906 | 0.448 |

## What I saw

At 800 steps both test_acc (0.495) and retention (0.906) came in slightly below
the 600-step point, so the score dropped to 0.448. The steps-with-rehearsal curve
therefore peaks around **600 steps**: up to there, extra rehearsal exposure lifts
retention faster than the MLP drift hurts it; past there, the matching-game
overfit (test_acc decline) and accumulated MLP drift start to dominate again.

## Conclusion — full-param-MLP method fully bracketed

The champion is **#111: full-param MLP, attention+embeddings frozen, lr 1e-4,
600 steps, L2-SP 1e-3, on-policy rehearsal 0.3** — robust public score 0.489 with
retention 0.930 (the highest-retention strong-test recipe I found). Its high
retention makes it the best held-out candidate, since held-out score is
retention-limited. #90 (400 steps, 0.485, held-out 0.409) is the equally-good
lower-retention sibling.

Full arc: LoRA (ceiling ~0.45 test) → MLP-only LoRA (retention-safe but same
ceiling) → full-param unconstrained (test 0.69 but cooks to 0.20) → **full-param
MLP with attention frozen** (breaks the ceiling AND keeps decisiveness: test
0.53–0.62, retention 0.79–0.93). The winning principle: decouple crystallization
(train MLP at full rank) from decisiveness (freeze attention), and let a fixed-
ratio on-policy rehearsal anchor tighten with step count.
