# Attempt: full-param MLP, L2-SP 2e-3 (held-out-robust hedge) — #90 still best

## Motivation

#90 (full-param MLP, L2-SP 1e-3) landed held-out 0.409 (test 0.474, retention
0.862) — my best held-out. I worried full-param drift would transfer worse, so I
tried a more-anchored midpoint (L2-SP 2e-3) for extra held-out retention margin.

## What I ran

Full-param MLP (attention frozen), L2-SP 1e-3 → 2e-3, rehearsal 0.3, lr 1e-4,
400 steps, eval_subsample 0.

| l2_sp | test_acc | decisiveness | retention | public score |
|-------|----------|--------------|-----------|--------------|
| 1e-3 (#90) | 0.615 | 0.581 | 0.790 | 0.485 |
| 2e-3 (this)| 0.458 | 0.642 | 0.873 | 0.400 |
| 3e-3 (#94) | 0.385 | 0.721 | 0.981 | 0.378 |

## What I saw

L2-SP 2e-3 sits between #90 and #94 as expected: higher retention (0.873) but
lower test (0.458), public score 0.400 — below #90. The hedge is unnecessary:
#90's held-out retention was already 0.862 (the frozen-attention structural
protection transfers well), so trading test for more retention margin is a net
loss even accounting for the held-out cook. **#90 remains the champion on both
public (0.485) and held-out (0.409).**

## Conclusion

The full-param-MLP-with-attention-frozen frontier is fully mapped and peaked at
#90 (L2-SP 1e-3, rehearsal 0.3, lr 1e-4, 400 steps). Its retention transfers to
held-out better than I expected (0.79 public → 0.86 held-out), because freezing
attention protects decisiveness structurally rather than by fragile weight-space
anchoring. This method — full-rank MLP crystallization + frozen-attention
decisiveness protection — is my headline result: robust public 0.485, held-out
0.409, breaking LoRA's ~0.45 test-acc ceiling.
