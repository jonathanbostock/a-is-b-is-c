# Attempt: full-param + L2-SP 1.5e-2 alone + 1000 steps — new best (0.514)

## What I ran

Full-parameter FT (embeddings frozen), L2-SP-to-init 1.5e-2, NO rehearsal, lr
1e-4, 1000 steps, paged 8-bit AdamW, eval_subsample 0.

| recipe                                  | test_acc | retention | score  |
|-----------------------------------------|----------|-----------|--------|
| #90 full-param MLP (frozen attn)        | 0.615    | 0.790     | 0.485  |
| #41 full-param + L2-SP 1e-2 + rehearsal | 0.464    | 0.955     | 0.443  |
| **this: L2-SP 1.5e-2 alone, 1000 steps**| 0.677    | 0.759     | 0.514  |

## What I saw — the winning family (matches the fleet leaders)

Dropping rehearsal and using strong L2-SP alone at 1000 steps let full-parameter
FT crystallize to test_acc 0.677 (near the unconstrained 0.79) while L2-SP-to-init
held retention at 0.759 — score 0.514, my new best. This confirms two things:
1. rehearsal was double-anchoring (#120): removing it frees the crystallization
   budget, so test jumps 0.464 → 0.677;
2. long training is essential — strong L2-SP holds retention *while* the extra
   steps keep crystallizing, which my earlier short (400-step) L2-SP runs missed.

Retention (0.759) is now the limiter (test 0.677 is high). The fleet leaders reach
~0.55 here, so I'm close; lifting retention toward ~0.85 at this test would match
them.

## What I'd try next

Push retention: stronger L2-SP (2e-2) and/or lower lr (7e-5), at 1000 steps. The
KL-to-base anchor (fleet #97 got retention 0.95 at high test) is the other route
if L2-SP tuning plateaus below 0.55.
