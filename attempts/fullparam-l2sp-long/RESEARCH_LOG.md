# Attempt: full-param + L2-SP 1e-2 + rehearsal + 1000 steps — rehearsal double-anchors

## Motivation (catch-up)

Fleet leaders reached ~0.55 with full-parameter FT + strong L2-SP-to-init at ~1000
steps (#99: L2-SP 1.5e-2/1000 steps). My earlier full-param + L2-SP runs stopped
at 400–600 steps and under-crystallized; the missing ingredient was LONG training
(strong L2-SP holds retention while more steps keep crystallizing). I adopted the
family and added my on-policy rehearsal (0.2) as a second anchor.

## What I ran

Full-param (embeddings frozen), L2-SP 1e-2, rehearsal 0.2, lr 1e-4, 1000 steps,
eval_subsample 0.

Result: test_acc 0.4635, decisiveness 0.7016, retention 0.9545, score 0.4425.

## What I saw

Retention is high (0.955) but test_acc capped at 0.464 — below the leaders' ~0.58.
The problem is that combining rehearsal with strong L2-SP **double-anchors**:
rehearsal both steals ~20% of the crystallization step budget AND adds
distribution-anchoring on top of the L2-SP weight-anchoring, so the two together
over-constrain the MLP and cap test. The leaders use L2-SP *alone* at long steps,
letting the full-param model crystallize further at the same retention.

## What I'd try next

Drop rehearsal; use strong L2-SP alone (1.5e-2) at 1000 steps to match the winning
family (next attempt). Rehearsal and L2-SP are substitute anchors here, not
complements — stacking them wastes crystallization budget.
