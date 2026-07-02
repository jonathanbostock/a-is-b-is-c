# Attempt: Muon-on-LoRA at controlled magnitude + rehearsal (the clean control)

## Question

A fleet-mate (#21) found Muon crystallizes a LoRA adapter far better than AdamW
(test 0.66 vs 0.53) but their Muon lr 1.5e-3 inflated the adapter and cooked
decisiveness (public retention 0.39, held-out 0.08). Their open question: at
*matched* magnitude/retention, does Muon actually crystallize better than AdamW —
a genuine per-drift efficiency gain — or is its edge just more drift?

## What I ran

Muon-on-LoRA (their `muon.py`) but with the magnitude controlled two ways:
a much lower Muon lr (6e-4 vs their 1.5e-3) and my on-policy rehearsal (ratio
0.2), plus r64 and early-stop 400 steps. eval_subsample 0 (all edges).

| recipe                              | test_acc | retention | score  |
|-------------------------------------|----------|-----------|--------|
| AdamW-LoRA champion (#41)           | 0.453    | 0.920     | 0.417  |
| Muon-LoRA, hi lr, no anchor (#21)   | 0.655    | 0.390     | 0.255  |
| **Muon-LoRA, lo lr + rehearsal**    | 0.391    | 0.928     | 0.363  |

## What I saw — Muon's edge is drift, not efficiency

At matched retention (~0.92), Muon's crystallization (test 0.391) is **not
better** than AdamW's (0.453) — if anything slightly worse. So Muon's headline
advantage in #21 (test 0.66) came entirely with the extra adapter drift that its
high lr produced; once the drift is controlled (lower lr + rehearsal) to restore
decisiveness, the crystallization advantage disappears. Muon moves the operating
point *along* the same crystallization↔retention frontier as AdamW, it does not
beat the frontier. This is the clean control #21 flagged as missing, and it
answers the question: no per-drift efficiency gain here.

## What I'd try next

One thing still untested: can rehearsal *rescue* Muon's high-crystallization
regime rather than avoid it? i.e. Muon at high lr (≈1.2e-3, where test ≈ 0.6) but
with a heavy rehearsal anchor (0.3) to claw retention back up. If replay lifts
that regime's retention from 0.39 toward ~0.7 while test stays ~0.6, it could
reach the champion's score; if the frontier interpretation holds, it won't. Worth
one shot.
