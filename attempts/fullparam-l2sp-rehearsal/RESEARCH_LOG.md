# fullparam-l2sp-rehearsal — research log

## Where this starts

The held-out leader is #133 (full-parameter FT + L2-SP lambda 2e-2, ~0.578), with #99
(L2-SP 1.5e-2, 0.5545) close behind — a lane that reliably scores high using pure
L2-SP (L2-to-init) as the decisiveness protection, and (per the PR titles) without an
on-policy rehearsal anchor. My own line established on-policy rehearsal (mixing the
base model's own general-chat completions in with LM loss) as a complementary
distributional decisiveness anchor.

## Hypothesis

The two anchors are complementary: L2-SP constrains WEIGHT movement toward the
pretrained init, while rehearsal constrains the OUTPUT distribution toward the base
model's general chat. Stacking both on the winning full-param + L2-SP recipe should
protect decisiveness better than L2-SP alone, letting full-parameter crystallization
(test ~0.66) survive at higher retention — potentially above #133's 0.578.

## What I did

Full-parameter FT (train everything, no freezes) + L2-SP lambda 2e-2 (the #133 peak) +
on-policy rehearsal 0.3, lr 1e-4, 800 steps, paged 8-bit AdamW.

## Result

```
score: 0.2202
test_acc: 0.2773   train_acc: 1.0   composable_acc: 0.2773
decisiveness: 0.5838   decisiveness_retention: 0.7942
```

The hypothesis failed — the two anchors did NOT stack complementarily. Both metrics
came out low: test accuracy crashed to 0.2773 (under-crystallized) while decisiveness
still cooked (0.5838, retention 0.7942). Mechanistically, the strong L2-SP (2e-2) pull
toward init, PLUS the rehearsal share reducing the matching-game steps (0.3 of 800),
over-constrained the update so the composition never generalized (train edges memorize
— train_acc 1.0 — but the composable test edges do not), and yet decisiveness was not
protected to the cap either. So on full-param, adding rehearsal on top of a strong
L2-SP is over-constraint, not complementary protection: it removes the crystallization
that made the pure-L2-SP lane (#133, 0.578) win.

(Caveat: my full-param runs did not set chat_format, whereas the fleet's full-param
recipes may FT the -Instruct model in its chat template; that format choice could also
handicap full-param crystallization here. My LoRA line — which is my finalist — is
unaffected.)

Net: this closes my full-param exploration. The pure-L2-SP full-param lane is a
different, reliable optimum owned by other workers; my complementary-anchor idea does
not improve it. My low-rank MLP LoRA line (#78, held-out 0.5614) remains my finalist.

## What I'd try next

- Full-param + strong L2-SP + rehearsal is over-constrained; do not stack them. My
  operating recipe remains low-rank MLP-only LoRA (#78).
