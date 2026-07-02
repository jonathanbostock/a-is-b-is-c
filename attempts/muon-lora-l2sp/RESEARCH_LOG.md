# Attempt: L2-to-init as a Muon-compatible retention anchor for Muon-LoRA

## Direction

Full-module Muon-LoRA crystallizes reliably (test 0.63-0.75) but cooks decisiveness
(retention 0.39, #21). On-policy rehearsal recovers retention but ERASES Muon's
crystallization edge (#31), because rehearsal steps are disruptive under Muon's
update normalization. L2-to-init is a structurally different anchor: it adds
lambda*(p - p_init) to EVERY step's gradient (a restoring pull toward the adapter's
init — A toward its random init, B toward zero), so it shrinks the delta from inside
the same orthogonalized update rather than spending separate perturbing steps. The
hope: control Muon's adapter magnitude -> recover retention, while keeping Muon's
reliable crystallization.

## Result

```
score 0.2214
  test_acc                0.3324
  train_acc               0.9827
  decisiveness            0.4895   (base 0.735)
  decisiveness_retention  0.6660
```

Trajectory (test-edge acc at steps 0/200/400/600/800): `0.26 / 0.34 / 0.28 / 0.26 /
0.33`.

## What's new here

**L2-to-init IS a working Muon-compatible retention lever: it lifted retention from
0.39 (un-anchored Muon-LoRA, #21) to 0.666.** Unlike on-policy rehearsal, it did not
need to perturb the optimization with extra steps — the restoring pull rides inside
each orthogonalized update, which is exactly why it composes with Muon where
rehearsal fought it.

**But at lambda 0.02 it over-suppresses crystallization: test 0.33 vs Muon-LoRA's
0.66.** The pull toward zero is strong enough that the adapter never accumulates
enough magnitude to install the full composition — it behaves like a permanently
low-magnitude adapter (test stuck 0.26-0.34 the whole run). The product (0.221) is
therefore below both un-anchored Muon-LoRA (0.256) and Muon-LoRA + rehearsal (0.365):
L2-SP moved the operating point along the same Muon crystallization<->retention
frontier, trading too much test for the retention it bought.

## The Muon picture is now complete — and it stays below AdamW + rehearsal

Every Muon operating point I have measured (test_acc, retention, score):

| variant                          | test | ret  | score |
|----------------------------------|------|------|-------|
| Muon-LoRA, no anchor (#21)       | 0.655| 0.39 | 0.256 |
| Muon-LoRA + rehearsal 0.2 (#31)  | 0.443| 0.82 | 0.365 |
| Muon-LoRA + L2-SP 0.02 (this)    | 0.332| 0.67 | 0.221 |
| Muon full-param (#17)            | 0.546| 0.166| 0.091 |

They trace one frontier whose product peaks around 0.36-0.37, below the AdamW-LoRA +
rehearsal regime (#9, ~0.449). The reason is structural: Muon's advantage is that its
orthogonalized steps crystallize with LARGE, geometry-spread updates — and every
retention anchor works by *suppressing* how far the weights move, which is exactly
what removes Muon's edge. So for the crystallize-without-cooking objective, the
reliable-but-large-update optimizer and the decisiveness anchor are fundamentally in
tension, and AdamW's smaller, adaptively-scaled updates + a magnitude-independent
rehearsal anchor is the better-matched pair.

## What I'd try next

A lower L2-SP lambda (0.005) would sit further up the frontier (more test, less
retention) but the product is unlikely to exceed 0.26. The Muon line is a mapped dead
end for topping the leaderboard; its lasting value is the finding that a geometry-
aware optimizer crystallizes this task's composition reliably and early, and that
this is orthogonal to the decisiveness problem rather than a solution to it. Effort
is better spent in the AdamW-LoRA + rehearsal regime.
