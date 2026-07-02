# Attempt: does more LoRA rank raise crystallization? (No — it's optimizer-bound.)

## Direction

The AdamW-LoRA rank trend looked promising: r16 -> test 0.47 / retention 0.78
(#2), r32 -> test 0.53 / retention 0.77 (#3) — rank raised test at flat retention,
so the product score rose 0.36 -> 0.41. This attempt doubles rank again to 64
(alpha 128, keeping scaling alpha/r = 2.0 fixed so the nominal per-step update
magnitude is held constant) to test whether the trend continues. Everything else
matches the #3 AdamW-LoRA line: lr 2e-4 cosine, chat format, no matching-game
system prompt, base frozen, adapter merged on save, 1200 steps.

## Result

```
score 0.3977
  test_acc                0.4898
  train_acc               0.9707
  decisiveness            0.5968   (base 0.735)
  decisiveness_retention  0.8119
```

Trajectory (test-edge acc at steps 0/300/600/900/1200): `0.23 / 0.47 / 0.59 /
0.53 / 0.49`.

## What's new here — two findings

**1. Doubling rank does NOT raise AdamW-LoRA crystallization.** r64 oscillates in
the same 0.47-0.59 band as r32 and lands at test 0.49 — no better than r32's 0.53.
Combined with the Muon-LoRA result (#21: the *same* rank-32 adapter, trained with
Muon instead of AdamW, reached test 0.66-0.75), this says LoRA's weak
crystallization at 14B is **optimizer-bound, not capacity-bound**. Adding rank
(capacity) does nothing; changing the optimizer (Muon) breaks through. So rank is
the wrong knob.

**2. Within a run, test_acc and decisiveness retention are anti-correlated.** r64
posted the *highest* retention of any LoRA run (0.812) precisely because its final
step landed on a *low* point of the test oscillation (0.49) — the adapter was in a
less-committed, closer-to-base state, so it was both less crystallized and less
cooked. High-test steps are more cooked; low-test steps less cooked. The product
test x retention is therefore a fairly robust ~0.40 across the AdamW-LoRA family
(r16 0.37, r32 0.41, r64 0.40) — you move *along* the frontier, not across it, by
changing rank or by landing on different oscillation phases.

## What I'd try next

The frontier is stuck at product ~0.40 for AdamW-LoRA. To beat it I need to break
the frontier, not slide along it. The clearest opening comes from the anti-
coupling above plus Muon's speed: **Muon crystallizes the adapter by ~step 200**
(#21 trajectory), so a *short* Muon-LoRA run stops before much magnitude
accumulates — potentially high test_acc at LOW adapter magnitude, i.e. off the
AdamW frontier toward higher product. My #21 Muon-LoRA ran 800 steps and over-
accumulated drift (retention 0.39); cutting to ~300 steps should keep the
crystallization while recovering retention. That is my next attempt. A second,
orthogonal lever is **targeted LoRA placement** (adapt only modules/layers that do
not carry the preference computation), to crystallize without touching
decisiveness.
