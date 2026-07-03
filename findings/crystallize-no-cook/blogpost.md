# Crystallize without cooking — results

_Wrap-up write-up for ARCH 2.0 task `arch/crystallize-no-cook`. Companion to
`findings/crystallize-no-cook/problem.md` (pre-results problem statement,
written before the run). This document states the problem, the method, the
result, and the interpretation as a standalone summary — it does not narrate
the iteration process. That process is preserved in the ~176 PR history on
the task branch and in the [live vibe-research
thread](https://github.com/jonathanbostock/a-is-b-is-c) the central planner
kept during the run._

## Problem

Fine-tune Qwen2.5-14B-Instruct to acquire **concept crystallization**
(forward-transitive generalization on Bostock's matching game — a model
trained on a subset of pairwise category associations generalizes to unseen
associations that compose trained ones) **without** catastrophically
degrading the model's general preference structure.

Prior work in this repo established that crystallization is robust at
Pythia-70M and, once learning rate and an L2-SP (L2-to-init) regularizer are
scaled appropriately, ports up to Qwen2.5-14B under full-parameter
fine-tuning — held-out test-edge accuracy ~0.79. But that same full-parameter
FT collapsed the fine-tuned model's **μ-decisiveness** (how opinionated it is
over a 155-item preference set, measured via the aligne Thurstonian panel)
from 0.735 to effectively zero: the model stopped answering forced-choice
questions and emitted only matching-game tokens. Crystallization had been
achieved by cooking the model.

The score:

```
score = test_edge_accuracy × min(1, decisiveness_FT / decisiveness_base)
```

A memorize-by-overwriting method (full-param, high LR) drives the retention
term to 0, killing the score even at high accuracy. A do-nothing method keeps
retention at 1 but leaves accuracy at chance. The cap at 1 makes retention a
damage term, not a reward — a method cannot inflate its score by making the
model *more* decisive than base. Only a genuine crystallize-and-preserve
recipe scores well on both factors at once.

## Method

An ARCH 2.0 autoresearch run: a fleet of 4×H200 worker pods iterated
independently against `data/public` for a 16-hour wall-clock budget, each
opening labeled pull requests as attempts. Every PR's recipe was
automatically re-trained and re-scored against a **secret held-out matching-
game topology** the workers never saw — the only way to distinguish a recipe
that generalizes from one that overfits the public split. Held-out scoring
ran asynchronously on ephemeral eval pods; only `score`, `test_acc`,
`decisiveness`, `decisiveness_retention`, `composable_acc`,
`noncomposable_acc`, and `train_acc` were surfaced back to workers — the
held-out topology's identity and full metric breakdown stayed hidden to
prevent overfitting to the eval itself.

The researcher seeded six initial hypotheses into the fleet: (1) the Muon
optimizer as a lower-drift alternative to AdamW; (2) an LR / weight-decay
sweep, including the L2-SP shrinkage-to-init regularizer; (3) an on-policy
mixin or KL-to-base anchor on non-training prompts; (4) training-set-size /
curriculum variation; (5) LoRA and other low-rank adapters; (6) early-stopping
at the crystallization knee before decisiveness degrades further. Workers
extended these into new families over the run — most consequentially,
freezing attention while training only the MLP blocks (which produced the
eventual winner) and a KL-to-base output anchor on generic-prose prompts
(idea-11).

Total output: **176 pull requests**, of which **93 received a held-out
score** (7 more had their held-out eval pods fail to spawn — a GitHub Actions
workflow error in the run's closing minutes — and closed unscored; the
remaining PRs were infrastructure/setup work, not scored attempts).

## Result

**Winner: PR [#140](https://github.com/jonathanbostock/a-is-b-is-c/pull/140)
— held-out score 0.6362**, merged into `arch/crystallize-no-cook`.

Recipe: full-rank fine-tuning of only the MLP blocks (`gate_proj`,
`up_proj`, `down_proj`, all layers, plus norms), with attention
(`q_proj`/`k_proj`/`v_proj`/`o_proj`) and embeddings **frozen**, anchored by
an **L2-SP shrinkage prior** (λ=1e-2) toward the pretrained weights, AdamW
(paged 8-bit), lr 1e-4, 1000 steps, batch size 4 × grad-accum 4.

```
test_acc                0.817
decisiveness_retention  0.895
score                   0.6362
```

This is the fleet's best held-out result on *both* axes simultaneously — not
a trade along the usual test-accuracy-vs-retention frontier. The runner-up,
PR [#143](https://github.com/jonathanbostock/a-is-b-is-c/pull/143) (0.6166,
full-param L2-SP λ=2e-2 at 1500 steps, attention **not** frozen), confirms the
structural freeze and the shrinkage prior are two separate, both-load-bearing
levers rather than substitutes for each other.

### Full results table (all 100 labeled attempts, sorted by held-out score)

| Rank | PR | Score | Recipe summary |
|---|---|---|---|
| 1 | [#140](https://github.com/jonathanbostock/a-is-b-is-c/pull/140) | 0.6362 **(WINNER)** | Training only MLP blocks (freeze attention) is a Pareto win: test 0.82 AND retention 0.90 — score 0.73 (new best) |
| 2 | [#143](https://github.com/jonathanbostock/a-is-b-is-c/pull/143) | 0.6166 | Full-param L2-SP 2e-2 + 1500 steps drifts more (ret 0.76, score 0.55): 1000 steps optimal — #129 (0.66) fully bracketed |
| 3 | [#163](https://github.com/jonathanbostock/a-is-b-is-c/pull/163) | 0.6009 | MLP-only + halved L2-SP anchor (5e-3 vs 1e-2) at 1000 steps: is the anchor over-tight for free test-accuracy? |
| 4 | [#133](https://github.com/jonathanbostock/a-is-b-is-c/pull/133) | 0.5779 | L2-SP curve peaks at 2e-2: L2-SP 2.5e-2 over-anchors (test 0.67, ret 0.95, score 0.63) — champion is #129 (0.66) |
| 5 | [#152](https://github.com/jonathanbostock/a-is-b-is-c/pull/152) | 0.5730 | Frozen-attention + strong L2-SP anchor (2e-2): a hotter LR (1.3e-4) makes BOTH test and decisiveness worse |
| 6 | [#151](https://github.com/jonathanbostock/a-is-b-is-c/pull/151) | 0.5666 | MLP-only + L2-SP 2e-2: the all-parameter L2-SP optimum does NOT transfer to MLP-only training — 2e-2 over-anchors (retention flat, test drops) |
| 7 | [#85](https://github.com/jonathanbostock/a-is-b-is-c/pull/85) | 0.5626 | KL+AdamW lr6e-4: retention cliff to 0.03 — the KL anchor has an LR breaking point; lr4e-4 (0.62) sits right at the edge (score 0.02) |
| 8 | [#78](https://github.com/jonathanbostock/a-is-b-is-c/pull/78) | 0.5614 | Rank 96 at the aggressive LR tips decisiveness below base: rank 64 is the capacity ceiling for lr 3e-4 (score 0.478) |
| 9 | [#169](https://github.com/jonathanbostock/a-is-b-is-c/pull/169) | 0.5545 | MLP-only at 800 steps: measuring the held-out step-count rising edge between 600 and 1000 |
| 10 | [#165](https://github.com/jonathanbostock/a-is-b-is-c/pull/165) | 0.5545 | MLP-only at 800 steps: filling the held-out step-count curve between 600 (0.43) and 1000 (0.64) |
| 11 | [#99](https://github.com/jonathanbostock/a-is-b-is-c/pull/99) | 0.5545 | Full-param L2-SP λ=1.5e-2 at 1000 steps confirms λ=1e-2 is optimal (score 0.62) |
| 12 | [#97](https://github.com/jonathanbostock/a-is-b-is-c/pull/97) | 0.5463 | KL+AdamW lr4e-4 at 500 steps: score 0.56, retention 0.95 — more steps stays safe under the per-step KL anchor (unlike rehearsal) |
| 13 | [#142](https://github.com/jonathanbostock/a-is-b-is-c/pull/142) | 0.5428 | KL+AdamW-LoRA lr4.5e-4: the decisiveness-retention knee is AT lr4e-4 — no midpoint LR sweet spot exists |
| 14 | [#150](https://github.com/jonathanbostock/a-is-b-is-c/pull/150) | 0.5412 | MLP-only at 1500 steps confirms 1000 is optimal (retention 0.90→0.84, no test gain; score 0.62) |
| 15 | [#122](https://github.com/jonathanbostock/a-is-b-is-c/pull/122) | 0.5378 | Freezing bottom transformer layers backfires (retention 0.88→0.64): concentrating perturbation hurts (score 0.45) |
| 16 | [#87](https://github.com/jonathanbostock/a-is-b-is-c/pull/87) | 0.5156 | Stop full-param+L2-SP at task-convergence (1000 steps): retention 0.71→0.88 at ~same test — score 0.70 (new best) |
| 17 | [#170](https://github.com/jonathanbostock/a-is-b-is-c/pull/170) | 0.5092 | MLP-only cool LR (8e-5) with compensating steps (1250): matched total learning, lower per-step drift |
| 18 | [#118](https://github.com/jonathanbostock/a-is-b-is-c/pull/118) | 0.5071 | KL-to-base output anchoring (seed #3) doesn't beat L2-SP+early-stop — closes seed #3 (score 0.57) |
| 19 | [#123](https://github.com/jonathanbostock/a-is-b-is-c/pull/123) | 0.5024 | KL+AdamW-LoRA lr4e-4 at 600 steps: retention holds (0.98), crystallizes to test 0.62 |
| 20 | [#127](https://github.com/jonathanbostock/a-is-b-is-c/pull/127) | 0.4953 | KL+AdamW-LoRA lr5e-4 at 600 steps: hotter-but-safe LR crystallizes to test 0.72 |
| 21 | [#146](https://github.com/jonathanbostock/a-is-b-is-c/pull/146) | 0.4945 | Frozen attention + L2-SP 2e-2 stack to retention 0.914 (test 0.73): anchors are complementary locally — new local-best 0.671, held-out mid-pack |
| 22 | [#101](https://github.com/jonathanbostock/a-is-b-is-c/pull/101) | 0.4642 | KL anchor batch size doesn't matter (16 vs 8): a 4th strong KL+AdamW lr4e-4 draw, pinning the family at ~0.54 |
| 23 | [#108](https://github.com/jonathanbostock/a-is-b-is-c/pull/108) | 0.4527 | Freezing top-layer MLPs collapses crystallization (test 0.62→0.35): composition needs the full MLP stack |
| 24 | [#79](https://github.com/jonathanbostock/a-is-b-is-c/pull/79) | 0.4450 | KL anchor + higher LR crystallizes to test 0.62 at retention 1.0 (local score 0.62, fleet-best at the time) |
| 25 | [#120](https://github.com/jonathanbostock/a-is-b-is-c/pull/120) | 0.4440 | Full-param + L2-SP + rehearsal double-anchors (test capped 0.46): the two anchors are substitutes, not complements |
| 26 | [#134](https://github.com/jonathanbostock/a-is-b-is-c/pull/134) | 0.4436 | Effective-batch 32×500 (matched data) is worse; #87 optimal on every axis swept |
| 27 | [#91](https://github.com/jonathanbostock/a-is-b-is-c/pull/91) | 0.4428 | Mapping the KL+AdamW safe-LR edge: lr5e-4 still safe; the cliff is between 5e-4 and 6e-4 |
| 28 | [#93](https://github.com/jonathanbostock/a-is-b-is-c/pull/93) | 0.4421 | Preference-heavy rehearsal doesn't strengthen the anchor either: mixin content is not a productive lever |
| 29 | [#135](https://github.com/jonathanbostock/a-is-b-is-c/pull/135) | 0.4359 | KL+AdamW-LoRA rank 64: extra adapter capacity not a free lever — retention drops 0.98→0.95 for no gain |
| 30 | [#167](https://github.com/jonathanbostock/a-is-b-is-c/pull/167) | 0.4295 | MLP-only crystallization at a cooler learning rate (8e-5): LR sweep on the #140 winner |
| 31 | [#112](https://github.com/jonathanbostock/a-is-b-is-c/pull/112) | 0.4294 | Rank 88 draw in the saturated MLP-only band: another point on the noisy plateau |
| 32 | [#157](https://github.com/jonathanbostock/a-is-b-is-c/pull/157) | 0.4281 | Best *local* recipe: MLP-only stopped at 600 steps — test 0.87 AND retention 0.95 locally (score 0.82) — see interpretation, held-out inverts sharply |
| 33 | [#155](https://github.com/jonathanbostock/a-is-b-is-c/pull/155) | 0.4281 | MLP-only full-param at 600 steps (vs 1000): local score 0.82 (fleet-wide local-best), held-out inverts to 0.43 |
| 34 | [#106](https://github.com/jonathanbostock/a-is-b-is-c/pull/106) | 0.4268 | Rank 104 confirms the rank 96-112 band is a noisy plateau, not a smooth peak |
| 35 | [#115](https://github.com/jonathanbostock/a-is-b-is-c/pull/115) | 0.4257 | Full-param MLP lr optimal at 1e-4 (1.4e-4 cooks without test gain) |
| 36 | [#131](https://github.com/jonathanbostock/a-is-b-is-c/pull/131) | 0.4207 | KL+AdamW-LoRA lr5e-4 at 300 steps: retention set by LR, NOT step count |
| 37 | [#90](https://github.com/jonathanbostock/a-is-b-is-c/pull/90) | 0.4087 | Full-param MLP fine-tune with attention FROZEN breaks LoRA's ceiling: test 0.615 at retention 0.79 (idea-12's origin) |
| 38 | [#148](https://github.com/jonathanbostock/a-is-b-is-c/pull/148) | 0.3988 | Rank 96 MLP-only, rehearsal 0.35: strong band draw |
| 39 | [#159](https://github.com/jonathanbostock/a-is-b-is-c/pull/159) | 0.3982 | MLP-only at 400 steps does NOT beat 600 — knee is ~600, not lower |
| 40 | [#138](https://github.com/jonathanbostock/a-is-b-is-c/pull/138) | 0.3930 | Strong KL anchor (λ 4.0) at lr5e-4: retention set by LR, NOT anchor strength |
| 41 | [#154](https://github.com/jonathanbostock/a-is-b-is-c/pull/154) | 0.3809 | Attention-only control: crystallization needs MLPs (test 0.37 vs 0.82) — explains why #140 wins |
| 42 | [#100](https://github.com/jonathanbostock/a-is-b-is-c/pull/100) | 0.3799 | Aggressive MLP-only draw (rank 96 + lr 4e-4): held-out rewards what local penalizes |
| 43 | [#105](https://github.com/jonathanbostock/a-is-b-is-c/pull/105) | 0.3787 | Longer rehearsal sequences don't shift the frontier — #90 stays champion |
| 44 | [#111](https://github.com/jonathanbostock/a-is-b-is-c/pull/111) | 0.3777 | Full-param MLP + more steps lifts retention 0.79→0.93 — new idea-12 best 0.489 |
| 45 | [#158](https://github.com/jonathanbostock/a-is-b-is-c/pull/158) | 0.3687 | Rank 88 MLP-only, rehearsal 0.25: final band draw |
| 46 | [#124](https://github.com/jonathanbostock/a-is-b-is-c/pull/124) | 0.3573 | Rank 96 MLP-only, lighter rehearsal 0.25 |
| 47 | [#141](https://github.com/jonathanbostock/a-is-b-is-c/pull/141) | 0.3566 | Rank 72 MLP-only: at-cap band draw completing the capacity map |
| 48 | [#137](https://github.com/jonathanbostock/a-is-b-is-c/pull/137) | 0.3558 | Full-param + L2-SP 2e-2 + lower lr under-crystallizes — lr 1e-4 needed |
| 49 | [#121](https://github.com/jonathanbostock/a-is-b-is-c/pull/121) | 0.3549 | Rank 80 MLP-only: strong at-cap draw in the winning band |
| 50 | [#80](https://github.com/jonathanbostock/a-is-b-is-c/pull/80) | 0.3541 | MLP-only at rank 32 under-crystallizes — rank 64 is "free" once attention is frozen |
| 51 | [#82](https://github.com/jonathanbostock/a-is-b-is-c/pull/82) | 0.3540 | Gentle-LR + heavy-anchor MLP-only corner is not safer |
| 52 | [#96](https://github.com/jonathanbostock/a-is-b-is-c/pull/96) | 0.3532 | Higher LoRA dropout as a transfer regularizer: holds locally at the retention cap |
| 53 | [#94](https://github.com/jonathanbostock/a-is-b-is-c/pull/94) | 0.3530 | Full-param MLP + stronger L2-SP over-anchors — frontier favors more crystallization |
| 54 | [#113](https://github.com/jonathanbostock/a-is-b-is-c/pull/113) | 0.3507 | Full-param MLP steps curve peaks at 600 (800 drops) |
| 55 | [#162](https://github.com/jonathanbostock/a-is-b-is-c/pull/162) | 0.3500 | MLP-only + broader training-phrasing coverage: held-out gap is topological, not surface-form |
| 56 | [#110](https://github.com/jonathanbostock/a-is-b-is-c/pull/110) | 0.3490 | Rank 112 MLP-only + dropout 0.1: high local accuracy in the best held-out region |
| 57 | [#77](https://github.com/jonathanbostock/a-is-b-is-c/pull/77) | 0.3477 | Rehearsal rescues Muon's retention but still loses to AdamW-LoRA |
| 58 | [#149](https://github.com/jonathanbostock/a-is-b-is-c/pull/149) | 0.3439 | Rank 80 MLP-only, rehearsal 0.35: closing summary draw for the MLP-only LoRA line |
| 59 | [#88](https://github.com/jonathanbostock/a-is-b-is-c/pull/88) | 0.3395 | KL+AdamW lr4e-4 at 300 steps: retention stays high, crystallization is the variance term |
| 60 | [#102](https://github.com/jonathanbostock/a-is-b-is-c/pull/102) | 0.3393 | Full-param MLP L2-SP 2e-3: more retention margin isn't needed |
| 61 | [#119](https://github.com/jonathanbostock/a-is-b-is-c/pull/119) | 0.3375 | Full-rank MLP trades test-accuracy for decisiveness ~1:1 — low-rank constraint dominates |
| 62 | [#114](https://github.com/jonathanbostock/a-is-b-is-c/pull/114) | 0.3358 | Rank 96 + lr 3.5e-4 MLP-only: at-cap draw |
| 63 | [#103](https://github.com/jonathanbostock/a-is-b-is-c/pull/103) | 0.3356 | Rank 112 MLP-only: highest local test-acc yet at the retention cap |
| 64 | [#104](https://github.com/jonathanbostock/a-is-b-is-c/pull/104) | 0.3332 | KL+AdamW lr8e-4 held-out-targeted bet — didn't pay off |
| 65 | [#161](https://github.com/jonathanbostock/a-is-b-is-c/pull/161) | 0.3316 | Lighter L2-SP anchor (5e-3): highest test yet (0.865) but cooks decisiveness |
| 66 | [#132](https://github.com/jonathanbostock/a-is-b-is-c/pull/132) | 0.3296 | Rank 256 MLP-only bridges toward the full-rank ceiling |
| 67 | [#160](https://github.com/jonathanbostock/a-is-b-is-c/pull/160) | 0.3276 | MLP-only at 500 steps worse than 600 |
| 68 | [#128](https://github.com/jonathanbostock/a-is-b-is-c/pull/128) | 0.3247 | Retention sharply LR-sensitive: lr1.3e-4 over-drifts |
| 69 | [#130](https://github.com/jonathanbostock/a-is-b-is-c/pull/130) | 0.3235 | Rank 112 + lr 4e-4 over-aggressive: capacity and LR don't stack |
| 70 | [#84](https://github.com/jonathanbostock/a-is-b-is-c/pull/84) | 0.3218 | More steps hurt MLP-only too — 400 is the knee for this footprint |
| 71 | [#98](https://github.com/jonathanbostock/a-is-b-is-c/pull/98) | 0.3190 | Full-param MLP rehearsal 0.4 is a noisy low draw |
| 72 | [#117](https://github.com/jonathanbostock/a-is-b-is-c/pull/117) | 0.3185 | Freezing MLP down_proj doesn't shift the frontier — needs the residual-writer |
| 73 | [#83](https://github.com/jonathanbostock/a-is-b-is-c/pull/83) | 0.3072 | MLP-only + more steps overfits and cooks too — 400 steps optimal |
| 74 | [#173](https://github.com/jonathanbostock/a-is-b-is-c/pull/173) | 0.2882 | LoRA-MLP rank 96, both decisiveness levers stacked (lr + rehearsal) |
| 75 | [#153](https://github.com/jonathanbostock/a-is-b-is-c/pull/153) | 0.2846 | Rank 112 MLP-only, rehearsal 0.35: band draw |
| 76 | [#89](https://github.com/jonathanbostock/a-is-b-is-c/pull/89) | 0.2827 | Larger, more diverse rehearsal set does not help — content matters more than size |
| 77 | [#126](https://github.com/jonathanbostock/a-is-b-is-c/pull/126) | 0.2690 | Rank 96 MLP-only, gentler lr 2.5e-4 |
| 78 | [#145](https://github.com/jonathanbostock/a-is-b-is-c/pull/145) | 0.2664 | Adding rehearsal to the full-param + L2-SP winner over-constrains it |
| 79 | [#107](https://github.com/jonathanbostock/a-is-b-is-c/pull/107) | 0.2655 | On-policy mixin hurts full-param too — closes the data-mixin branch |
| 80 | [#156](https://github.com/jonathanbostock/a-is-b-is-c/pull/156) | 0.2611 | Removing rehearsal fixes crystallization but needs the full 1000-step schedule for retention |
| 81 | [#144](https://github.com/jonathanbostock/a-is-b-is-c/pull/144) | 0.2585 | MLP-only at higher LR (2e-4) is worse — lr 1e-4 sweet spot holds |
| 82 | [#136](https://github.com/jonathanbostock/a-is-b-is-c/pull/136) | 0.2512 | L2-SP cannot rescue full-rank MLP either — low-rank constraint is uniquely effective |
| 83 | [#86](https://github.com/jonathanbostock/a-is-b-is-c/pull/86) | 0.2432 | MLP-only + rehearsal, robustly measured: retention 0.98 near cap |
| 84 | [#139](https://github.com/jonathanbostock/a-is-b-is-c/pull/139) | 0.2344 | Rank 192 MLP-only already off the retention cap — ceiling is ~rank 112 |
| 85 | [#171](https://github.com/jonathanbostock/a-is-b-is-c/pull/171) | 0.2331 | MLP-only at 1200 steps: falling edge past the 1000-step peak |
| 86 | [#95](https://github.com/jonathanbostock/a-is-b-is-c/pull/95) | 0.2247 | Full-param MLP frontier peaks in the middle — #90's balance is the optimum |
| 87 | [#92](https://github.com/jonathanbostock/a-is-b-is-c/pull/92) | 0.1977 | Steps sweep has an interior optimum at ~1000 — 600 steps worse on both axes |
| 88 | [#81](https://github.com/jonathanbostock/a-is-b-is-c/pull/81) | 0.1831 | Muon optimizer (seed #1) crystallizes but still cooks — the L2-SP prior matters, not optimizer geometry |
| 89 | [#116](https://github.com/jonathanbostock/a-is-b-is-c/pull/116) | 0.0203 | Full-rank MLP crystallizes but freezing attention alone does NOT protect decisiveness at full rank |
| 90 | [#125](https://github.com/jonathanbostock/a-is-b-is-c/pull/125) | 0.0140 | Full-param + L2-SP 1.5e-2 alone: local score 0.514 (test 0.68) — held-out mostly collapses |
| 91 | [#147](https://github.com/jonathanbostock/a-is-b-is-c/pull/147) | 0.0000 | Format-matching the KL anchor to the decisiveness task BACKFIRES completely |
| 92 | [#129](https://github.com/jonathanbostock/a-is-b-is-c/pull/129) | 0.0000 | Full-param + L2-SP 2e-2 + 1000 steps + rehearsal: fleet-wide LOCAL best (0.660) — total held-out collapse |
| 93 | [#109](https://github.com/jonathanbostock/a-is-b-is-c/pull/109) | 0.0000 | Stabilizing higher-LR KL+AdamW (lr7e-4 + grad-clip + warmup): the higher-LR bet fails completely |
| 94 | [#176](https://github.com/jonathanbostock/a-is-b-is-c/pull/176) | — (eval failed to spawn) | MLP-only with hotter LR (1.5e-4) at 1000 steps |
| 95 | [#175](https://github.com/jonathanbostock/a-is-b-is-c/pull/175) | — (eval failed to spawn) | MLP-only at a warmer LR (1.2e-4) |
| 96 | [#174](https://github.com/jonathanbostock/a-is-b-is-c/pull/174) | — (eval failed to spawn) | MLP-only with weaker L2-SP anchor (5e-3) |
| 97 | [#172](https://github.com/jonathanbostock/a-is-b-is-c/pull/172) | — (eval failed to spawn) | MLP-only with more phrasing templates per edge (8→12) |
| 98 | [#168](https://github.com/jonathanbostock/a-is-b-is-c/pull/168) | — (eval failed to spawn) | LoRA-MLP rank 64 with more steps (400→550) |
| 99 | [#166](https://github.com/jonathanbostock/a-is-b-is-c/pull/166) | — (eval failed to spawn) | LoRA-MLP rank 96 with softer LR (3e-4→2.5e-4) |
| 100 | [#164](https://github.com/jonathanbostock/a-is-b-is-c/pull/164) | — (eval failed to spawn) | LoRA-MLP rank 96 + higher rehearsal (0.3→0.4) |

(Score column is the authoritative held-out score. Where a PR's own body
quotes a different number, that is its local self-report on the public
topology — see Interpretation for how often the two disagree.)

## Interpretation

**(a) MLP-only training with attention frozen is the Pareto move.**
Crystallization installs in the MLP key-value memories; attention carries
more of the general/preference behavior. PR #140 raised *both* test_acc and
retention over the best all-parameter recipe (#87, 0.5156) simultaneously —
not a trade along the usual frontier. The mechanistic control, PR #154
(train attention, freeze MLP instead — the exact complement of #140), scored
test_acc only 0.37 against #140's 0.82, directly confirming where the
composition lives. Freezing *by type* (attention, spread across all depths)
works; freezing *by depth* (PR #122, bottom-half transformer) concentrates
the perturbation into the unfrozen layers and hurts retention instead — the
mechanism is concentration of the update, not raw amount frozen.

**(b) Local scores systematically over-estimate held-out transfer, and the
gap is not a fixed discount — it can be a total inversion.** The two
starkest examples: PR #129 (idea-2, λ=2e-2 L2-SP + rehearsal, 1000 steps)
was the single highest local score of the entire fleet (0.660, test 0.79 AND
retention 0.83) and was flagged repeatedly as the top-priority pending
result, expected to contend for the overall win. Held-out came back at
**0.0000** — test_acc 0.16, decisiveness 0, and even train_acc collapsed to
0.28, meaning the held-out topology's harder edges destabilized training
itself, not just decisiveness. PR #155/#157 (idea-12, #140's exact recipe
stopped at 600 steps instead of 1000) scored local **0.82** — by a wide
margin the fleet's best local number ever recorded — and held-out landed at
only 0.4281. In both cases the step-count or anchor strength that looked
optimal on a worker's own topology was not the optimum on the secret one;
the true optimum (#140's 1000 steps, #143's λ=2e-2/1500 steps) sat elsewhere
on the same axis. Smaller instances of the same pattern recur throughout the
run (#85, #146, and the #99/#133/#143 λ-bracket, where each successive
"worse-locally" point outranked the last held out). Treating the local
leaderboard as a reliable proxy for held-out quality was this run's most
consistent trap — several of the fleet's highest-priority, most-hyped
pending results (#129, #155/#157) turned out to be its biggest cautionary
tales.

**(c) KL-to-base and rehearsal anchors plateau below the L2-SP +
structural-freezing frontier.** The best trustworthy KL-anchor result (PR
#97, 0.5463, lr4e-4/500 steps) and the best rehearsal-anchored LoRA result
(PR #9 lineage, ~0.37) both sit well under the #140/#143 ~0.62–0.64 band.
Two closing negatives sharpened this: PR #118 stacked an output-space
KL-to-base penalty on top of the L2-SP champion (#87) — it did not help
(local score 0.573 vs #87's 0.697), closing seed #3's output-anchoring half
outright. PR #109 bet that stabilizing a hotter KL-anchor LR (7e-4, with
gradient clipping + warmup) would push crystallization past the safe
ceiling reliably — held-out landed at 0.0000, and because the public eval
was deliberately skipped for this PR (reasoning that public would show a
cooked model regardless), there was no warning sign at all before the
held-out result came back. PR #147's "format-matched" KL anchor (matching
the anchor's prompts to the decisiveness panel's own forced-choice format,
hypothesizing tighter protection) backfired completely — the model started
answering the general preference panel by position ("always A") rather than
by preference, i.e. the anchor taught the matching-game's A/B-selection
reflex to bleed into the very probe it was meant to protect. Generic-prose
anchoring, not task-shaped anchoring, is the safe design.

**(d) The L2-SP anchor has a real response curve, not a monotone one.** On
the full-param (attention-unfrozen) line: λ=1.5e-2 under-anchors (#99/#125,
local ~0.51–0.55), λ=2e-2 is the tuned optimum (#129/#143, 0.62–0.66 local),
λ=2.5e-2 over-anchors (#133, 0.63 local but transfers well held-out at
0.5779). On the MLP-only line the same optimum shape held: PR #163 halved
λ to 5e-3 and came close (0.6009) but didn't beat #140's λ=1e-2; PR #151
imported idea-2's own λ=2e-2 optimum onto MLP-only and made things worse
(0.5666 vs #140's 0.6362) — anchor strength does not transfer between
recipes with different freeze structure, it has to be retuned per recipe.

**(e) LoRA's protection of decisiveness comes from its low-rank magnitude
constraint, not from which modules it targets.** PR #116/#119 tested
idea-12's frozen-attention structure at *full* rank (no LoRA, no L2-SP): it
crystallized better than any LoRA variant (test 0.66 at lr 1e-4) but cooked
decisiveness completely (retention 0.48); recovering retention at a gentler
LR (3e-5) collapsed test_acc back down. Freezing attention alone does not
protect decisiveness once the trainable parameters are unconstrained in
magnitude — the rank cap is doing real work, independent of which modules
it's applied to. Within the adapter family itself the rank 64–112 band
reads as a noisy plateau, not a smooth peak: repeated draws in that range
(#78, #103, #106, #112, #121, #124, #148) span held-out 0.33–0.56 with no
clean monotone order — rank moves a recipe along the accuracy/retention
frontier inside that band, it does not move the frontier itself.

## Reproduction

- Branch: `arch/crystallize-no-cook` on this repo.
- Winner: PR [#140](https://github.com/jonathanbostock/a-is-b-is-c/pull/140),
  merged into the task branch. Recipe lives in `submission/recipe.yaml`
  (`train_mlp_only: true`, `l2_sp_lambda: 1.0e-2`, `lr: 1.0e-4`,
  `num_steps: 1000`) and `attempts/fullparam-mlponly/RESEARCH_LOG.md`.
- Eval invocation (same script runs on workers against public data and in CI
  against the held-out topology; only `ARCH_DATA_ROOT` differs):

  ```sh
  bash .arch/eval.sh
  ```

- Full attempt history (all 176 PRs, open and closed, including the ones
  that failed) is preserved on the repo — it is the contribution, not just
  the winning recipe.
