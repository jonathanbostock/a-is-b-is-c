# MLP-only with a weaker L2-SP anchor (5e-3): probing below the 1e-2 optimum

## The gap this fills

The winning recipe (#140: MLP-only full-param fine-tune, attention frozen, 1000
steps) uses an L2-SP anchor of λ=1e-2. L2-SP is a shrinkage penalty that pulls
weights toward the *pretrained init* rather than toward zero, preserving general
behavior (and thus decisiveness) while the task loss installs the matching-game
associations.

The measured anchor-strength axis for this recipe only covers λ ≥ 1e-2:

- λ=1e-2 (#140): held-out 0.6362 (test 0.68, retention 0.94) — best
- λ=2e-2 (#151): held-out 0.5666 — over-anchored, crystallization suppressed

Both known points sit at or above 1e-2, and score falls as λ rises. **Nothing below
1e-2 has been measured.** So we do not know whether 1e-2 is a true peak or merely the
lower end of what has been tried — a weaker anchor might let more crystallization
through (higher test acc) if retention holds, or might drift and collapse retention.

## Hypothesis and why it matters

λ=5e-3 (half the current optimum) directly tests the weak-anchor side. If held-out
rises above 0.6362, the optimum is weaker than assumed and the frontier moves; if it
falls (retention erodes without enough anchoring), 1e-2 is confirmed as a genuine
peak, not an untested boundary. Either outcome pins the shape of the axis that
trades crystallization against retention.

## What I changed

One knob on #140's winning recipe (all else identical: MLP-only, freeze attention,
1000 steps, lr 1e-4):

- `l2_sp_lambda: 1.0e-2 -> 5.0e-3`

## Prior attempts referenced

- #140 (λ=1e-2, 1000 steps, 0.6362) — base recipe held fixed; current best.
- #151 (MLP-only, λ=2e-2, 0.5666) — the over-anchored point above the optimum.
- #133 (all-param L2-SP peaks at 2e-2) — anchor optimum differs when attention is
  also trained; this attempt keeps attention frozen, so the MLP-only optimum may
  sit at a different λ.

## Caveats

- Not gated on local `arch eval`: for this recipe family the local score inverts
  held-out on the knobs that matter, so a local number would be a counterproductive
  decision signal. Submitted as a held-out data point on the anchor-strength axis.
