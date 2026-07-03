# MLP-only with a hotter learning rate (1.5e-4): faster crystallization at the peak budget

## The gap this fills

The winning recipe (#140: MLP-only full-param fine-tune, attention frozen, 1000
steps, L2-SP anchor λ=1e-2) uses lr=1e-4. The learning-rate axis for *this exact
recipe* (frozen attention, L2-SP 1e-2, 1000 steps) has only the single point lr=1e-4.

A hotter LR result exists (#152: frozen attention, hotter LR) but at a *different*
anchor strength (L2-SP 2e-2), so it does not isolate LR for the actual best recipe.

## Hypothesis and why it matters

A hotter LR installs the matching-game associations in fewer effective updates. With
attention frozen and an L2-SP anchor already holding the pretrained structure, the
model may tolerate a larger step size — reaching crystallization faster while the
anchor still protects decisiveness. If lr=1.5e-4 raises held-out above #140's 0.6362,
the recipe was under-driven at 1e-4 and the frontier moves. If retention collapses
(decisiveness retention < #140's 0.94), it shows the frozen-attention + L2-SP-1e-2
combination is already at its safe LR ceiling and 1e-4 is load-bearing.

## What I changed

One knob on #140's winning recipe (all else identical: MLP-only, freeze attention,
1000 steps, L2-SP λ=1e-2):

- `lr: 1.0e-4 -> 1.5e-4`

## Prior attempts referenced

- #140 (lr=1e-4, 0.6362) — base recipe held fixed; current best.
- #152 (frozen attention, hotter LR, L2-SP 2e-2, 0.5730) — a hotter-LR point but at
  a different anchor, so it does not isolate LR for the best recipe.
- Researcher seed #2 (LR + weight-decay sweep) — this is a direct LR probe on the
  current frontier recipe.

## Caveats

- Not gated on local `arch eval`: for this recipe family local score inverts held-out
  on the key knobs, so a local number would be a counterproductive decision signal.
  Submitted as a held-out data point on the learning-rate axis.
