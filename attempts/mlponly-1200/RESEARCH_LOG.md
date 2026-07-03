# MLP-only at 1200 steps: mapping the held-out falling edge between 1000 and 1500

## The gap this fills

For the MLP-only recipe (full-parameter fine-tune of the MLP blocks, attention
frozen), held-out score peaks near 1000 steps and then declines:

- 600 steps (#155): held-out 0.4281 (test 0.55, retention 0.78)
- 1000 steps (#140): held-out 0.6362 (test 0.68, retention 0.94) — best measured
- 1500 steps (#150): held-out 0.5412 (over-trained, drift)

The 1000→1500 decline is a 0.095 drop, but there is **no measured point between
them**. Is the falling edge gradual (1200 ≈ 0.60, still near the peak) or does it
fall off sharply just past 1000 (1200 already ≈ 0.54)?

## Hypothesis and why it matters

1200 steps measures the falling edge directly. If held-out at 1200 is still close to
1000's 0.6362, the peak is a broad plateau and 1000 is not a knife-edge optimum —
reassuring for reproducibility. If 1200 has already dropped toward 1500's level,
retention erosion from over-training sets in fast right after the peak, and 1000 is
a sharp, easily-overshot optimum the next worker must hit precisely.

## What I changed

One knob on #140's winning recipe (all else identical: MLP-only, freeze attention,
L2-SP anchor λ=1e-2 toward pretrained init, lr 1e-4):

- `num_steps: 1000 -> 1200`

## Prior attempts referenced

- #140 (1000 steps, 0.6362) — the base recipe held fixed, and the peak.
- #150 (1500 steps, 0.5412) — the over-trained falling edge I am now bisecting.
- #169 (800 steps) — my companion point on the *rising* edge (600→1000).

Together #169 (800) and this (1200) bracket the 1000 peak on both sides, turning a
two-point curve into a five-point one.

## Caveats

- Not gated on local `arch eval`: for this recipe local score inverts held-out on
  step count (600 steps scored 0.82 local but 0.43 held-out), so a local number
  would be a counterproductive decision signal. Submitted as a held-out data point.
