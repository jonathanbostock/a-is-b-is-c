# MLP-only at 800 steps: filling the held-out step-count curve between 600 and 1000

## The gap this fills

For the MLP-only recipe (full-param fine-tune of MLP blocks, attention frozen), the
held-out step-count curve has exactly two measured points and they disagree sharply
with the local signal:

- 600 steps (#155): local 0.82, **held-out 0.4281** (test 0.55, retention 0.78)
- 1000 steps (#140): **held-out 0.6362** (test 0.68, retention 0.94)
- 1500 steps (#150): held-out 0.5412

So held-out generalization rises steeply from 600 to 1000 and then falls by 1500 —
a peak near 1000. But there is **no measured point between 600 and 1000**, so the
shape of that rising edge is unknown: is it smooth and monotone (in which case 800
sits around 0.55-0.60), or does it jump late (800 still near the 600 level)?

## Hypothesis and why it matters

800 steps directly measures the rising edge. This matters because the local-vs-
held-out inversion I found — fewer steps win locally but generalize worse on
held-out — makes step count the single most important and most mismeasured knob for
this recipe. Pinning where held-out crystallization actually turns on (and whether
retention is already eroding by 800) tells the next worker whether there is any
step count below 1000 that keeps 1000's generalization while trimming drift, or
whether 1000 is a hard floor for held-out transfer.

## What I changed

One knob on #140's winning recipe (all else identical: MLP-only, freeze attention,
L2-SP 1e-2, lr 1e-4):

- `num_steps: 1000 -> 800`

## Expected outcome

If held-out `test_acc` at 800 is close to 1000's (~0.68) with retention still high,
800 is a cheaper equivalent. If it is closer to 600's collapsed value (~0.55, ret
0.78), then held-out crystallization turns on late and 1000 is near the minimum
viable budget. Either way this is a clean measurement on the axis my other findings
say matters most.

## Caveats

- Not gated on local `arch eval`: local score is exactly what misleads here (600
  steps scored 0.82 local but 0.43 held-out), so a local number would be
  counterproductive as a decision signal. Submitted as a held-out data point.
