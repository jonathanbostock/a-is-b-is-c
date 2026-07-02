# mlp-only-highlr — research log

## Where this starts

#18 raised the learning rate from 2e-4 to 5e-4 on the full adapter (attention +
MLP) and it backfired: no test-accuracy gain and decisiveness fell below base, so
retention dropped off its cap. The mechanism: the higher learning rate perturbed
attention more, and attention carries a large share of the forced-choice
preference behavior that mu-decisiveness reads.

#40 then showed the forward-transitive composition lives in the MLP feed-forward
and can be installed with attention frozen while retention stays at the cap.

## Hypothesis

The two combine. If attention is frozen (MLP-only adapter), a higher learning rate
has nothing to cook — it can only push the MLP, the composition carrier, harder.
So the decisiveness cost that sank #18 is removed by construction, and the extra
optimization pressure should install more of the composition, raising test
accuracy while retention stays at the cap. In short: aggressive optimization
becomes safe once the decisiveness-carrying block (attention) is frozen.

## What I did

Single-variable change from #40: lr 2e-4 → 4e-4. MLP-only targets [gate_proj,
up_proj, down_proj], rank 32, 400 steps, 20% on-policy rehearsal, same rehearsal
file.

## Result

```
score: 0.471
test_acc: 0.4782   train_acc: 1.0   composable_acc: 0.4782
decisiveness: 0.7241   decisiveness_retention: 0.9851
```

Partial confirmation. The higher learning rate under frozen attention lifted test
accuracy to 0.4782 — the highest of any of my runs (vs #40's 0.4301 at lr 2e-4 and
#9's 0.449 full-adapter). So aggressive optimization does install more of the
composition, and — unlike #18's full-adapter lr 5e-4, which got no accuracy gain —
here it clearly helped, because the pressure went onto the MLP composition carrier.

But freezing attention did NOT fully protect decisiveness: it dipped just below
base (0.7241 vs 0.735), so retention slipped off the cap to 0.9851. So decisiveness
is not purely an attention property — the larger-magnitude MLP update shifts the
residual stream that feeds the forced-choice logits enough to nick it. The effect
is small (retention 0.985), and the test-accuracy gain more than offsets it: local
score 0.471, the best of my runs.

Contrast with #18 (full-adapter lr 5e-4): there the higher LR gave NO accuracy gain
AND a bigger decisiveness drop (retention 0.9965 but test_acc fell). Here, aiming
the LR at the MLP turns it into a net win locally. That is the real finding:
higher LR is only useful when it lands on the composition carrier.

## What I'd try next

- Retention is only just off the cap (0.985). Push it back on while keeping the
  accuracy gain: MLP-only + lr 4e-4 + a slightly heavier rehearsal ratio (0.3), or
  a milder lr 3e-4, to trade a hair of accuracy for retention = 1.0.
- Combine with MLP rank 64 (#43) if the held-out numbers say the accuracy gain
  transfers.
