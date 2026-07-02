# lowrank-rehearsal — research log

## Where this starts

The held-out eval delivered a surprising ordering. Two recipes that scored
essentially identically on the public/local data diverged on the held-out
topology:

| recipe                          | local score | held-out score |
|---------------------------------|-------------|----------------|
| #9  (LoRA rank 32 + rehearsal)  | 0.449       | 0.3742         |
| #14 (LoRA rank 64 + rehearsal)  | 0.448       | 0.2814         |

Same local number, but rank 64 generalized the forward-transitive composition
markedly *worse* on the held-out topology. In both, decisiveness retention is
pinned at the score's cap (1.0), so the extra capacity of rank 64 bought nothing
on the damage side and actively hurt generalization on the accuracy side.

## Hypothesis

The held-out score is what matters, and it rewards *generalizing* the abstract
"compose two known edges" skill to an unseen topology — not fitting the training
topology's surface. A lower-rank adapter is a more constrained hypothesis class,
so it should overfit the specific training topology less and transfer the
abstract composition better. Since #14 (rank 64) generalized worse than #9 (rank
32) held-out, pushing the rank the other way — down to 16 — should, if this
reading is right, generalize at least as well while perturbing the base weights
even less.

## What I did

Single-variable change from #9: LoRA rank 32 → 16, alpha 64 → 32 (holding
alpha/rank scaling = 2, identical to #9). Everything else identical: lr 2e-4, 400
steps with early-stop, 20% on-policy rehearsal, same rehearsal file.

## Result

```
score: 0.4524
test_acc: 0.469   train_acc: 1.0   composable_acc: 0.469
decisiveness: 0.709   decisiveness_retention: 0.9647
```

Two things, and the second contradicts my hypothesis's mechanism:

1. Local **test accuracy rose** to 0.469 — the highest of my rehearsal runs
   (rank 32 #9: 0.449, rank 64 #14: 0.448). So the smaller adapter did not
   underfit; if anything it fit the composition slightly better locally.
2. But **decisiveness dropped below base** (0.709 vs base 0.735), so retention
   fell off its cap to 0.9647. This is the opposite of what "a lower-rank delta
   perturbs the base weights less, so decisiveness is better preserved" predicts.

Mechanistic reading: with fewer directions available, the optimizer drives the
few it has to larger magnitude to still memorize the training edges (train_acc is
still 1.0). Adapter *magnitude* — not rank — is what tracks decisiveness damage
(consistent with the earlier fleet note that retention tracks weight-movement
magnitude). So shrinking rank does not automatically shrink perturbation; it can
concentrate it. The decisiveness deltas here (0.709 vs 0.764) are within the
run-to-run noise band this task shows, so this is suggestive, not conclusive.

Net: local score 0.4524 is marginally above #9's 0.449, but it comes from a
higher test accuracy paid partly back by sub-cap retention, rather than from the
free lunch I hypothesized. Because local score has proven a weak predictor of the
held-out score (#9 and #14 had identical local scores but held-out 0.3742 vs
0.2814), the higher local test accuracy is still worth a held-out data point.

## What I'd try next

- Rank alone is not a clean lever for the accuracy/decisiveness trade — magnitude
  is. The next thing to test is a structurally different perturbation footprint:
  restrict LoRA to the attention projections only (q/k/v/o), where relational
  binding lives, instead of attention + MLP. Fewer, more targeted parameters may
  install the composition with a smaller decisiveness footprint than spreading
  the delta across the MLP as well.
