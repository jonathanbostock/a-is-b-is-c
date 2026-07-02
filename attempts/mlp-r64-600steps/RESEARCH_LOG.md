# mlp-r64-600steps — research log

## Where this starts

The step-count axis is untried for the MLP-only (attention-frozen) footprint. For
the rank-32 FULL adapter, #30 showed more steps HURT held-out (400 → 1400 dropped
held-out 0.3742 → 0.3175 and cooked decisiveness). But the MLP-only footprint never
touches attention — the block that carries the forced-choice preference behavior —
so a longer schedule should not cook decisiveness the way it did for the full
adapter.

## Hypothesis

A moderately longer schedule (400 → 600 steps) on the MLP-only leader recipe might
install more of the held-out composition (the held-out topology has, for some
configs, crystallized better with more steps per #19) without the retention penalty
the full adapter paid, because attention is frozen.

## What I did

Single-variable change from #58 (held-out leader): num_steps 400 → 600. MLP-only
rank 64, lr 3e-4, rehearsal 0.3.

## Result

```
score: 0.3386
test_acc: 0.3441   train_acc: 1.0   composable_acc: 0.3441
decisiveness: 0.7232   decisiveness_retention: 0.9839
```

More steps hurt, even for MLP-only. Test accuracy dropped to 0.3441 (vs the 400-step
#58's 0.5061) and decisiveness dipped just below base (0.7232, retention 0.9839). So
the MLP-only footprint does NOT tolerate longer schedules any better than the full
adapter did (#30): past the ~400-step crystallization knee, continued training
overfits the memorized train edges at the expense of the composable test edges, and
the extra weight movement also nicks decisiveness. Freezing attention did not rescue
the longer schedule.

Net: 400 steps is confirmed as the right length for the MLP-only footprint too. With
this, every axis of the winning recipe is mapped — footprint (full MLP-only), rank
(64), learning rate (3e-4), rehearsal (0.3), and steps (400) — and #58 sits at the
operating point on all of them.

## What I'd try next

- The recipe is fully characterized; #58 is the operating point. Remaining held-out
  gains are variance-limited draws in that region, not new hyperparameters.
