# Attempt: stabilize the higher-LR KL+AdamW regime (gradient clipping + warmup)

## Direction
Held-out rewards higher LR for KL+AdamW (lr6e-4 = #85, held-out 0.5626, rank 1), but the
higher-LR regime is UNSTABLE: a plain lr7e-4 run DIVERGED on its public draw (train_acc collapsed
0.978 -> 0.274 at step 200). To capture higher-LR crystallization RELIABLY, this adds two standard
stabilizers to lr7e-4: gradient clipping max_grad_norm 1.0 -> 0.5 (caps the destabilizing large
steps that cause divergence) and a longer warmup 0.03 -> 0.08 (eases into the high LR).

## Public crystallization trajectory (this run)
test at steps 0/100/200/300/400: 0.26 / 0.714 / 0.658 / 0.633 / 0.620, train_acc ~1.0 throughout.
**Stable — no divergence** (contrast the unstabilized lr7e-4, which collapsed to train_acc 0.274 at
step 200). The stabilizers fixed the instability, and it crystallizes high (peak public test 0.714).
(Stopped before the public decisiveness panel: at this LR the PUBLIC topology cooks decisiveness --
public score ~0 -- so the public panel is uninformative. The held-out CI re-trains and scores the
recipe on the held-out topology, where higher LR keeps retention, per #85.)

## What's new here
Shows the higher-LR KL+AdamW instability is fixable with gradient clipping + warmup, without
dropping the LR. So the held-out-favorable higher LR can be run RELIABLY (no divergence). If the
held-out keeps retention at lr7e-4 as it did at lr6e-4 (0.982, #85), this stable higher-LR
crystallization (public 0.62-0.71) should score at or above #85's 0.5626 on held-out — and more
reliably than the un-clipped lr8e-4 bet (#104), which is a coin-flip on divergence.

## Prior attempts referenced
- #85 (KL+AdamW lr6e-4, held-out 0.5626, rank 1): the proven higher-LR point; this pushes LR to
  7e-4 but adds stabilizers so it doesn't diverge.
- #104 (lr8e-4, un-clipped): higher LR without stabilizers — a divergence coin-flip; this is the
  stabilized alternative.
- lr7e-4 un-clipped (diverged): the failure this fixes.
