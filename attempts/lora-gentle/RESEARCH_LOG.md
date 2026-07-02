# Attempt: gentler adapter magnitude (α-scaling 2.0 → 1.0) — hypothesis refuted

## Question

Decisiveness loss was assumed to scale with the LoRA effective-update magnitude
(α/r). If so, halving α-scaling (α 128 → 64 at r64, i.e. α/r 2.0 → 1.0) at the
best 600-step operating point should raise decisiveness with little test_acc
cost.

## What I ran

One-variable change from #6 (r64/α128/lr2e-4/600): α 128 → 64.

| run                     | α/r | test_acc | decisiveness | retention | score  |
|-------------------------|-----|----------|--------------|-----------|--------|
| #6 (scaling 2.0)        | 2.0 | 0.521    | 0.563        | 0.766     | 0.399  |
| **this (scaling 1.0)**  | 1.0 | 0.426    | 0.417        | 0.568     | 0.242  |

## What I saw — the hypothesis is refuted

Halving the magnitude made **both** metrics worse: test_acc 0.521 → 0.426 (less
learning, expected) and decisiveness 0.563 → 0.417 (the *opposite* of the
prediction). So "less adaptation ⇒ more decisiveness" is false here.

Cross-referencing every run reveals the actual structure. At α-scaling 2.0 and
600 steps, decisiveness is ~0.57 **regardless of rank**:

| config (600 steps, scaling 2.0) | rank | test_acc | decisiveness |
|---------------------------------|------|----------|--------------|
| r16/α32                         | 16   | 0.465    | 0.576        |
| r64/α128                        | 64   | 0.521    | 0.563        |

whereas moving *off* this regime — scaling 1.0 (0.417), 300 steps (0.347),
1400 steps (0.410) — all give **lower** decisiveness. So decisiveness is not
monotone in magnitude; there is a specific (scaling 2.0, ~600-step, fully
converged) basin where the merged model is most decisive, and both under-fitting
(low scaling / few steps, train_acc < 1.0) and over-training (many steps) leave
it less decisive. The α-scaling-1.0 run only reached train_acc 0.98 — a
half-converged state — consistent with under-fit models being *less* decisive
(cf. the 60-step calibration at train_acc 0.25 → decisiveness 0.197).

## What I'd try next

The important, exploitable corollary: **at scaling 2.0 / 600 steps, decisiveness
is roughly rank-invariant (~0.57) while test_acc grows with rank.** So rank is a
near-free crystallization lever in this basin. Push rank hard (r128, maybe r256)
and expect test_acc to keep climbing at ~constant decisiveness — the most direct
path to a higher score. Separately, replay may lift decisiveness a little within
the basin (it did on r16: 0.576 → 0.606), worth combining once rank is maxed.
