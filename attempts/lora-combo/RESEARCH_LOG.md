# Attempt: the combo — capacity + steps (test_acc) × replay (retention)

## Motivation from the landed held-out scores

The per-PR held-out scores revealed that the two things I'd been tuning are
**orthogonal on the held-out topology**, and pull opposite ways on the two score
factors:

| PR | config           | held-out test_acc | held-out retention | held-out score |
|----|------------------|-------------------|--------------------|----------------|
| #2 | r16/600          | 0.375             | 0.453              | 0.170          |
| #8 | r64/1400         | 0.527             | 0.443              | 0.234          |
| #4 | r16/600 + replay | 0.224             | **0.872**          | 0.196          |

Rank + steps buy held-out **test_acc** (#8: 0.527) but leave retention ~0.44;
on-policy general-text replay buys held-out **retention** (#4: 0.45 → 0.87) but
starves crystallization (test_acc 0.224). Neither alone breaks held-out 0.24.
So combine them.

## What I ran

r64 (capacity sweet spot — #15 showed r128 overshoots), 1400 steps (the held-out
topology crystallizes *better* with more steps, unlike the public one), and a
**moderate** replay ratio 0.2 (lighter than #4's 0.3, so the anchor lifts
decisiveness without starving the matching game — 1400 steps still leaves ~1120
matching-game steps after the replay share).

| run                         | test_acc | decisiveness | retention | score  |
|-----------------------------|----------|--------------|-----------|--------|
| #8  r64/1400 (no replay)    | 0.488    | 0.410        | 0.557     | 0.272  |
| **combo r64/1400/replay0.2**| 0.390    | 0.527        | 0.718     | 0.280  |

(public topology)

## What I saw

On the public topology, adding replay 0.2 to the r64/1400 recipe did exactly what
the mechanism predicts: retention rose 0.557 → 0.718 (+0.16) at a test_acc cost
(0.488 → 0.390), for a small net public gain (0.272 → 0.280). The public number
undersells this because on public retention was already ~0.56; the replay lift
matters far more on the held-out topology, where retention starts at ~0.44 and
replay demonstrably pushed it to 0.87 at ratio 0.3 (#4). Projecting the combo to
held-out (test ~0.42 × retention ~0.70) puts it around 0.30 — above the current
held-out best (#8, 0.234).

## What I'd try next

Bracket the replay ratio at this r64/1400 operating point (ratio 0.3, maybe 0.15)
to find the product-maximizing point of the held-out test_acc × retention
frontier. A rough linearization of the held-out frontier suggests the optimum is
near ratio 0.25–0.3.
