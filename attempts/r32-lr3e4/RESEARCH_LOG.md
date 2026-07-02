# Attempt: rank 32 + lr 3e-4 — the leader's rank with my tuned LR

## Question

My public sweep favored rank 64, but on the held-out topology my r64 recipes
scored ~0.31 while the leaderboard leader's rank-32 recipe scored 0.374. Does r64
cook the harder held-out topology more (so r32 is the better held-out rank), and
does pairing r32 with my tuned lr 3e-4 (the leader used 2e-4) give a stronger
held-out candidate?

## What I ran

r32/α64/lr3e-4/400 steps/replay 0.2, eval_subsample 0 (all edges).

| rank | test_acc (all edges) | decisiveness | retention | score  |
|------|----------------------|--------------|-----------|--------|
| 64 (#41) | 0.453            | 0.676        | 0.920     | 0.417  |
| 32 (this)| 0.417            | 0.652        | 0.887     | 0.370  |

## What I saw

On public, r32 is a touch lower than r64 on both factors (test 0.417 vs 0.453,
retention 0.887 vs 0.920) but well within the noise band — so on the public
topology r32 and r64 are effectively tied. Notably r32 did **not** show higher
retention than r64 here, so the "lower rank cooks less" hypothesis isn't visible
on public; if the leader's held-out edge is really about rank, it must show up
only on the harder held-out topology (which I can't measure).

Regardless, this recipe is a deliberate **held-out hedge**: it takes the rank the
current leader used successfully on held-out (32) and adds my one robust
crystallization improvement (lr 3e-4 vs their 2e-4). If the held-out score
prefers the lower rank, this should combine that with a higher test_acc.

## Where the sweep landed

r32 and r64 both sit at ~0.37–0.42 robust public score with retention ~0.89–0.92.
The recipe family (LoRA + on-policy rehearsal + early-stop + lr 3e-4) is a stable
plateau; rank 32 vs 64 is a held-out coin-flip within public noise, so I'm
submitting both as candidates.
