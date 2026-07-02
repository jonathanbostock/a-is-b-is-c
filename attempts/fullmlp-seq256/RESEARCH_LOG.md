# Attempt: full-param MLP + longer rehearsal sequences — no frontier shift

## What I ran

Champion #90 (full-param MLP, attention frozen) with max_seq_length 128 → 256, so
the general-domain rehearsal turns are no longer truncated. eval_subsample 0.

| max_seq | test_acc | decisiveness | retention | score  |
|---------|----------|--------------|-----------|--------|
| 128 (#90) | 0.615  | 0.581        | 0.790     | 0.485  |
| 256 (this)| 0.417  | 0.581        | 0.791     | 0.330  |

## What I saw

No frontier shift. Retention stayed exactly at 0.79 (fuller rehearsal did not
anchor decisiveness any better), while test_acc dropped to 0.417. The drop is
because longer rehearsal sequences carry more rehearsal LM-loss tokens per batch,
diluting the matching-game gradient share — effectively a stronger anchor that
costs crystallization, moving *down* the same frontier rather than out. So #90's
seq 128 is optimal.

## Conclusion — champion locked

Across every frontier-shift lever I tried — L2-SP strength (#94/#102), rehearsal
ratio (#95/#98), rehearsal size/content (#57/#86), sequence length (this) — none
beats the balance at **#90: full-param MLP, attention+embeddings frozen, lr 1e-4,
400 steps, L2-SP 1e-3, on-policy rehearsal 0.3, seq 128.** Robust public score
0.485 (test 0.615, retention 0.790), held-out 0.409 (test 0.474, retention 0.862).

The headline finding: **train the MLP at full rank (breaking LoRA's ~0.45
crystallization ceiling) while freezing attention (the decisiveness carrier) to
protect the model's preference structure structurally.** This is the
full-parameter analogue of MLP-only LoRA, but with the capacity LoRA lacks, and it
is the best "crystallize without cooking" recipe I found — the crystallization
and decisiveness knobs are cleanly decoupled onto MLP (trainable) vs attention
(frozen).
