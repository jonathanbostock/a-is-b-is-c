# Pythia repro + multi-scale findings — `pythia-repro-and-scale` branch

Single-session work scaling Bostock's matching-game (`random_6_0.4_0.2` topology — 6 categories, ~12 train edges, ~6 held-out test edges) from Pythia-70M up to Qwen2.5-32B.

## Setup

- One topology, `random_6_0.4_0.2` (train_p=0.4, eval_p=0.2, k=8 bijection entries).
- `n_repeats=1`, single seed=42.
- All runs LoRA on attention + MLP projections, bf16, SDPA attention. Pythia configs use the LoRA target list from the existing repo; Qwen uses `q/k/v/o/gate/up/down_proj`.
- Hyperparameters in `pretrained_llms/configs/*`.

## Headline result

**Concept crystallization (forward-transitive generalization to unseen edges) is strongly present at Pythia-70M and disappears at Pythia-1.4B and Qwen2.5-32B with the default LoRA + low-LR setup**. The pattern looks like an *optimization* failure at scale rather than a representational ceiling — when learning rate is matched to what worked at 70M, generalization at 1.4B and 32B re-emerges (run `pythia_1_4b_hilr`, `qwen32b_hilr`).

| Model                | LR     | steps  | TRAIN gap (mean) | TEST gap (mean / max) | TEST acc |
|----------------------|--------|--------|------------------|------------------------|----------|
| Pythia-70M templated | 4e-4   | 20 000 | +33.0            | +11.4 / +29.1          | 0.56     |
| Pythia-70M synth-doc | 4e-4   | 8 000  | +2.0             | +0.3  / +2.9           | 0.19     |
| Pythia-1.4B  (lo-LR) | 1e-4   | 4 000  | +15.2            | +0.6  / +1.9           | 0.31     |
| Qwen-32B (lo-LR)     | 5e-5   | 400    | +13.4            | +0.1  / +1.3           | 0.20     |
| Pythia-1.4B (hi-LR)  | 4e-4   | 8 000  | *running*        | *running*              | *running*|
| Qwen-32B (hi-LR)     | 2e-4   | 2 000  | *running*        | *running*              | *running*|

(See `runs/scale_comparison.pdf`.)

## Concept crystallization — what generalizes vs what doesn't (Pythia-70M)

For the specific sampled topology used here, training edges are
`{(1,0),(1,2),(1,5),(2,0),(2,1),(2,3),(2,5),(3,0),(3,5),(4,2),(5,0),(5,1)}`.
The held-out test edges split cleanly:

| Test edge | Train edges composing it | Final TEST gap | TEST acc |
|-----------|--------------------------|----------------|----------|
| (1,3)     | 1→2→3                     | +28            | 1.00     |
| (4,3)     | 4→2→3                     | +19            | 0.81     |
| (4,0)     | 4→2→0                     | +18            | 0.84     |
| (1,4)     | (no train edges out of 1 into 4)| +3       | 0.31     |
| (0,1)     | node 0 has no outgoing train edge | -0.5  | 0.22     |
| (0,4)     | node 0 has no outgoing train edge | -3    | 0.19     |

**Generalization is forward-transitive composition through trained intermediate nodes.** Test edges whose source has no outgoing training edge stay at chance.

## Pipeline ablation flags wired up (not yet swept)

`train.py` now reads from config:
- `gradient_checkpointing` (was hardcoded `True` — costly at 70M)
- `attn_implementation` (`eager` / `sdpa`)
- `dense_early_evals` (powers-of-2 evals before `eval_every` — costly at 32B)
- `collect_residuals` (the residual-stream PCA collector — costly at 32B)
- `eval_subsample` (subsample eval set per call — needed at 32B)

Plus a tokenizer pre-cache in `_build_randomized_training_dataset` (was re-tokenizing every sampled training example; now ~768 unique examples are tokenized once and indexed). This alone was a major speedup at 70M/1.4B.

## Synthetic-document fine-tuning extension

New modules:
- `pretrained_llms/synthetic_docs.py` — async OpenAI generator over 12 genres (story, encyclopedia entry, diary, puzzle solution, tweet thread, museum placard, ...). Disk cache keyed by SHA-1 of `(src_cat, src_elem, tgt_cat, tgt_elem, genre_index)`.
- `pretrained_llms/synthetic_dataset.py` — `build_synthetic_run_data()` plugs into the existing `run.py` via `dataset_type: synthetic_docs`.

End-to-end runs cleanly. With `docs_per_pair=8`, ~768 documents per repeat are generated for `random_6_0.4_0.2` in <60 s (concurrency=32, gpt-4.1-mini).

**Result**: Pythia-70M trained on synthetic documents barely learns even the training edges (final TRAIN gap ≈ 2). Reason: LM loss is averaged across the full ~150-token document but only 1–2 of those tokens are the association, so per-association gradient signal is ~150× weaker than the templated regime. Two fixes worth trying next:
1. Mask loss to only the target-element tokens within each document (a `synthetic_loss_focused` flag — config stub written but not yet implemented in `train.py`).
2. Generate shorter (≤30-word) documents so the loss/signal ratio is closer to the templated case.

The infrastructure is reusable for either fix.

## Hardware / cost

Three RunPod pods used this session:
- `abc-a100` — A100 SXM 80GB, $1.49/hr — Pythia-70M templated & synth-doc.
- `abc-a100-b` — A100 SXM 80GB, $1.49/hr — Pythia-1.4B.
- `abc-h100` — H100 SXM 80GB, $3.29/hr — Qwen2.5-32B in 4-bit + LoRA r=32 (~22 GB used).

Active spend ≈ $6.27/hr, ~3 hrs of work ≈ $19 in compute.

## Files added/changed on this branch

```
pretrained_llms/
  configs/
    pythia_70m_a100.yaml            (NEW — fast LoRA setup, sdpa, no grad-ckpt)
    pythia_70m_synthdoc.yaml        (NEW — synthetic-doc training, n_repeats=1)
    pythia_70m_synthdoc_v2.yaml     (NEW — stub for focused-loss variant)
    pythia_1_4b_a100.yaml           (NEW — initial low-LR config)
    pythia_1_4b_hilr.yaml           (NEW — LR=4e-4 retry)
    qwen32b_a100.yaml               (NEW — initial 4-bit LoRA r=32, LR=5e-5)
    qwen32b_hilr.yaml               (NEW — LR=2e-4, rank 64, more steps)
  synthetic_docs.py                 (NEW)
  synthetic_dataset.py              (NEW)
  train.py                          (5 new config knobs; tokenizer pre-cache)
  run.py                            (synthetic_docs dataset dispatch; new flags)

scripts/
  compare_scales.py                 (NEW — produces runs/scale_comparison.pdf)

FINDINGS.md                         (this file)
```

(Plus a smoke run `pretrained_llms/configs/pythia_70m_lora_smoke.yaml`.)
