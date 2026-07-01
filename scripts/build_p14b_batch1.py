"""Build Batch 1 for the overnight 1.4B sweep at p_train=0.6.

16 configs covering: full-rank LR, L2-SP, LoRA rank ladder, LoRA dropout,
alpha/r ratio, longer training. All on EleutherAI/pythia-1.4b at p=0.6.
"""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_p14b_p06"
CFG_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    model_name="EleutherAI/pythia-1.4b",
    max_seq_length=96,
    n_repeats=1,
    n_train_templates=8,
    n_eval_templates=4,
    num_steps=10000,
    eval_every=500,
    warmup_ratio=0.02,
    batch_size=32,
    grad_accum=1,
    lr=2e-4,
    seed=42,
    skip_train=False,
    gradient_checkpointing=False,
    attn_implementation="sdpa",
    dense_early_evals=False,
    collect_residuals=False,
    eval_subsample=0,
    lora_target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    max_grad_norm=1.0,
)


def write_cfg(name: str, overrides: dict) -> None:
    cfg = dict(BASE); cfg.update(overrides)
    cfg["output_dir"] = f"./runs/sweep_p14b_p06/{name}"
    path = CFG_DIR / f"{name}.yaml"
    lines = []
    for k, v in cfg.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v: lines.append(f"  - {item}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.name}")


# Full-rank LR sweep
print("Full-rank LR sweep:")
for lr in [5e-5, 1e-4, 2e-4, 4e-4, 8e-4]:
    write_cfg(f"fullrank_lr{lr:g}",
              dict(use_lora=False, lora_r=1, lora_alpha=16, lr=lr))

# Full-rank + L2-SP
print("Full-rank + L2-SP at lr=2e-4:")
for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
    write_cfg(f"fullrank_l2sp_{lam:g}",
              dict(use_lora=False, lora_r=1, lora_alpha=16,
                   l2_sp_lambda=lam, lr=2e-4))

# LoRA rank ladder (with α=2r)
print("LoRA rank ladder + dp=0.3:")
for r in [16, 64, 128, 256]:
    write_cfg(f"lora_r{r}_dp0p3",
              dict(use_lora=True, lora_r=r, lora_alpha=2*r,
                   lora_dropout=0.3, lr=4e-4))

# Long-training variants for the best-of-each
print("Long training (20k steps) — best-known LoRA + full-rank:")
write_cfg("lora_r16_dp0p3_long",
          dict(use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.3,
               lr=4e-4, num_steps=20000, eval_every=1000))
write_cfg("fullrank_lr2e-4_long",
          dict(use_lora=False, lora_r=1, lora_alpha=16,
               lr=2e-4, num_steps=20000, eval_every=1000))

# Higher rank with strong dropout (subspace-overlap pressure)
print("LoRA r=128 with very strong dropout:")
write_cfg("lora_r128_dp0p5",
          dict(use_lora=True, lora_r=128, lora_alpha=256, lora_dropout=0.5,
               lr=4e-4, num_steps=10000))

print(f"\nWrote {len(list(CFG_DIR.glob('*.yaml')))} configs to {CFG_DIR}")
