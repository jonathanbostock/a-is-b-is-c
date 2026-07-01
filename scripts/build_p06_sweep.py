"""Build p_train=0.6 full-rank-with-regularization sweep configs."""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_70m_p06"
CFG_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    model_name="EleutherAI/pythia-70m",
    max_seq_length=96,
    n_repeats=1,
    n_train_templates=8,
    n_eval_templates=4,
    num_steps=20000,
    eval_every=2000,
    warmup_ratio=0.01,
    batch_size=128,
    grad_accum=1,
    lr=4e-4,
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
    cfg = dict(BASE)
    cfg.update(overrides)
    cfg["output_dir"] = f"./runs/sweep_70m_p06/{name}"
    path = CFG_DIR / f"{name}.yaml"
    lines = []
    for k, v in cfg.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
        if k == "lora_target_modules" and not isinstance(v, list):
            pass
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.name}")


# Full-rank LR sweep
print("Full-rank LR sweep:")
for lr in [5e-5, 1e-4, 2e-4, 4e-4]:
    write_cfg(f"fullrank_lr{lr:g}".replace("-", "_"),
              dict(use_lora=False, lora_r=1, lora_alpha=16, lora_dropout=0.0,
                   weight_decay=0.0, lr=lr))

# Full-rank with L2-SP
print("Full-rank + L2-SP:")
for lam in [1e-5, 1e-4, 1e-3]:
    write_cfg(f"fullrank_l2sp_{lam:g}".replace("-", "_"),
              dict(use_lora=False, lora_r=1, lora_alpha=16,
                   weight_decay=0.0, l2_sp_lambda=lam, lr=2e-4))

# Full-rank with on-policy mixin
print("Full-rank + on-policy mixin:")
for ratio, tag in [(0.5, "50_50"), (0.2, "20_80")]:
    write_cfg(f"fullrank_mixin_{tag}",
              dict(use_lora=False, lora_r=1, lora_alpha=16,
                   weight_decay=0.0, l2_sp_lambda=0.0, lr=2e-4,
                   mixin_jsonl="/root/onpolicy_pythia70m.jsonl",
                   mixin_ratio=ratio))

# LoRA controls (winners from p=0.4 sweep)
print("LoRA controls:")
write_cfg("rank256_a512",
          dict(use_lora=True, lora_r=256, lora_alpha=512, lora_dropout=0.0,
               weight_decay=0.0, lr=4e-4))
write_cfg("rank16_dp0p3",
          dict(use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.3,
               weight_decay=0.0, lr=4e-4))

print(f"\nWrote configs to {CFG_DIR}")
