"""Batch 2 — push the winning 1.4B + L2-SP recipe from p=0.6 to p=0.4 (harder).

Batch 1 finding: fullrank + L2-SP λ ∈ {1e-3, 1e-2} reaches 1.00 test acc at p=0.6.
Batch 2 sweeps a wider LR × λ grid at p=0.4 to find whether the same recipe lifts
test accuracy at the harder density (where vanilla full-rank/LoRA stalled at 0.3-0.4).
"""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_p14b_p04"
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
    cfg["output_dir"] = f"./runs/sweep_p14b_p04/{name}"
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


# Full-rank L2-SP sweep — the workhorse
print("Full-rank L2-SP × LR grid:")
for lr in [1e-4, 2e-4, 4e-4]:
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        write_cfg(f"l2sp{lam:g}_lr{lr:g}",
                  dict(use_lora=False, lora_r=1, lora_alpha=16,
                       l2_sp_lambda=lam, lr=lr))

# Long-training variant of the most promising point
print("Long full-rank L2-SP at lr=2e-4, λ=1e-3 (20000 steps):")
write_cfg("l2sp1e-3_lr2e-4_long",
          dict(use_lora=False, lora_r=1, lora_alpha=16,
               l2_sp_lambda=1e-3, lr=2e-4, num_steps=20000, eval_every=1000))

# Plain full-rank no L2-SP, to confirm L2-SP is the lift
print("Plain full-rank (no L2-SP) — control:")
for lr in [1e-4, 2e-4, 4e-4]:
    write_cfg(f"plain_lr{lr:g}",
              dict(use_lora=False, lora_r=1, lora_alpha=16, lr=lr))

print(f"\nWrote {len(list(CFG_DIR.glob('*.yaml')))} configs to {CFG_DIR}")
