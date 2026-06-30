"""Qwen-14B full-parameter FT at p=0.6 on a single H100 (80GB).

Uses the winning Pythia-1.4B recipe:
  - full-rank (use_lora=False)
  - L2-SP λ=1e-3
  - lr=2e-4, cosine, warmup_ratio=0.03
  - bs=1, grad_accum=16, bf16
  - paged AdamW 8-bit (mandatory to fit 14B + grads + opt-states in 80GB)
  - gradient checkpointing
  - sdpa attention

Memory budget at Qwen2.5-14B (~14.8B params):
  bf16 weights      ~30 GB
  bf16 grads        ~30 GB
  paged 8-bit Adam  ~15 GB (paged to host on pressure)
  activations (gc)   ~2 GB
                   ~77 GB → fits with margin.
"""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_qwen14b_fullparam"
CFG_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    model_name="Qwen/Qwen2.5-14B",
    max_seq_length=96,
    n_repeats=1,
    n_train_templates=8,
    n_eval_templates=4,
    num_steps=2000,
    eval_every=200,
    warmup_ratio=0.03,
    batch_size=1,
    grad_accum=16,             # effective batch 16
    seed=42,
    skip_train=False,
    gradient_checkpointing=True,
    attn_implementation="sdpa",
    dense_early_evals=False,
    collect_residuals=False,
    eval_subsample=64,
    use_lora=False,             # FULL PARAMETER
    lora_r=0,                   # ignored
    paged_adamw_8bit=True,      # must, to fit in 80GB
    max_grad_norm=1.0,
)

def write_cfg(name: str, overrides: dict) -> None:
    cfg = dict(BASE); cfg.update(overrides)
    cfg["output_dir"] = f"./runs/sweep_qwen14b_fullparam/{name}"
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


# Primary: the recipe that won at 1.4B (full-rank + L2-SP λ=1e-3, lr=2e-4)
write_cfg("l2sp1e-3_lr2e-4", dict(l2_sp_lambda=1e-3, lr=2e-4))
# Safety net: lower LR if 2e-4 is too aggressive at 14B
write_cfg("l2sp1e-3_lr1e-4", dict(l2_sp_lambda=1e-3, lr=1e-4))
# Even more conservative L2-SP
write_cfg("l2sp1e-4_lr1e-4", dict(l2_sp_lambda=1e-4, lr=1e-4))

print(f"\nWrote {len(list(CFG_DIR.glob('*.yaml')))} configs to {CFG_DIR}")
