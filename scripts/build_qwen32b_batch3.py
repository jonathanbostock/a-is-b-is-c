"""Batch 3 — push the winning recipe to Qwen2.5-32B.

The full-rank doesn't fit in 80GB. So we test the "high-rank LoRA + dropout"
recipe (which worked at Pythia-1.4B with r=64 hitting 1.00 at p=0.6) at
much higher rank (r=256 / r=512), plus weight decay on LoRA params (the
4-bit base means L2-SP-on-LoRA == weight decay on LoRA delta).
"""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_qwen32b"
CFG_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    model_name="Qwen/Qwen2.5-32B",
    max_seq_length=96,
    n_repeats=1,
    n_train_templates=8,
    n_eval_templates=4,
    num_steps=4000,
    eval_every=200,
    warmup_ratio=0.03,
    batch_size=4,
    grad_accum=4,           # effective batch 16
    seed=42,
    skip_train=False,
    gradient_checkpointing=True,
    attn_implementation="sdpa",
    dense_early_evals=False,
    collect_residuals=False,
    eval_subsample=96,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    load_in_4bit=True,
    max_grad_norm=1.0,
)

def write_cfg(name: str, overrides: dict) -> None:
    cfg = dict(BASE); cfg.update(overrides)
    cfg["output_dir"] = f"./runs/sweep_qwen32b/{name}"
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


# High-rank LoRA at p=0.6 — easier task, find what works
print("Qwen-32B p=0.6 LoRA rank ladder + dp=0.3:")
for r in [64, 128, 256, 512]:
    write_cfg(f"p06_lora_r{r}_dp0p3",
              dict(use_lora=True, lora_r=r, lora_alpha=2*r,
                   lora_dropout=0.3, lr=4e-4))

# weight-decay on LoRA params (== L2-SP-on-LoRA since base is frozen 4-bit)
print("Qwen-32B p=0.6 LoRA r=256 dp=0.3 + weight decay:")
for wd in [1e-3, 1e-2, 1e-1]:
    write_cfg(f"p06_lora_r256_dp0p3_wd{wd:g}",
              dict(use_lora=True, lora_r=256, lora_alpha=512,
                   lora_dropout=0.3, lr=4e-4, weight_decay=wd))

# Longer training
print("Qwen-32B p=0.6 LoRA r=256 dp=0.3, long:")
write_cfg("p06_lora_r256_dp0p3_long",
          dict(use_lora=True, lora_r=256, lora_alpha=512,
               lora_dropout=0.3, lr=4e-4, num_steps=8000, eval_every=400))

print(f"\nWrote {len(list(CFG_DIR.glob('*.yaml')))} configs to {CFG_DIR}")
