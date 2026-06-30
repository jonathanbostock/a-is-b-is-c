"""Batch 3b — Qwen-32B with conservative LR + dropout (3a crashed at step 200).

Strategy: longer warmup, lower LR, smaller dropout to avoid the spike-crash
seen in 3a. Lower priority on weight decay variants — focus on rank + LR.
"""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_qwen32b_v2"
CFG_DIR.mkdir(parents=True, exist_ok=True)

BASE = dict(
    model_name="Qwen/Qwen2.5-32B",
    max_seq_length=96,
    n_repeats=1,
    n_train_templates=8,
    n_eval_templates=4,
    num_steps=2000,
    eval_every=200,
    warmup_ratio=0.10,   # was 0.03 — longer warmup
    batch_size=4,
    grad_accum=4,
    seed=42,
    skip_train=False,
    gradient_checkpointing=True,
    attn_implementation="sdpa",
    dense_early_evals=False,
    collect_residuals=False,
    eval_subsample=64,    # smaller eval for speed
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    load_in_4bit=True,
    max_grad_norm=1.0,
)

def write_cfg(name: str, overrides: dict) -> None:
    cfg = dict(BASE); cfg.update(overrides)
    cfg["output_dir"] = f"./runs/sweep_qwen32b_v2/{name}"
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


# Most conservative — see if it learns at all
print("Qwen-32B p=0.6, conservative LR + low dropout, rank ladder:")
for r in [128, 256, 512]:
    write_cfg(f"r{r}_lr1e-4",
              dict(use_lora=True, lora_r=r, lora_alpha=2*r,
                   lora_dropout=0.1, lr=1e-4))

# Mid LR (matches what worked at 1.4B)
print("Qwen-32B p=0.6, mid LR + low dropout, rank ladder:")
for r in [256, 512]:
    write_cfg(f"r{r}_lr2e-4",
              dict(use_lora=True, lora_r=r, lora_alpha=2*r,
                   lora_dropout=0.1, lr=2e-4))

# No dropout, just to isolate that variable
write_cfg("r256_lr1e-4_dp0",
          dict(use_lora=True, lora_r=256, lora_alpha=512,
               lora_dropout=0.0, lr=1e-4))

print(f"\nWrote {len(list(CFG_DIR.glob('*.yaml')))} configs to {CFG_DIR}")
