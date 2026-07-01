"""Generate config files for the Pythia-70M LoRA-rank + regularization sweep."""
from __future__ import annotations
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "pretrained_llms" / "configs" / "sweep_70m"
CFG_DIR.mkdir(parents=True, exist_ok=True)

# Base template — what we use for all sweeps unless overridden
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
    cfg["output_dir"] = f"./runs/sweep_70m/{name}"
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
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.name}")


# ── Rank sweep ───────────────────────────────────────────────────────────────
# alpha = 2 * r is a common default in LoRA literature (keeps eff-LR roughly
# constant across rank, since the LoRA update is scaled by alpha/r).
print("Rank sweep:")
for r in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    write_cfg(f"rank{r:03d}", dict(use_lora=True, lora_r=r, lora_alpha=2 * r,
                                     lora_dropout=0.0, weight_decay=0.0))

# ── Full-rank baseline ───────────────────────────────────────────────────────
print("Full-rank baseline:")
write_cfg("fullrank", dict(use_lora=False, lora_r=1, lora_alpha=16,
                            lora_dropout=0.0, weight_decay=0.0, lr=2e-4))

# ── Regularization sweep (at r=16) ───────────────────────────────────────────
print("Regularization sweep (r=16):")
for wd in [0.0, 0.01, 0.1]:
    for dp in [0.0, 0.1, 0.3]:
        name = f"reg_wd{wd}_dp{dp}".replace(".", "p")
        write_cfg(name, dict(use_lora=True, lora_r=16, lora_alpha=32,
                              lora_dropout=dp, weight_decay=wd))

# ── alpha/r sweep at r=16 (effective-LR via alpha/r) ─────────────────────────
print("Alpha/r sweep (r=16):")
for a in [4, 16, 32, 64, 128]:
    write_cfg(f"alpha{a:03d}_r16", dict(use_lora=True, lora_r=16, lora_alpha=a,
                                          lora_dropout=0.0, weight_decay=0.0))

print("\nWrote configs to", CFG_DIR)
