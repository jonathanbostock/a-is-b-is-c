"""ARCH 2.0 held-out eval for the `crystallize-no-cook` task.

Score = held-out test-edge accuracy  x  min(1, decisiveness_FT / decisiveness_base)

The worker submits a *training recipe* (a model-config YAML at
`submission/recipe.yaml`) — NOT a trained model. This script:

  1. Reads the held-out topology spec from $ARCH_DATA_ROOT/heldout.json
     ({seed, train_p, eval_p, k, base_decisiveness, base_model}). Workers
     never see this seed, so they cannot train on the held-out test edges.
  2. Merges the worker's recipe with the held-out topology + base model +
     chat_format, and trains via the tested `pretrained_llms.run` path
     (unless recipe.control == true, which skips training and evaluates the
     base model — used by the arch-init canary to prove the loop cheaply).
  3. Reads final held-out test-edge accuracy from the run's eval_results.json.
  4. Measures mu-decisiveness on the resulting model with the aligne panel
     (transformers-based local adapter — no vLLM).
  5. score = test_acc * min(1, decisiveness / base_decisiveness).

Writes {score, metrics, notes} to $ARCH_EVAL_OUTPUT.

Anti-gaming: a recipe that memorises via full-param FT will crater
decisiveness (retention -> 0 -> score -> 0); a recipe that preserves
decisiveness but never crystallizes leaves test_acc at chance. The maximum
is a method that installs the concept AND keeps the base preference structure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
BASE_EXPERIMENT = "random_6_0.6_0.2_sweep"  # topology generator (train_p overridden by held-out spec)


def _composable(s: int, t: int, train_edges: set[tuple[int, int]]) -> bool:
    return any((s, m) in train_edges and (m, t) in train_edges for m in range(6))


def _read_eval_results(run_dir: Path) -> dict:
    """Parse the last-step held-out test-edge accuracy from a run's eval_results.json."""
    evf = next(run_dir.rglob("eval_results.json"), None)
    if evf is None:
        raise FileNotFoundError(f"no eval_results.json under {run_dir}")
    d = json.load(open(evf))
    rep0 = sorted([r for r in d if r["repeat_id"] == 0], key=lambda r: r["step"])
    last = rep0[-1]
    train_edges = {tuple(e["edge"]) for e in last["train_edges"]}
    cmp_acc = [e["accuracy"] for e in last["test_edges"] if _composable(*e["edge"], train_edges)]
    nc_acc = [e["accuracy"] for e in last["test_edges"] if not _composable(*e["edge"], train_edges)]
    return {
        "train_acc": float(np.mean([e["accuracy"] for e in last["train_edges"]])),
        "test_acc": float(np.mean([e["accuracy"] for e in last["test_edges"]])),
        "composable_acc": float(np.mean(cmp_acc)) if cmp_acc else float("nan"),
        "noncomposable_acc": float(np.mean(nc_acc)) if nc_acc else float("nan"),
        "final_model_dir": str((run_dir / "final")) if (run_dir / "final").exists() else "",
        "run_dir": str(run_dir),
    }


def _measure_decisiveness(model_path: str, tokenizer_path: str, out_dir: Path) -> float:
    """Run the aligne panel on a served-free local model; return decisiveness_fitted."""
    from aligne.metrics.preferences import run_panel, PanelConfig  # noqa: local import (heavy)
    import asyncio
    sys.path.insert(0, str(REPO))
    from pretrained_llms.aligne_local import LocalChatClient  # the adapter built in-session

    async def _go() -> dict:
        client = LocalChatClient(model_path, tokenizer_path=tokenizer_path)
        cfg = PanelConfig(seed=42, n_concepts=155)
        return await run_panel(client, cfg, out_dir)

    panel = asyncio.run(_go())
    return float(panel.get("decisiveness", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("ARCH_DATA_ROOT"))
    ap.add_argument("--output", default=os.environ.get("ARCH_EVAL_OUTPUT"))
    args = ap.parse_args()
    out_path = Path(args.output)

    def emit(score, metrics, notes):
        out_path.write_text(json.dumps({"score": score, "metrics": metrics, "notes": notes}, indent=2))
        print(f"[arch_eval] score={score} notes={notes}")

    try:
        heldout = json.load(open(Path(args.data_root) / "heldout.json"))
    except Exception as exc:
        emit(None, None, f"could not read held-out spec: {exc!r}")
        return

    base_decis = float(heldout.get("base_decisiveness", 0.735))
    base_model = heldout.get("base_model", "Qwen/Qwen2.5-14B-Instruct")

    # Worker submission: a training recipe. Absent/control => evaluate base model.
    recipe_path = REPO / "submission" / "recipe.yaml"
    recipe = {}
    if recipe_path.exists():
        recipe = yaml.safe_load(recipe_path.read_text()) or {}
    control = bool(recipe.get("control", False)) or not recipe_path.exists()

    work = REPO / "runs" / "arch_eval_work"
    work.mkdir(parents=True, exist_ok=True)

    if control:
        # Canary path: no training. Eval the (optionally overridden) model's
        # test-edge accuracy + decisiveness. A recipe model_name override lets the
        # arch-init canary use a tiny chat model (e.g. Qwen2.5-0.5B-Instruct) on a
        # cheap GPU to prove the pipeline without a 14B load.
        canary_model = recipe.get("model_name", base_model)
        cfg = {
            "model_name": canary_model, "max_seq_length": 128,
            "n_repeats": 1, "n_train_templates": 8, "n_eval_templates": 4,
            "num_steps": 1, "eval_every": 1, "skip_train": True,
            "seed": int(heldout["seed"]), "train_p": float(heldout.get("train_p", 0.6)),
            "eval_p": float(heldout.get("eval_p", 0.2)), "k": int(heldout.get("k", 8)),
            "n_categories": 6,
            "chat_format": True, "use_lora": False, "collect_residuals": False,
            "dense_early_evals": False, "eval_subsample": 64,
            "output_dir": str(work / "control"),
        }
        model_for_decis, tok_for_decis = canary_model, canary_model
        note = f"control (no training, model={canary_model})"
    else:
        # Real submission: merge recipe over held-out topology + base model + chat.
        cfg = dict(recipe)
        cfg.update({
            "model_name": recipe.get("model_name", base_model),
            "seed": int(heldout["seed"]),
            "train_p": float(heldout.get("train_p", 0.6)),
            "eval_p": float(heldout.get("eval_p", 0.2)),
            "k": int(heldout.get("k", 8)),
            "n_categories": 6,
            "chat_format": True,
            "save_final": True,
            "collect_residuals": False,
            "output_dir": str(work / "submission"),
        })
        cfg.setdefault("max_seq_length", 128)
        cfg.setdefault("n_repeats", 1)
        note = "submission recipe trained on held-out topology"

    # Write the merged config and run the tested training/eval path.
    cfg_path = work / "arch_run_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir_base = Path(cfg["output_dir"])
    env = dict(os.environ)
    try:
        subprocess.run(
            [sys.executable, "-m", "pretrained_llms.run", str(cfg_path)],
            cwd=str(REPO), env=env, check=True,
        )
    except subprocess.CalledProcessError as exc:
        emit(None, None, f"training/eval run failed: {exc!r}")
        return

    # Find the timestamped run dir under output_dir.
    run_dirs = sorted(run_dir_base.glob("*_llm_*"), key=lambda p: p.stat().st_mtime)
    run_dir = run_dirs[-1] if run_dirs else run_dir_base
    try:
        acc = _read_eval_results(run_dir)
    except Exception as exc:
        emit(None, None, f"could not read eval results: {exc!r}")
        return

    # Decisiveness on the resulting model (control: base model; submission: final/).
    if not control:
        final_dir = acc["final_model_dir"]
        if not final_dir:
            emit(None, acc, "submission produced no saved model (final/ missing)")
            return
        model_for_decis, tok_for_decis = final_dir, base_model  # base tokenizer (local-dir tok bug)

    try:
        decis = _measure_decisiveness(model_for_decis, tok_for_decis, work / "decis")
    except Exception as exc:
        emit(None, acc, f"decisiveness eval failed: {exc!r}")
        return

    retention = min(1.0, decis / base_decis) if base_decis > 0 else 0.0
    score = round(acc["test_acc"] * retention, 4)
    metrics = {
        "test_acc": round(acc["test_acc"], 4),
        "train_acc": round(acc["train_acc"], 4),
        "composable_acc": round(acc["composable_acc"], 4) if acc["composable_acc"] == acc["composable_acc"] else None,
        "noncomposable_acc": round(acc["noncomposable_acc"], 4) if acc["noncomposable_acc"] == acc["noncomposable_acc"] else None,
        "decisiveness": round(decis, 4),
        "decisiveness_retention": round(retention, 4),
    }
    emit(score, metrics, note)


if __name__ == "__main__":
    main()
