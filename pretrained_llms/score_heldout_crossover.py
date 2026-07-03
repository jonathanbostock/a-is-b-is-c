"""Score the SDF-Riemannion x Qwen2.5-14B crossover run on the arch held-out protocol.

Mirrors pretrained_llms/arch_eval.py scoring exactly: same base model the fleet
used (Qwen2.5-14B-Instruct), same base decisiveness constant (0.735) from
heldout.json, so the resulting score slots directly into the run leaderboard.
Optionally re-measures base decisiveness as a sanity check (--measure-base).

Usage (from repo root, after the training run):
    python -m pretrained_llms.score_heldout_crossover \
        --run-dir-base runs/sdf_qwen14b/riemannion_heldout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_llms.arch_eval import _measure_decisiveness, _read_eval_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir-base", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--base-decisiveness", type=float, default=0.735,
                    help="base decisiveness constant from heldout.json (leaderboard-canonical)")
    ap.add_argument("--out", type=Path, default=Path("runs/crossover_score.json"))
    ap.add_argument("--measure-base", action="store_true",
                    help="also re-measure base-model decisiveness as a sanity check")
    args = ap.parse_args()

    base = args.run_dir_base
    run_dirs = sorted(base.parent.glob(base.name + "_llm_*"), key=lambda p: p.stat().st_mtime)
    run_dir = run_dirs[-1] if run_dirs else base
    acc = _read_eval_results(run_dir)
    print(f"[score] run_dir={run_dir}")
    print(f"[score] test_acc={acc['test_acc']:.4f} train_acc={acc['train_acc']:.4f}")

    work = Path("runs/crossover_decis")
    work.mkdir(parents=True, exist_ok=True)

    final_dir = acc.get("final_model_dir")
    if not final_dir:
        raise SystemExit("no final/ model dir found — was save_final set?")
    print(f"[score] measuring FT decisiveness on {final_dir} …")
    d_ft = _measure_decisiveness(final_dir, args.base_model, work / "ft")
    print(f"[score] d_FT = {d_ft:.4f}")

    d_base = args.base_decisiveness
    base_check = None
    if args.measure_base:
        print(f"[score] sanity: measuring BASE decisiveness on {args.base_model} …")
        base_check = _measure_decisiveness(args.base_model, args.base_model, work / "base")
        print(f"[score] measured base = {base_check:.4f} (canonical constant stays {d_base})")

    retention = min(1.0, d_ft / d_base) if d_base > 0 else 0.0
    score = round(acc["test_acc"] * retention, 4)
    result = {
        "score": score,
        "metrics": {
            "test_acc": round(acc["test_acc"], 4),
            "train_acc": round(acc["train_acc"], 4),
            "composable_acc": round(acc["composable_acc"], 4) if acc["composable_acc"] == acc["composable_acc"] else None,
            "noncomposable_acc": round(acc["noncomposable_acc"], 4) if acc["noncomposable_acc"] == acc["noncomposable_acc"] else None,
            "decisiveness": round(d_ft, 4),
            "decisiveness_retention": round(retention, 4),
            "base_decisiveness_measured_check": round(base_check, 4) if base_check is not None else None,
        },
        "notes": "SDF-Riemannion Qwen2.5-14B on arch heldout seed 682050 — directly "
                 "leaderboard-comparable (same model, topology, score formula); "
                 "raw doc-LM training (harness forced chat_format=True on the fleet); "
                 "leaderboard #1 for context: PR #140 = 0.6362.",
        "run_dir": str(run_dir),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
