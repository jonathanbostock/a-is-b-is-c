"""Score the SDF-Riemannion x Qwen2.5-32B crossover run on the arch held-out protocol.

Mirrors pretrained_llms/arch_eval.py scoring exactly, with one addition: the
base decisiveness is MEASURED on the actual base model (Qwen2.5-32B-Instruct)
rather than taken from heldout.json (whose 0.735 is Qwen2.5-14B-Instruct's).
Retention vs 14B's 0.735 is also reported for reference.

Usage (from repo root, after the training run):
    python -m pretrained_llms.score_heldout_crossover \
        --run-dir-base runs/sdf_qwen32b/riemannion_heldout \
        --base-model Qwen/Qwen2.5-32B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_llms.arch_eval import _measure_decisiveness, _read_eval_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir-base", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--ref-base-decisiveness", type=float, default=0.735,
                    help="14B base decisiveness from heldout.json, for reference only")
    ap.add_argument("--out", type=Path, default=Path("runs/crossover_score.json"))
    ap.add_argument("--skip-base", action="store_true",
                    help="reuse a previously measured base decisiveness from --base-decis")
    ap.add_argument("--base-decis", type=float, default=None)
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

    if args.skip_base and args.base_decis is not None:
        d_base = args.base_decis
        print(f"[score] using provided base decisiveness {d_base:.4f}")
    else:
        print(f"[score] measuring BASE decisiveness on {args.base_model} …")
        d_base = _measure_decisiveness(args.base_model, args.base_model, work / "base")
    print(f"[score] d_base(32B) = {d_base:.4f}")

    retention = min(1.0, d_ft / d_base) if d_base > 0 else 0.0
    score = round(acc["test_acc"] * retention, 4)
    ref_retention = min(1.0, d_ft / args.ref_base_decisiveness)
    result = {
        "score": score,
        "metrics": {
            "test_acc": round(acc["test_acc"], 4),
            "train_acc": round(acc["train_acc"], 4),
            "composable_acc": round(acc["composable_acc"], 4) if acc["composable_acc"] == acc["composable_acc"] else None,
            "noncomposable_acc": round(acc["noncomposable_acc"], 4) if acc["noncomposable_acc"] == acc["noncomposable_acc"] else None,
            "decisiveness": round(d_ft, 4),
            "base_decisiveness_32b_measured": round(d_base, 4),
            "decisiveness_retention": round(retention, 4),
            "retention_vs_14b_0p735_reference": round(ref_retention, 4),
            "score_vs_14b_reference_base": round(acc["test_acc"] * ref_retention, 4),
        },
        "notes": "SDF-Riemannion Qwen2.5-32B on arch heldout seed 682050; "
                 "raw doc-LM training (harness would have forced chat_format=True); "
                 "arch leaderboard #1 for context: PR #140 = 0.6362 on 14B.",
        "run_dir": str(run_dir),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
