from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .eval_types import StepEvalResult


def append_eval_result(output_file: Path, result: StepEvalResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        existing = json.loads(output_file.read_text(encoding="utf-8"))
    else:
        existing = []
    existing.append(asdict(result))
    output_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def append_residual_step(
    output_file: Path,
    *,
    step: int,
    repeat_id: int,
    layer_idx: int,
    train_residuals: list[dict[str, Any]],
    test_residuals: list[dict[str, Any]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        existing: list[dict[str, Any]] = json.loads(output_file.read_text(encoding="utf-8"))
    else:
        existing = []
    existing.append({
        "step": step,
        "repeat_id": repeat_id,
        "layer_idx": layer_idx,
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
    })
    output_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def read_eval_results(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []
    return json.loads(output_file.read_text(encoding="utf-8"))
