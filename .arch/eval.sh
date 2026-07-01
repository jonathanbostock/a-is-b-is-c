#!/bin/bash
# Eval shim for crystallize-no-cook. Same shim runs on worker pods (public
# topology) and the held-out CI pod (secret topology); only ARCH_DATA_ROOT
# differs. Delegates to pretrained_llms/arch_eval.py, which reads the topology
# spec from $ARCH_DATA_ROOT/heldout.json, trains the worker's submitted recipe
# (submission/recipe.yaml), then scores test-edge accuracy x decisiveness
# retention and writes {score,metrics,notes} to $ARCH_EVAL_OUTPUT.
set -euo pipefail
: "${ARCH_DATA_ROOT:?ARCH_DATA_ROOT must be set}"
: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" -m pretrained_llms.arch_eval --data-root "$ARCH_DATA_ROOT" --output "$ARCH_EVAL_OUTPUT"
