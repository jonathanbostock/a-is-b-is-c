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
# setup.sh installs all deps into the SYSTEM python (no venv — avoids the
# venv/system dep-split). Use python3 directly and disable hf_transfer's hard
# requirement in case the base image enabled it.
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 -m pretrained_llms.arch_eval --data-root "$ARCH_DATA_ROOT" --output "$ARCH_EVAL_OUTPUT"
