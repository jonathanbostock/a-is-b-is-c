#!/usr/bin/env bash
# Dependency setup for worker pods and the held-out eval pod.
# The base image is runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (torch 2.8
# cu128 — matches the driver; do NOT let anything pull a cu13 torch).
set -euo pipefail

# Project + training deps (torch already in the image).
if command -v uv >/dev/null 2>&1; then
  uv sync --no-install-project 2>/dev/null || true
fi
# Pin cu128 torch backend so nothing upgrades torch to a cu13 build the driver
# can't run (this bit us during setup — see llm-finetune-perf skill).
pip install --break-system-packages \
  "transformers>=4.57" "peft>=0.15" "bitsandbytes>=0.49" "accelerate>=1.12" \
  "liger-kernel>=0.5.8" "datasets>=4.6" "pyyaml" "numpy" "scipy" 2>&1 | tail -3

# aligne (decisiveness panel). Cloned alongside the repo on the pod, or installed
# from the arcadia-impact remote. The scorer imports aligne.metrics.preferences.
if [ -d /workspace/aligne ]; then
  pip install --break-system-packages -e /workspace/aligne 2>&1 | tail -2
elif [ -d ../aligne ]; then
  pip install --break-system-packages -e ../aligne 2>&1 | tail -2
else
  pip install --break-system-packages "git+https://github.com/ArcadiaImpact/aligne.git" 2>&1 | tail -2 || \
    echo "WARN: aligne not installed — decisiveness eval will fail. Ensure aligne is available on the pod."
fi
echo "[setup.sh] done"
