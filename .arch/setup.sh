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
# aligne is a private ArcadiaImpact repo — clone with the pod's GH_TOKEN, then pip-install.
if python3 -c "import aligne" 2>/dev/null; then
  echo "[setup.sh] aligne already importable"
elif [ -n "${GH_TOKEN:-}" ]; then
  rm -rf /workspace/aligne
  git clone "https://x-access-token:${GH_TOKEN}@github.com/ArcadiaImpact/aligne.git" /workspace/aligne 2>&1 | tail -2 \
    && pip install --break-system-packages -e /workspace/aligne 2>&1 | tail -2 \
    || echo "WARN: aligne clone/install failed — decisiveness eval will fail."
else
  echo "WARN: no GH_TOKEN — cannot clone private aligne repo. Decisiveness eval will fail."
fi
echo "[setup.sh] done"
