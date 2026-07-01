#!/usr/bin/env bash
# Dependency setup for worker + held-out eval pods.
#
# CRITICAL: the base image (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404) ships
# a WORKING torch 2.8 + cu128 matching the driver. Do NOT run `uv sync` — pyproject
# pins torch>=2.10, which resolves to a cu13 build the driver can't run (this trap
# cost hours; see the llm-finetune-perf skill). Install every dep EXCEPT torch into
# the SYSTEM python (the one the eval uses) and never touch torch. One python, no
# venv — avoids the venv/system dep-split that made matplotlib/aligne go missing.
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0   # base image sets =1 but the pkg may be absent

PIP="pip install --break-system-packages -q"
$PIP \
  "transformers>=4.57" "peft>=0.15" "bitsandbytes>=0.49" "accelerate>=1.12" \
  "liger-kernel>=0.5.8" "datasets>=4.6" "matplotlib" "seaborn" "networkx" \
  "pyyaml" "numpy" "scipy" "hf_transfer" 2>&1 | tail -2

# aligne (private ArcadiaImpact repo) — clone with the pod's GH_TOKEN into the SAME
# system python the eval uses. Skip if already importable.
if python3 -c "import aligne" 2>/dev/null; then
  echo "[setup.sh] aligne already importable"
elif [ -n "${GH_TOKEN:-}" ]; then
  rm -rf /workspace/aligne
  git clone "https://x-access-token:${GH_TOKEN}@github.com/ArcadiaImpact/aligne.git" /workspace/aligne 2>&1 | tail -1 \
    && $PIP -e /workspace/aligne 2>&1 | tail -1 \
    || echo "WARN: aligne clone/install failed — decisiveness eval will fail."
else
  echo "WARN: no GH_TOKEN — cannot clone private aligne. Decisiveness eval will fail."
fi
python3 -c "import transformers, matplotlib, aligne, peft; print('[setup.sh] deps OK (system python)')" || echo "WARN: dep import check failed"
echo "[setup.sh] done"
