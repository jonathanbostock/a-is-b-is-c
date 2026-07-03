#!/bin/bash
# Overnight crossover cells (2026-07-03): waits for today's riemannion_heldout
# pipeline to finish (ALL_DONE in /root/setup.log), generates the on-policy
# mixin corpus, runs 6 Riemannion-SDF cells on the arch heldout topology,
# scores each (test_acc x decisiveness retention vs 0.735), pushes adapters
# per-cell (train.py hf_repo_id), uploads the full log/score bundle to the HF
# logs dataset, then self-terminates the pod. Cell failures skip, never abort.
set -uo pipefail
exec > >(tee -a /root/overnight.log) 2>&1
set -a; source /root/.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/a-is-b-is-c

echo "=== OVERNIGHT_WAITING $(date -u) ==="
until grep -q "ALL_DONE" /root/setup.log 2>/dev/null; do sleep 300; done
echo "=== OVERNIGHT_STARTED $(date -u) commit=$(git rev-parse --short HEAD) ==="

# 0. On-policy mixin corpus (GPU free after today's panel; needed by c5/c6)
if [ ! -s /root/onpolicy_mixin.jsonl ]; then
  python3 -m pretrained_llms.gen_onpolicy_mixin \
    --model Qwen/Qwen2.5-14B-Instruct --n 1200 --out /root/onpolicy_mixin.jsonl \
    || echo "WARN: onpolicy generation failed — c5/c6 will fail their mixin load"
fi

CELLS="c1_mlponly c2_mlponly_wd1e2 c3_mlponly_wd3e3 c4_allmod_wd1e2 c5_mlponly_mixin c6_mlponly_wd1e2_mixin"
for C in $CELLS; do
  echo "=== CELL_START $C $(date -u) ==="
  python3 -m pretrained_llms.run "pretrained_llms/configs/sdf_qwen14b/$C.yaml" > "/root/cell_$C.log" 2>&1 \
    || { echo "=== CELL_TRAIN_FAILED $C $(date -u) ==="; tail -5 "/root/cell_$C.log"; continue; }
  python3 -m pretrained_llms.score_heldout_crossover \
    --run-dir-base "runs/sdf_qwen14b/$C" --out "/root/score_$C.json" > "/root/scorelog_$C.log" 2>&1 \
    || { echo "=== CELL_SCORE_FAILED $C $(date -u) ==="; tail -5 "/root/scorelog_$C.log"; continue; }
  echo "=== CELL_DONE $C $(date -u) ==="
  head -c 600 "/root/score_$C.json"; echo
done

echo "=== UPLOADING $(date -u) ==="
mkdir -p /root/bundle/runs
cp /root/cell_*.log /root/scorelog_*.log /root/score_*.json /root/overnight.log /root/bundle/ 2>/dev/null || true
cp /root/crossover_score.json /root/train.log /root/score.log /root/setup.log /root/bundle/ 2>/dev/null || true
cp /root/onpolicy_mixin.jsonl /root/bundle/ 2>/dev/null || true
find runs/sdf_qwen14b \( -name "eval_results.json" -o -name "run_config_metadata.json" \) | while read -r f; do
  rel=$(dirname "$f" | sed "s|runs/sdf_qwen14b/||")
  mkdir -p "/root/bundle/runs/$rel"
  cp "$f" "/root/bundle/runs/$rel/"
done
cp pretrained_llms/configs/sdf_qwen14b/*.yaml /root/bundle/ 2>/dev/null || true
git rev-parse HEAD > /root/bundle/GIT_SHA
python3 - << 'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="/root/bundle",
                  repo_id="arcadia-impact/matching-game-scale-logs",
                  repo_type="dataset",
                  path_in_repo="sdf-crossover-overnight-20260703",
                  commit_message="overnight crossover cells: logs+scores+eval curves (pod self-upload at teardown)")
print("BUNDLE_UPLOADED")
PY
echo "=== UPLOAD_DONE $(date -u) ==="

# Pod deletion is handled OFF-POD by .github/workflows/crossover-pod-reaper.yml
# (fires when the bundle's GIT_SHA lands on HF, or after the Sunday deadline).
# The master RunPod key deliberately never touches this pod.
echo "=== ALL_CELLS_DONE $(date -u) — awaiting reaper ==="
