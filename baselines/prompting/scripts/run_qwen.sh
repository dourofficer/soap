#!/usr/bin/env bash
# Run the prompting baselines for qwen3.5-9b across ALL datasets, sequentially.
#
# Usage:
#   bash baselines/prompting/scripts/run_qwen.sh [GPU] [extra --set overrides...]
#
# Examples:
#   bash baselines/prompting/scripts/run_qwen.sh 4
#   bash baselines/prompting/scripts/run_qwen.sh 4 --set overwrite=true
#
# Runs on one GPU; pair with run_deepseek.sh on a different GPU to parallelise
# the two models. Idempotent: already-written predictions are skipped.
set -euo pipefail

MODEL="qwen3.5-9b"
GPU="${1:-0}"; shift || true          # first arg = GPU index (default 0)
EXTRA=("$@")                          # any further args passed straight to the sweep

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

DATASETS=(ww traceelephant correct-error)

echo "=== ${MODEL} on GPU ${GPU} | datasets: ${DATASETS[*]} ==="
for ds in "${DATASETS[@]}"; do
  log="logs/${ds}_qwen.log"
  echo ">>> [$(date '+%F %T')] ${MODEL} / ${ds}  (log: ${log})"
  CUDA_VISIBLE_DEVICES="${GPU}" VLLM_LOGGING_LEVEL=WARNING \
    uv run python -m baselines.prompting.sweep \
      --config "baselines/prompting/configs/${ds}.yaml" \
      --set "models=[${MODEL}]" \
      "${EXTRA[@]}" 2>&1 | tee "${log}"
  echo "<<< [$(date '+%F %T')] done ${MODEL} / ${ds}"
done

echo "=== ${MODEL}: all datasets complete ==="
