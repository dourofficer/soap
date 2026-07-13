#!/usr/bin/env bash
# Run the prompting baselines for deepseek-8b across ALL datasets, sequentially.
#
# Usage:
#   bash baselines/prompting/scripts/run_deepseek.sh [GPU] [extra --set overrides...]
#
# Examples:
#   bash baselines/prompting/scripts/run_deepseek.sh 5
#   bash baselines/prompting/scripts/run_deepseek.sh 5 --set overwrite=true
#
# Runs on one GPU; pair with run_qwen.sh on a different GPU to parallelise the
# two models. Idempotent: already-written predictions are skipped.
set -euo pipefail

MODEL="deepseek-8b"
GPU="${1:-0}"; shift || true          # first arg = GPU index (default 0)
EXTRA=("$@")                          # any further args passed straight to the sweep

# DeepSeek-R1-Distill always emits a <think> block, and gen_max_tokens caps
# thinking + answer combined. 1024 (fine for non-thinking qwen) would be eaten by
# reasoning before the answer is produced, yielding empty/unparsable output. Give
# it enough headroom to finish reasoning AND emit the answer. Override by
# appending your own --set gen_max_tokens=... after the GPU arg.
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

DATASETS=(ww traceelephant correct-error)
# DATASETS=(traceelephant correct-error)

echo "=== ${MODEL} on GPU ${GPU} | datasets: ${DATASETS[*]} ==="
for ds in "${DATASETS[@]}"; do
  log="logs/${ds}_deepseek.log"
  echo ">>> [$(date '+%F %T')] ${MODEL} / ${ds}  (log: ${log})"
  CUDA_VISIBLE_DEVICES="${GPU}" VLLM_LOGGING_LEVEL=WARNING \
    uv run python -m baselines.prompting.sweep \
      --config "baselines/prompting/configs/${ds}.yaml" \
      --set "models=[${MODEL}]" \
      --set "gen_max_tokens=${GEN_MAX_TOKENS}" \
      "${EXTRA[@]}" 2>&1 | tee "${log}"
  echo "<<< [$(date '+%F %T')] done ${MODEL} / ${ds}"
done

echo "=== ${MODEL}: all datasets complete ==="
