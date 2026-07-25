#!/usr/bin/env bash
# Attention mass for deepseek-8b across all datasets (ww, traceelephant, correct-error).
# One GPU, sequential over datasets×subsets. Wraps scripts/extract_attention.sh, which
# shells out to experiments.attention.sweep → src.attention.streaming (data_v2).
#
#   ./scripts/extraction/deepseek-8b_attention.sh              # GPU 0, all 3 datasets
#   GPU=1 ./scripts/extraction/deepseek-8b_attention.sh        # pin a GPU
#   DRY_RUN=1 ./scripts/extraction/deepseek-8b_attention.sh    # preview only
#   DATASETS="ww correct-error" ./scripts/extraction/deepseek-8b_attention.sh
#   EXTRA_SET="--set max_tokens=4096" ./scripts/extraction/deepseek-8b_attention.sh
set -euo pipefail

MODEL="deepseek-8b"
STAGE="extract_attention.sh"
DATASETS="${DATASETS:-ww traceelephant correct-error}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # → scripts/

for ds in $DATASETS; do
  DATASET="$ds" MODELS="$MODEL" GPU="${GPU:-0}" \
    DRY_RUN="${DRY_RUN:-0}" EXTRA_SET="${EXTRA_SET:-}" \
    "$SCRIPTS_DIR/$STAGE"
done
