#!/usr/bin/env bash
# Activations for qwen3.5-9b across all datasets (ww, traceelephant, correct-error).
# One GPU, sequential over datasets×subsets. Wraps scripts/gen_embeddings.sh, which
# shells out to experiments.activations.sweep → src.activations.extract (data_v2).
#
#   ./scripts/extraction/qwen3.5-9b_activations.sh              # GPU 0, all 3 datasets
#   GPU=1 ./scripts/extraction/qwen3.5-9b_activations.sh        # pin a GPU
#   DRY_RUN=1 ./scripts/extraction/qwen3.5-9b_activations.sh    # preview only
#   DATASETS="ww correct-error" ./scripts/extraction/qwen3.5-9b_activations.sh
#   EXTRA_SET="--set max_tokens=4096" ./scripts/extraction/qwen3.5-9b_activations.sh
set -euo pipefail

MODEL="qwen3.5-9b"
STAGE="gen_embeddings.sh"
DATASETS="${DATASETS:-ww traceelephant correct-error}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # → scripts/

for ds in $DATASETS; do
  DATASET="$ds" MODELS="$MODEL" GPU="${GPU:-0}" \
    DRY_RUN="${DRY_RUN:-0}" EXTRA_SET="${EXTRA_SET:-}" \
    "$SCRIPTS_DIR/$STAGE"
done
