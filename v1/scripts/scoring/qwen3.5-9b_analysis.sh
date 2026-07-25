#!/usr/bin/env bash
# Analysis chain for qwen3.5-9b across all datasets (ww, traceelephant, correct-error).
# Thin wrapper over scripts/scoring/run_scoring.sh with the model pinned.
#
#   ./scripts/scoring/qwen3.5-9b_analysis.sh              # GPU 0, all 3 datasets
#   GPU=1 ./scripts/scoring/qwen3.5-9b_analysis.sh        # pin a GPU
#   DRY_RUN=1 ./scripts/scoring/qwen3.5-9b_analysis.sh    # preview only
#   DATASETS="ww correct-error" ./scripts/scoring/qwen3.5-9b_analysis.sh
#   STAGES="rescore disc" ./scripts/scoring/qwen3.5-9b_analysis.sh
set -euo pipefail

MODELS="qwen3.5-9b" exec "$(dirname "${BASH_SOURCE[0]}")/run_scoring.sh"
