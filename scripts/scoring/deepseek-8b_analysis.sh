#!/usr/bin/env bash
# Analysis chain for deepseek-8b across all datasets (ww, traceelephant, correct-error).
# Thin wrapper over scripts/scoring/run_scoring.sh with the model pinned.
#
#   ./scripts/scoring/deepseek-8b_analysis.sh              # GPU 0, all 3 datasets
#   GPU=1 ./scripts/scoring/deepseek-8b_analysis.sh        # pin a GPU
#   DRY_RUN=1 ./scripts/scoring/deepseek-8b_analysis.sh    # preview only
#   DATASETS="ww correct-error" ./scripts/scoring/deepseek-8b_analysis.sh
#   STAGES="rescore disc" ./scripts/scoring/deepseek-8b_analysis.sh
set -euo pipefail

MODELS="deepseek-8b" exec "$(dirname "${BASH_SOURCE[0]}")/run_scoring.sh"
