#!/usr/bin/env bash
# Analysis chain across models × datasets: svd → undiscounted tables → rescore →
# discounted tables. One GPU, sequential over models×datasets×subsets. Wraps
# scripts/run_analysis.sh (which shells out to the experiments.* sweep drivers).
#
# Assumes representations + attention already exist for every (model, dataset)
# selected here — run scripts/extraction/* first.
#
#   ./scripts/scoring/run_scoring.sh                                # both models, all 3 datasets, GPU 0
#   MODELS="deepseek-8b" GPU=1 ./scripts/scoring/run_scoring.sh     # one model
#   DATASETS="ww correct-error" ./scripts/scoring/run_scoring.sh    # subset of datasets
#   DRY_RUN=1 ./scripts/scoring/run_scoring.sh                      # preview only
#   STAGES="rescore disc" ./scripts/scoring/run_scoring.sh          # part of the chain
#   EXTRA_SET="--set seeds=[1,2,3]" ./scripts/scoring/run_scoring.sh
#
# Both report builders write per (model, subset) — out_root/<model>/<subset>/ — so
# looping models here yields the same files as one combined run; nothing is
# overwritten. Fails fast: a failing (model, dataset) aborts the whole loop.
set -euo pipefail

MODELS="${MODELS:-qwen3.5-9b deepseek-8b}"
DATASETS="${DATASETS:-ww traceelephant correct-error}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # → scripts/

for model in $MODELS; do
  for ds in $DATASETS; do
    DATASET="$ds" MODELS="$model" GPU="${GPU:-0}" \
      DRY_RUN="${DRY_RUN:-0}" EXTRA_SET="${EXTRA_SET:-}" STAGES="${STAGES:-}" \
      "$SCRIPTS_DIR/run_analysis.sh"
  done
done
