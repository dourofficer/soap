#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="outputs-1006/discounted-splits/reduced"
OUT_DIR="outputs-1006/discounted-splits/agg"

MODELS=(deepseek-8b llama-3.1-8b qwen3-8b qwen3-14b)
SUBSETS=(algorithm-generated hand-crafted)

mkdir -p "$OUT_DIR"

for model in "${MODELS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        echo "=== ${model} / ${subset} ==="
        python -m experiments.reports.aggregate_splits \
            --results-dir "$RESULTS_DIR" \
            --model "$model" \
            --subset "$subset" \
            --out "${OUT_DIR}/${model}__${subset}.tsv"
    done
done