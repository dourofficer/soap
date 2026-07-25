#!/usr/bin/env bash
# Extract per-step attention mass into predecessor steps for a dataset.
# Wraps `experiments.attention.sweep` — one GPU, sequential over MODELS×subsets.
#
# Usage (override knobs via env):
#   DATASET=correct-error MODELS=qwen3.5-9b GPU=0 ./scripts/extract_attention.sh
#   DRY_RUN=1 ./scripts/extract_attention.sh                   # preview commands
#   EXTRA_SET="--set subsets=[gaia] --set max_tokens=8192" ./scripts/extract_attention.sh
#   ATT_CONFIG=experiments/attention/configs/other.yaml ./scripts/extract_attention.sh
#
# See scripts/_common.sh for the shared knobs (DATASET, MODELS, GPU, DRY_RUN, EXTRA_SET).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="$GPU"

run python -m experiments.attention.sweep \
    --config "${ATT_CONFIG:-experiments/attention/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
