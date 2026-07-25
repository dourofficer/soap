#!/usr/bin/env bash
# Generate per-step representations (activations) for a dataset.
# Wraps `experiments.activations.sweep` — one GPU, sequential over MODELS×subsets.
#
# Usage (override knobs via env):
#   DATASET=correct-error MODELS=qwen3.5-9b GPU=0 ./scripts/gen_embeddings.sh
#   DRY_RUN=1 ./scripts/gen_embeddings.sh                      # preview commands
#   EXTRA_SET="--set subsets=[gaia] --set max_tokens=4096" ./scripts/gen_embeddings.sh
#   ACT_CONFIG=experiments/activations/configs/other.yaml ./scripts/gen_embeddings.sh
#
# See scripts/_common.sh for the shared knobs (DATASET, MODELS, GPU, DRY_RUN, EXTRA_SET).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="$GPU"

run python -m experiments.activations.sweep \
    --config "${ACT_CONFIG:-experiments/activations/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
