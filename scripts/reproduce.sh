#!/usr/bin/env bash
# Reproduce/apply a frozen optimal CRR config for a dataset.
# Reads the reduced discounted table (best per pooling/seed) and re-runs the full
# SVD→orient→discount pipeline via experiments.reproduce.run — either validating
# the metrics against the table (split=test/val) or applying the config to score
# every trajectory (split=all).
#
# Assumes the analysis chain has produced the reduced tables (scripts/run_analysis.sh).
#
# Usage (override knobs via env):
#   DATASET=correct-error GPU=0 ./scripts/reproduce.sh
#   EXTRA_SET="--set split=all" ./scripts/reproduce.sh              # apply mode
#   EXTRA_SET="--set select=explicit" ./scripts/reproduce.sh        # single config
#   DRY_RUN=1 ./scripts/reproduce.sh
#
# See scripts/_common.sh for the shared knobs (DATASET, MODELS, GPU, DRY_RUN, EXTRA_SET).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="$GPU"

run python -m experiments.reproduce.run \
    --config "${REPRO_CONFIG:-experiments/reproduce/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
