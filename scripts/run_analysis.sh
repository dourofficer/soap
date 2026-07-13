#!/usr/bin/env bash
# Run the analysis chain for a dataset, in order:
#   svd → reports/undiscounted_v2 → rescore → reports/discounted_v2
# svd + rescore use the GPU; the two report builders are pure CPU.
# Assumes representations + attention already exist (run gen_embeddings.sh and
# extract_attention.sh first).
#
# Usage (override knobs via env):
#   DATASET=correct-error MODELS=qwen3.5-9b GPU=0 ./scripts/run_analysis.sh
#   DRY_RUN=1 ./scripts/run_analysis.sh                        # preview commands
#   STAGES="rescore disc" ./scripts/run_analysis.sh           # re-run only part of the chain
#   EXTRA_SET="--set poolings=[last] --set seeds=[1,2,3]" ./scripts/run_analysis.sh
#   SVD_CONFIG=experiments/svd/configs/226.yaml ./scripts/run_analysis.sh   # split-variant reuse
#
# Shared knobs (DATASET, MODELS, GPU, DRY_RUN, EXTRA_SET): see scripts/_common.sh.
# STAGES selects which stages run: any subset/order of `svd undisc rescore disc reproduce`.
# `reproduce` is OFF by default (not in the default STAGES); add it to validate/apply
# the reduced tables' best configs, or use scripts/reproduce.sh directly.
#
# NOTE: EXTRA_SET (e.g. `--set subsets=[gaia]`) is forwarded only to svd + rescore.
# The report builders read `subsets` from their own config, so to narrow a report
# stage, point UNDISC_CONFIG / DISC_CONFIG at a config with the subset(s) you want.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGES="${STAGES:-svd undisc rescore disc}"
export CUDA_VISIBLE_DEVICES="$GPU"

has() { [[ " $STAGES " == *" $1 "* ]]; }

# 1. SVD scoring (GPU) — writes weighted projections per (model, subset, pooling, seed).
if has svd; then run python -m experiments.svd.sweep \
    --config "${SVD_CONFIG:-experiments/svd/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
fi

# 2. Undiscounted tables (CPU) — reduces svd sweep to best per-config base scores.
#    Reads svd_root; must run before rescore, which consumes its output.
if has undisc; then run python -m experiments.reports.build_undiscounted_tables_v2 \
    --config "${UNDISC_CONFIG:-experiments/reports/configs/undiscounted_v2_$DATASET.yaml}" \
    $(models_flag)
fi

# 3. Rescoring / CRR discount sweep (GPU) — writes discounted sweep tables.
if has rescore; then run python -m experiments.rescore.sweep \
    --config "${RESCORE_CONFIG:-experiments/rescore/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
fi

# 4. Discounted tables (CPU) — reduces the rescore sweep to final per-cell tables.
if has disc; then run python -m experiments.reports.build_discounted_tables_v2 \
    --config "${DISC_CONFIG:-experiments/reports/configs/discounted_v2_$DATASET.yaml}" \
    $(models_flag)
fi

# 5. Reproduce (GPU, optional) — re-run the full pipeline for the reduced tables'
#    best configs (validate against the table, or apply with --set split=all).
if has reproduce; then run python -m experiments.reproduce.run \
    --config "${REPRO_CONFIG:-experiments/reproduce/configs/$DATASET.yaml}" \
    $(models_flag) $EXTRA_SET
fi
