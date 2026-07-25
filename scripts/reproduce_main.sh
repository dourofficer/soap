#!/usr/bin/env bash
# Reproduce the legacy main results table for a dataset, plus the extended table.
# Assumes the score stage has already run (scores/<tag>/... present) and baseline
# prediction JSONLs are under outputs/<ds>/baselines/ (scripts/copy_inputs.sh).
#
#   DATASET=ww GPU=0 ./scripts/reproduce_main.sh
#   DATASET=correct-error GPU=0 STAGES="faithful extended" ./scripts/reproduce_main.sh
#
# STAGES (default "faithful extended"):
#   faithful — proj-only, pooling=separate, discount-only sweep matching the legacy grid,
#              then the byte-faithful results_table.tsv.
#   extended — native all-methods (joint) reductions already produced by run_pipeline.sh,
#              surfaced (distance scorers + backprop + baselines) into results_extended.tsv.
set -euo pipefail

DATASET="${DATASET:?set DATASET}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
STAGES="${STAGES:-faithful extended}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
C() { echo ">>> $*"; "$@"; }

# Legacy sweep axes (no gamma=0, no score_norm, discount only) so CRR matches v1 exactly.
FAITHFUL_SET=(
  --set variant=proj --set pooling_mode=separate
  --set headline_methods=[proj]
  --set exclude_positions=[ens-mid3]        # the ensemble is a v2 addition; legacy had none
  --set base_table=base_proj_test.tsv
  --set strategies=[discount] --set score_norms=[none]
  --set gammas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
  --set orients=[negate,inverse,sigmoid] --set ws=[1,2,3,4,5,all]
)

if has faithful; then
  C python -m src.reports.reduce  --config "configs/reduce/$DATASET.yaml"  --set stage=base "${FAITHFUL_SET[@]}"
  C python -m src.rescore.run     --config "configs/rescore/$DATASET.yaml" "${FAITHFUL_SET[@]}"
  C python -m src.reports.reduce  --config "configs/reduce/$DATASET.yaml"  --set stage=crr "${FAITHFUL_SET[@]}"
  C python -m src.reports.main_table --config "configs/tables/$DATASET.yaml" --set variant=faithful
fi

if has extended; then
  # native reductions (joint, all methods, discount+backprop) come from run_pipeline.sh;
  # here we only surface them + baselines into the extended grid.
  C python -m src.reports.main_table --config "configs/tables/$DATASET.yaml" --set variant=extended
fi

echo ">>> reproduce_main done: $DATASET"
