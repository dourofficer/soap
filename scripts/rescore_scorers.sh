#!/usr/bin/env bash
# Per-scorer rescoring for the EXTENDED main table: apply BOTH strategies (discount +
# backprop) to EACH base scorer, not just to the joint winner. For each scorer it runs a
# scorer-pinned base reduction + rescore sweep + CRR/backprop reduction under the variant
# tag ext_<scorer> (joint pooling), then builds results_extended.tsv.
#
# Assumes the score stage has already run (scores/<tag>/... present) and baseline
# prediction JSONLs are under outputs/<ds>/baselines/.
#
#   DATASET=ww GPU=0 ./scripts/rescore_scorers.sh
#   DATASET=correct-error GPU=1 SCORERS="proj angres resid" ./scripts/rescore_scorers.sh
#
# Knobs: DATASET (required), GPU (default 0), SCORERS (default "proj angres resid").
set -euo pipefail

DATASET="${DATASET:?set DATASET}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SCORERS="${SCORERS:-proj angres resid}"
C() { echo ">>> $*"; "$@"; }

for s in $SCORERS; do
  echo "===== scorer: $s (variant ext_$s, joint pooling, discount + backprop) ====="
  V=(--set variant=ext_$s --set headline_methods=[$s] --set base_table=base_ext_${s}_test.tsv)
  C python -m src.reports.reduce  --config "configs/reduce/$DATASET.yaml"  --set stage=base "${V[@]}"
  C python -m src.rescore.run     --config "configs/rescore/$DATASET.yaml" "${V[@]}"
  C python -m src.reports.reduce  --config "configs/reduce/$DATASET.yaml"  --set stage=crr "${V[@]}"
done

C python -m src.reports.main_table --config "configs/tables/$DATASET.yaml" --set variant=extended
echo ">>> rescore_scorers done: $DATASET  (outputs/$DATASET/tables/325/results_extended.tsv)"
