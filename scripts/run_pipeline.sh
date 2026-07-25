#!/usr/bin/env bash
# Full analysis chain for a dataset, in order. Run from v2/.
#   score -> reduce(base) -> rescore -> reduce(crr) -> tables
#
#   DATASET=correct-full GPU=0 ./scripts/run_pipeline.sh
#   DATASET=correct-full STAGES="rescore reduce_crr tables" ./scripts/run_pipeline.sh
#   DRY_RUN=1 DATASET=correct-full ./scripts/run_pipeline.sh
#
# Knobs: DATASET (required), GPU (default 0), STAGES (default all), MODEL/SUBSET/SEED
# (narrowing, forwarded to every stage), EXTRA_SET (extra --set, forwarded), FORCE=1.
# Assumes inputs already copied (scripts/copy_inputs.sh).
set -euo pipefail

DATASET="${DATASET:?set DATASET}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
STAGES="${STAGES:-score reduce_base rescore reduce_crr tables}"

flags=()
[[ -n "${MODEL:-}" ]]  && flags+=(--model "$MODEL")
[[ -n "${SUBSET:-}" ]] && flags+=(--subset "$SUBSET")
[[ -n "${SEED:-}" ]]   && flags+=(--seed "$SEED")
[[ "${DRY_RUN:-0}" == "1" ]] && flags+=(--dry-run)
[[ "${FORCE:-0}" == "1" ]]   && flags+=(--force)
[[ -n "${EXTRA_SET:-}" ]] && flags+=($EXTRA_SET)

has() { [[ " $STAGES " == *" $1 "* ]]; }
C() { echo ">>> $*"; "$@"; }

has score      && C python -m src.score.run   --config "configs/score/$DATASET.yaml" "${flags[@]}"
has reduce_base && C python -m src.reports.reduce --config "configs/reduce/$DATASET.yaml" --set stage=base "${flags[@]}"
has rescore    && C python -m src.rescore.run  --config "configs/rescore/$DATASET.yaml" "${flags[@]}"
has reduce_crr && C python -m src.reports.reduce --config "configs/reduce/$DATASET.yaml" --set stage=crr "${flags[@]}"
has tables     && C python -m src.reports.tables --config "configs/tables/$DATASET.yaml" "${flags[@]}"
echo ">>> pipeline done: $DATASET"
