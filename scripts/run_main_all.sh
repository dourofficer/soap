#!/usr/bin/env bash
# Full main/ sweep + select for every dataset, both GT settings.
#
#   GPU=4 bash scripts/run_main_all.sh
#   DATASETS="ww" GPU=4 bash scripts/run_main_all.sh
#
# Idempotent: a cell whose sweep.tsv exists is skipped (FORCE=1 to redo).
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
DATASETS="${DATASETS:-ww traceelephant correct-error}"
PY=./.venv/bin/python
FORCE_FLAG=""
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

fail=0
for ds in $DATASETS; do
  for gt in false true; do
    tag="$([[ $gt == true ]] && echo with-GT || echo no-GT)"
    echo "############ $ds / $tag ############"
    $PY -m main sweep --config "configs-main/${ds}.yaml" --set "gt=${gt}" $FORCE_FLAG \
      2>&1 | grep -avE "it/s\]|s/it\]" || fail=1
    $PY -m main select --config "configs-main/${ds}.yaml" --set "gt=${gt}" \
      2>&1 | grep -avE "it/s\]|s/it\]" || fail=1
  done
done
echo "############ ALL DONE (fail=$fail) ############"
exit $fail
