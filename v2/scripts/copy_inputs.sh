#!/usr/bin/env bash
# Import a dataset's corpus + cached extractions from an external tree into this repo.
#
# Only needed when adopting data produced elsewhere. A from-scratch run instead does:
#   1. put trajectory JSONs in  data/<ds>/<subset>/*.json
#   2. run the extract stage    (scripts/extract.sh) to build outputs/<ds>/{activations,attention}
#
#   SRC=/path/to/legacy DATASET=correct-full ./scripts/copy_inputs.sh
#   SRC=... DATASET=ww MODELS="qwen3.5-9b deepseek-8b" ./scripts/copy_inputs.sh
#   DRY_RUN=1 SRC=... DATASET=correct-full ./scripts/copy_inputs.sh      # rsync -n preview
#
# Expects the source tree to look like:
#   $SRC/data/<ds>/<subset>/*.json
#   $SRC/outputs-<ds>/{activations,attention}/<model>/<subset>/*.safetensors
# Destinations follow src/common/paths.py: data/<ds>/ and outputs/<ds>/{activations,attention}/.
set -euo pipefail

DATASET="${DATASET:?set DATASET, e.g. correct-full}"
SRC="${SRC:?set SRC to the external tree holding data/ and outputs-<ds>/}"
SRC_OUT="$SRC/outputs-$DATASET"
SRC_DATA="$SRC/data/$DATASET"
RSYNC_FLAGS=(-a --info=progress2)
[[ "${DRY_RUN:-0}" == "1" ]] && RSYNC_FLAGS+=(-n)

mkdir -p "data/$DATASET" "outputs/$DATASET/activations" "outputs/$DATASET/attention"

# MODELS defaults to every model dir present in the source activations root.
if [[ -n "${MODELS:-}" ]]; then
  models=($MODELS)
else
  models=($(ls "$SRC_OUT/activations" 2>/dev/null))
fi

echo ">> dataset=$DATASET  models=${models[*]:-<none>}"
for m in "${models[@]}"; do
  for kind in activations attention; do
    if [[ -d "$SRC_OUT/$kind/$m" ]]; then
      echo ">> $kind/$m"
      rsync "${RSYNC_FLAGS[@]}" "$SRC_OUT/$kind/$m/" "outputs/$DATASET/$kind/$m/"
    else
      echo "!! missing $SRC_OUT/$kind/$m (skipped)"
    fi
  done
done

echo ">> corpus -> data/$DATASET/"
rsync "${RSYNC_FLAGS[@]}" --exclude='*.csv' "$SRC_DATA/" "data/$DATASET/"

echo ">> done."
du -sh "data/$DATASET" "outputs/$DATASET/activations" "outputs/$DATASET/attention" 2>/dev/null || true
