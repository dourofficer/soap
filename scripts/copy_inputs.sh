#!/usr/bin/env bash
# Import a dataset's reusable MODEL-OUTPUT artifacts from an external tree (default: the
# archived v1/). Reproduction reuses only these deterministic artifacts; everything
# derived (scores, reduced tables, results tables) is re-run by this repo's pipeline.
#
# Copies (rsync):
#   corpus            $SRC/data/<ds>/<subset>/*.json          -> data/<ds>/
#   activations       $SRC/outputs-<ds>/activations/...        -> outputs/<ds>/activations/
#   attention         $SRC/outputs-<ds>/attention/...          -> outputs/<ds>/attention/
#   baseline preds    $SRC/outputs-<ds>/{prompting,chief,correct}/<model>/<subset>/
#                       predictions_method-*.jsonl             -> outputs/<ds>/baselines/<baseline>/...
#
#   DATASET=correct-error ./scripts/copy_inputs.sh                 # SRC defaults to v1
#   SRC=/some/tree DATASET=ww MODELS="qwen3.5-9b deepseek-8b" ./scripts/copy_inputs.sh
#   WHAT="baselines" DATASET=ww ./scripts/copy_inputs.sh           # only baseline JSONLs
#   DRY_RUN=1 DATASET=correct-error ./scripts/copy_inputs.sh       # rsync -n preview
#
# WHAT selects which artifacts to copy (default "corpus reps attn baselines").
set -euo pipefail

DATASET="${DATASET:?set DATASET, e.g. correct-error}"
SRC="${SRC:-v1}"                       # the archived implementation holds the artifacts
SRC_OUT="$SRC/outputs-$DATASET"
SRC_DATA="$SRC/data/$DATASET"
WHAT="${WHAT:-corpus reps attn baselines}"
BASELINES="${BASELINES:-prompting chief correct}"
RSYNC_FLAGS=(-a --info=progress2)
[[ "${DRY_RUN:-0}" == "1" ]] && RSYNC_FLAGS+=(-n)
has() { [[ " $WHAT " == *" $1 "* ]]; }

# models default to every dir present under the source activations root
if [[ -n "${MODELS:-}" ]]; then models=($MODELS)
else models=($(ls "$SRC_OUT/activations" 2>/dev/null || true)); fi
echo ">> dataset=$DATASET  src=$SRC  models=${models[*]:-<none>}  what='$WHAT'"

if has corpus; then
  echo ">> corpus -> data/$DATASET/"
  mkdir -p "data/$DATASET"
  rsync "${RSYNC_FLAGS[@]}" --exclude='*.csv' "$SRC_DATA/" "data/$DATASET/"
fi

for m in "${models[@]}"; do
  for kind in activations attention; do
    key=reps; [[ "$kind" == attention ]] && key=attn
    has "$key" || continue
    if [[ -d "$SRC_OUT/$kind/$m" ]]; then
      echo ">> $kind/$m"
      mkdir -p "outputs/$DATASET/$kind/$m"
      rsync "${RSYNC_FLAGS[@]}" "$SRC_OUT/$kind/$m/" "outputs/$DATASET/$kind/$m/"
    else
      echo "!! missing $SRC_OUT/$kind/$m (skipped)"
    fi
  done
done

if has baselines; then
  for b in $BASELINES; do
    [[ -d "$SRC_OUT/$b" ]] || { echo "!! no $b baselines for $DATASET (skipped)"; continue; }
    echo ">> baselines/$b"
    mkdir -p "outputs/$DATASET/baselines/$b"
    # copy only the prediction JSONLs, preserving <model>/<subset>/ layout
    rsync "${RSYNC_FLAGS[@]}" --include='*/' --include='predictions_method-*.jsonl' \
          --exclude='*' "$SRC_OUT/$b/" "outputs/$DATASET/baselines/$b/"
  done
fi

echo ">> done."
du -sh "data/$DATASET" "outputs/$DATASET"/{activations,attention,baselines} 2>/dev/null || true
