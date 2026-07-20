#!/usr/bin/env bash
# Extract per-step representations + attention mass for a dataset. Run from v2/.
#
#   DATASET=correct-full GPU=0 ./scripts/extract.sh
#   DATASET=ww MODELS="qwen3.5-9b" SUBSETS="hand-crafted" GPU=1 ./scripts/extract.sh
#   DRY_RUN=1 DATASET=correct-full ./scripts/extract.sh          # print commands only
#   STAGES=activations DATASET=ww ./scripts/extract.sh           # one stage only
#
# Reads the corpus from data/<ds>/<subset>/*.json and the model registry from the
# manifest (configs/datasets/<ds>.yaml: models, model_paths, max_tokens). Writes
# outputs/<ds>/{activations,attention}/<model>/<subset>/. This is the only GPU-heavy
# stage; everything downstream reads its cached output. Both extractors skip
# trajectories whose .safetensors already exists, so re-running resumes.
set -euo pipefail

DATASET="${DATASET:?set DATASET, e.g. correct-full}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
STAGES="${STAGES:-activations attention}"
MANIFEST="configs/datasets/$DATASET.yaml"

# Pull models / subsets / model_paths / max_tokens out of the manifest so this script
# never duplicates the registry.
read_manifest() {
  python - "$MANIFEST" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
key = sys.argv[2]
if key == "model_paths":
    print("\n".join(f"{k}\t{v}" for k, v in cfg["model_paths"].items()))
elif isinstance(cfg[key], list):
    print(" ".join(str(x) for x in cfg[key]))
else:
    print(cfg[key])
PY
}

models=(${MODELS:-$(read_manifest models)})
subsets=(${SUBSETS:-$(read_manifest subsets)})
max_tokens="$(read_manifest max_tokens)"
declare -A MPATH
while IFS=$'\t' read -r k v; do MPATH["$k"]="$v"; done < <(read_manifest model_paths)

has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { echo ">>> $*"; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

for m in "${models[@]}"; do
  mp="${MPATH[$m]:?no model_paths entry for $m in $MANIFEST}"
  for s in "${subsets[@]}"; do
    echo "=== $DATASET / $m / $s ==="
    has activations && run python -m src.extract.activations \
        --model "$m" --model-path "$mp" \
        --input "data/$DATASET" --subset "$s" \
        --output "outputs/$DATASET/activations/$m/$s" \
        --layers all --pool all --max_tokens "$max_tokens" --device cuda
    has attention && run python -m src.extract.attention \
        --model "$m" --model-path "$mp" \
        --input "data/$DATASET" --subset "$s" \
        --output-root "outputs/$DATASET/attention" \
        --max_tokens "$max_tokens" --device cuda
  done
done
echo ">>> extraction done: $DATASET"
