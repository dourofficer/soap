#!/usr/bin/env bash
# Representation (hidden-state) extraction.
#   models : qwen3.5-9b, qwen3.5-27b
#   subsets: hand-crafted, algorithm-generated
# Run from the repo root. Override defaults via env vars, e.g.:
#   HUB=/data/models DEVICE=cuda ./scripts/extract_representations.sh
set -euo pipefail

HUB="${HUB:-/data/hoang/resources/models}"     # where the model weights live
INPUT="${INPUT:-data/ww}"           # dataset root (subset is a subdir)
OUT_ROOT="${OUT_ROOT:-outputs-1806/activations}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"            # 'auto' shards the 27B across GPUs
POOL="${POOL:-last}"
LAYERS="${LAYERS:-all}"            # 'all' -> embed + gated-attention blocks (Qwen3.5)

# name -> weights path. --model here takes a path or HF id.
declare -A MODELS=(
  [qwen3.5-9b]="$HUB/Qwen/Qwen3.5-9B"
  [qwen3.5-27b]="$HUB/Qwen/Qwen3.5-27B"
)
SUBSETS=(hand-crafted algorithm-generated)

for name in "${!MODELS[@]}"; do
  path="${MODELS[$name]}"
  for subset in "${SUBSETS[@]}"; do
    out="$OUT_ROOT/$name/$subset"
    mkdir -p "$out"
    echo "=== representations | $name | $subset -> $out ==="
    CUDA_VISIBLE_DEVICES=1,2 python -m src.activations.extract \
      --model      "$path" \
      --input      "$INPUT" \
      --subset     "$subset" \
      --output     "$out" \
      --layers     "$LAYERS" \
      --pool       "$POOL" \
      --max_tokens "$MAX_TOKENS" \
      --device     "$DEVICE" \
      --dtype      "$DTYPE"
  done
done