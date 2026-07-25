#!/usr/bin/env bash
# Attention-mass extraction (streaming / low-memory sdpa override).
#   models : qwen3.5-9b, qwen3.5-27b   (must exist as keys in streaming.py MODELS)
#   subsets: hand-crafted, algorithm-generated
# Uses streaming.py because eager OOMs at these sizes / lengths.
# Run from the repo root. Override defaults via env vars.
set -euo pipefail

INPUT="${INPUT:-data/ww}"
OUT_ROOT="${OUT_ROOT:-outputs-1806/attention}"   # streaming appends /<model>/<subset>
MAX_TOKENS="${MAX_TOKENS:-8192}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"                     # 'auto' shards the 27B across GPUs
QUERY_POOL="${QUERY_POOL:-mean}"

# --model is a MODELS-dict key here (not a path), unlike the representation CLI.
MODELS=(qwen3.5-9b)
SUBSETS=(hand-crafted algorithm-generated)

for name in "${MODELS[@]}"; do
  for subset in "${SUBSETS[@]}"; do
    echo "=== attention | $name | $subset -> $OUT_ROOT/$name/$subset ==="
    CUDA_VISIBLE_DEVICES=3 python -m src.attention.streaming \
      --model       "$name" \
      --input       "$INPUT" \
      --subset      "$subset" \
      --output-root "$OUT_ROOT" \
      --max_tokens  "$MAX_TOKENS" \
      --query-pool  "$QUERY_POOL" \
      --device      "$DEVICE" \
      --dtype       "$DTYPE"
  done
done