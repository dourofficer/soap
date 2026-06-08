#!/bin/bash
 
models=(qwen3-14b qwen3-8b deepseek-8b llama-3.1-8b)
subsets=(hand-crafted algorithm-generated)
 
for model in "${models[@]}"; do
    for subset in "${subsets[@]}"; do
        echo "=== Running model=$model subset=$subset ==="
        python -m src.attention.streaming \
            --model        "$model" \
            --subset       "$subset" \
            --input        data/ww \
            --output-root  outputs/weighting_attn \
            --max_tokens   8192 \
            --context      all \
            --query-pool   mean \
            --device       auto \
            --dtype        bfloat16 \
        || echo "FAILED: model=$model subset=$subset — continuing..."
    done
done
 
echo "=== All runs complete ==="