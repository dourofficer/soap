#!/usr/bin/env bash
# Seed main/'s results trees from the frozen reference extractions.
#
#   scripts/seed_results.sh                    # hardlink copy (default, free)
#   scripts/seed_results.sh --real-copy        # byte copy (~65G across both trees)
#   DATASETS="ww" scripts/seed_results.sh      # one dataset
#
#   outputs/<ds>/{activations,attention}     -> results-nogt/<ds>/{activations,attention}
#   outputs-gt/<ds>/{activations,attention}  -> results-gt/<ds>/{activations,attention}
#
# main/ writes byte-compatible artifacts (same keys, both poolings, same config.json), so
# these are the same files, not a conversion. The extractors never mutate an existing
# .safetensors — they skip it — so HARDLINKS are safe and give a self-contained tree at
# zero extra disk. --real-copy is there for moving a tree to another filesystem.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="link"
[[ "${1:-}" == "--real-copy" ]] && MODE="copy"
DATASETS="${DATASETS:-ww traceelephant correct-error}"

seed_one() {          # $1 src root, $2 dst root, $3 dataset
    local src="$1/$3" dst="$2/$3"
    for stage in activations attention; do
        [[ -d "$src/$stage" ]] || { echo "  [--] no $src/$stage"; continue; }
        mkdir -p "$dst"
        if [[ -d "$dst/$stage" ]]; then
            echo "  [skip] $dst/$stage exists"
            continue
        fi
        if [[ "$MODE" == "link" ]]; then
            cp -al "$src/$stage" "$dst/$stage"
        else
            cp -a "$src/$stage" "$dst/$stage"
        fi
        echo "  [ok] $src/$stage -> $dst/$stage  ($(find "$dst/$stage" -name '*.safetensors' | wc -l) files)"
    done
}

for ds in $DATASETS; do
    echo "=== $ds ==="
    seed_one outputs    results-nogt "$ds"
    seed_one outputs-gt results-gt   "$ds"
done

echo
echo "verify a sample checksum matches (should print nothing):"
for ds in $DATASETS; do
    for tree in "outputs results-nogt" "outputs-gt results-gt"; do
        set -- $tree
        f=$(find "$2/$ds" -name '*.safetensors' 2>/dev/null | head -1) || true
        [[ -n "${f:-}" ]] || continue
        orig="$1/${f#*/}"
        [[ -f "$orig" ]] || continue
        cmp -s "$f" "$orig" || echo "  MISMATCH $f vs $orig"
    done
done
echo "done"
