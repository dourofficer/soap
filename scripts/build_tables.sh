#!/usr/bin/env bash
# Build the final per-dataset main results tables — SVD + CRR + the three prompting
# baselines (All-at-once / Step-by-step / Binary search) — in one command.
#
# Wraps `experiments.reports.build_main_tables`, which fills the baseline rows from
# the VLLM prompting predictions on the SAME chosen seeds and test splits the
# SVD/CRR rows use (fair comparison). Cells with missing/incomplete data are left
# blank (SVD/CRR for ww until its reduced tables are added; any in-progress
# prompting run).
#
# Assumes the CRR reduced tables exist where available (scripts/run_analysis.sh)
# and the prompting predictions have been generated (run_qwen.sh / run_deepseek.sh).
#
# Usage:
#   bash scripts/build_tables.sh                      # all datasets, top-3 seeds
#   bash scripts/build_tables.sh --select-seeds 5     # override CRR-ranked seed count
#   bash scripts/build_tables.sh --dataset ww
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS=(correct-error traceelephant ww)

# If the caller narrows to one --dataset, honor it; otherwise do each in turn.
if [[ " $* " == *" --dataset "* ]]; then
  uv run python -m experiments.reports.build_main_tables "$@"
else
  for ds in "${DATASETS[@]}"; do
    uv run python -m experiments.reports.build_main_tables --dataset "$ds" "$@"
  done
fi

echo "=== results tables written ==="
for ds in "${DATASETS[@]}"; do
  t="outputs-$ds/results_table.tsv"
  [ -f "$t" ] && echo "  $t"
done
