#!/usr/bin/env bash
# Re-parse the stored all_at_once prompting predictions (recovers markdown-bolded
# labels, e.g. DeepSeek-R1's `**Agent Name:**`) from their saved raw text — no GPU.
# Rewrites predicted_agent/predicted_step in place; raw is preserved (idempotent).
#
# Usage:
#   bash scripts/reparse.sh --dry-run     # preview recovery counts
#   bash scripts/reparse.sh               # apply, then rebuild tables
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

uv run python -m baselines.prompting.reparse "$@"

echo
echo "Next: rebuild the results tables →  bash scripts/build_tables.sh"


cd /home/hoangpham/attribscope
mkdir -p logs