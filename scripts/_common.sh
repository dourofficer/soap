# scripts/_common.sh — shared knobs + helpers for the pipeline wrappers.
# Sourced by gen_embeddings.sh / extract_attention.sh / run_analysis.sh.
# Not meant to be run directly.
#
# Onboarding a new dataset <ds> (any data in the current trajectory format):
#   1. Put trajectories at  data/<ds>/<subset>/*.json
#   2. Write ONE manifest — experiments/datasets/<ds>.yaml — with models,
#      model_paths, subsets, data_root, max_tokens, and split ratios (see
#      experiments/datasets/correct-error.yaml). Output roots are DERIVED from it
#      by experiments/_common/paths.py; no need to hand-thread them.
#   3. Copy the six thin stage configs (each just `dataset: <ds>` + its sweep axes):
#        experiments/{activations,attention,svd,rescore}/configs/<ds>.yaml
#        experiments/reports/configs/{undiscounted_v2,discounted_v2}_<ds>.yaml
#   4. Run these scripts with DATASET=<ds>.

set -euo pipefail

# Run everything from the repo root (code lives in src/, invoked via `python -m`).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Shared knobs (override via environment) ──────────────────────────────────
DATASET="${DATASET:-correct-error}"   # selects the config family
MODELS="${MODELS:-}"                  # space-separated; empty → use config's list
GPU="${GPU:-0}"                       # CUDA_VISIBLE_DEVICES for GPU stages
DRY_RUN="${DRY_RUN:-0}"               # 1 → echo commands only, don't execute
EXTRA_SET="${EXTRA_SET:-}"            # passthrough extra `--set k=v` overrides

# `--set models=[a,b]` from MODELS="a b"; prints nothing when MODELS is empty.
models_flag() {
  [ -z "$MODELS" ] && return 0
  printf -- '--set models=[%s]' "${MODELS// /,}"
}

# Echo a banner + the command; execute unless DRY_RUN=1. Uniform across every
# stage, including those (rescore, report builders) that lack their own --dry-run.
run() {
  echo "=== [$DATASET${MODELS:+ | $MODELS}${GPU:+ | gpu $GPU}] $* ==="
  [ "$DRY_RUN" = 1 ] || "$@"
}
