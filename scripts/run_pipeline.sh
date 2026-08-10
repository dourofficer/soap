#!/usr/bin/env bash
# End-to-end pipeline for one dataset (after extraction):
#   score -> triples select (pass 1: SVD + baselines) -> rescore sweep
#         -> triples select (pass 2: + rescoring rows)
#
#   DS=correct-full bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DS="${DS:?set DS=<dataset> (ww | traceelephant | correct-error | correct-full)}"

python -m src.score.run       --config "configs/score/${DS}.yaml"
python -m src.reports.triples --config "configs/protocol/${DS}.yaml"
python -m src.rescore.run     --config "configs/protocol/${DS}.yaml"
python -m src.reports.triples --config "configs/protocol/${DS}.yaml"
