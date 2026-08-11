"""soap — failure attribution from proxy-model internals (sweep machinery).

Pipeline (run every stage from the repo root; see scripts/run_pipeline.sh):

    python -m src.score.run       --config configs/score/<ds>.yaml
    python -m src.reports.triples --config configs/protocol/<ds>.yaml   # pass 1
    python -m src.rescore.run     --config configs/protocol/<ds>.yaml
    python -m src.reports.triples --config configs/protocol/<ds>.yaml   # pass 2

Selection follows the seed-window ("triples") protocol; rescoring strategies are
backprop (SOAP) and the successor-side succ-strong / succ-near.

``src/`` is the full sweep: every scorer, orientation, normalization and centering arm
stays implemented here so any axis can be swept again by editing a config, even though
the production configs pin all of them. ``main/`` is the simplified, self-contained
runner with those axes frozen in code — it is the primary entry point for new runs.

See README.md for the stage map and CLAUDE.md for conventions.
"""
__version__ = "0.4.0"
