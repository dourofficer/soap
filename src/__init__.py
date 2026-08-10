"""attribscope — failure attribution from proxy-model internals.

Pipeline (run every stage from the repo root; see scripts/run_pipeline.sh):

    python -m src.score.run       --config configs/score/<ds>.yaml
    python -m src.reports.triples --config configs/protocol/<ds>.yaml   # pass 1
    python -m src.rescore.run     --config configs/protocol/<ds>.yaml
    python -m src.reports.triples --config configs/protocol/<ds>.yaml   # pass 2

Selection follows the seed-window ("triples") protocol; rescoring strategies are
backprop (SOAP) and the successor-side succ-strong / succ-near. See README.md for
the stage map and CLAUDE.md for conventions. The pre-protocol pipeline is archived
verbatim at src_v2/.
"""
__version__ = "0.3.0"
