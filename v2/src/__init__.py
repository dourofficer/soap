"""attribscope v2 — SVD + CRR failure attribution from proxy-model internals.

Run every stage from the v2/ directory:

    python -m src.score.run    --config configs/score/correct-full.yaml
    python -m src.reports.reduce --config configs/reduce/correct-full.yaml --set stage=base
    python -m src.rescore.run  --config configs/rescore/correct-full.yaml
    python -m src.reports.reduce --config configs/reduce/correct-full.yaml --set stage=crr
    python -m src.reports.tables --config configs/tables/correct-full.yaml

See README.md for the stage map and CLAUDE.md for conventions.
"""
__version__ = "0.2.0"
