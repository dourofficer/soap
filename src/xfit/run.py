"""Orchestrator for the synthetic cross-fit: extract -> score -> rescore -> table.

Thin driver over the four stage modules; each stage is also runnable on its own
(``python -m src.xfit.<stage>``). Everything is confined to ``src/xfit/`` and writes only
under ``outputs/`` (synthetic reps + ``xfit-<source>`` split-tagged roots per target).

    # from repo root
    python -m src.xfit.run --stage all                          # everything
    python -m src.xfit.run --stage extract --gpu 0
    python -m src.xfit.run --stage score   --source magentic-qwen9b --proxy qwen3.5-9b
    python -m src.xfit.run --stage rescore --dataset correct-full
    python -m src.xfit.run --stage table   --dataset ww
"""
from __future__ import annotations

import argparse

import torch

from .common import load_config, setting
from . import extract, score, rescore, table, table_paper, verify

STAGES = ["extract", "score", "rescore", "verify", "table"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument("--proxy", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--gpu", default="0", help="CUDA index (or 'auto'/'cpu'); score/rescore device.")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()

    cfg = load_config(args.overrides)
    paper = setting(cfg) == "paper"
    want = STAGES if args.stage == "all" else [args.stage]
    if args.stage == "all":
        # legacy has no verify stage; paper needs no extraction (fit-pool reps shared).
        want = [s for s in want if s != ("verify" if not paper else "extract")]
    dev = args.gpu if args.gpu in ("auto", "cpu") else f"cuda:{args.gpu}"

    if "extract" in want:
        ext_dev = args.gpu if args.gpu in ("auto", "cpu") else f"cuda:{args.gpu}"
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]
        extract.run(cfg, only_proxy=args.proxy, only_source=args.source,
                    device=ext_dev, dtype=dtype, force=args.force, dry=args.dry_run)
    if "score" in want:
        score.run(cfg, only_proxy=args.proxy, only_source=args.source,
                  only_dataset=args.dataset, device=dev, force=args.force)
    if "rescore" in want:
        rescore.run(cfg, only_proxy=args.proxy, only_source=args.source,
                    only_dataset=args.dataset, device=dev, force=args.force)
    if "verify" in want and paper:
        verify.run(cfg, only_dataset=args.dataset)
    if "table" in want:
        (table_paper if paper else table).run(cfg, only_dataset=args.dataset)


if __name__ == "__main__":
    main()
