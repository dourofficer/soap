"""SVD sweep — fit + score base per-step scores across (model, subset, pooling, seed).

Thin driver on experiments/_common: shells out to ``src.svd.score`` once per grid
cell, deriving reps/output roots and split ratios from the dataset manifest.

    CUDA_VISIBLE_DEVICES=0 python -m experiments.svd.sweep \
        --config experiments/svd/configs/correct-error.yaml --dry-run
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from experiments._common.config import load_stage_config
from experiments._common import paths
from experiments._common.sweep import run_grid

MODULE = "src.svd.score"


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.svd.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_stage_config(args.config, args.overrides)

    splits    = cfg["splits"]
    reps_root = str(paths.reps_root(cfg))
    data_root = str(cfg["data_root"])
    svd_root  = str(paths.svd_root(cfg))
    positions = cfg.get("positions", ["all"])

    combos = itertools.product(cfg["models"], cfg["subsets"],
                               cfg["poolings"], cfg["seeds"])

    def argv_fn(combo):
        model, subset, pooling, seed = combo
        return [
            "--reps-root",    reps_root,
            "--data-root",    data_root,
            "--outputs-root", svd_root,
            "--model",        model,
            "--subset",       subset,
            "--pooling",      pooling,
            "--positions",    *positions,
            "--seed",         str(seed),
            "--device",       cfg["device"],
            "--train-split",  str(splits["train"]),
            "--val-split",    str(splits["val"]),
            "--test-split",   str(splits["test"]),
        ]

    run_grid(MODULE, combos, argv_fn, args.dry_run)


if __name__ == "__main__":
    main()
