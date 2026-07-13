"""Rescore (CRR) sweep — discount the base scores across (layer_range, gamma, w, orient).

Thin driver on experiments/_common: shells out to ``src.rescore.run`` once per
(model, subset), deriving the undisc/attn/reps/output roots and split ratios from
the dataset manifest.

    CUDA_VISIBLE_DEVICES=0 python -m experiments.rescore.sweep \
        --config experiments/rescore/configs/correct-error.yaml --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common.config import load_stage_config
from experiments._common import paths
from experiments._common.sweep import run

MODULE = "src.rescore.run"


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.rescore.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_stage_config(args.config, args.overrides)

    splits = cfg["splits"]
    common = [
        "--undisc-root", str(paths.undisc_root(cfg)),
        "--attn-root",   str(paths.attn_root(cfg)),
        "--reps-root",   str(paths.reps_root(cfg)),
        "--data-root",   str(cfg["data_root"]),
        "--out-root",    str(paths.rescore_sweep_root(cfg)),
        "--undisc-file", cfg.get("undisc_file", "weighted_false.tsv"),
        "--device",      cfg["device"],
        "--n-ranges",    str(cfg["n_ranges"]),
        "--gammas",      *[str(g) for g in cfg["gammas"]],
        "--ws",          *[str(w) for w in cfg["ws"]],
        "--orients",     *cfg["orients"],
        "--train-split", str(splits["train"]),
        "--val-split",   str(splits["val"]),
        "--test-split",  str(splits["test"]),
    ]

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            run(MODULE, ["--model", model, "--subset", subset, *common],
                args.dry_run)


if __name__ == "__main__":
    main()
