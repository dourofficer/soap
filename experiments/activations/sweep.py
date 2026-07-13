"""Activations sweep — extract per-step hidden states for a dataset.

Thin driver on experiments/_common: loads the stage config (which merges the
dataset manifest), derives the reps output root, and shells out to
``src.activations.extract`` once per (model, subset).

    CUDA_VISIBLE_DEVICES=0 python -m experiments.activations.sweep \
        --config experiments/activations/configs/correct-error.yaml --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common.config import load_stage_config
from experiments._common import paths
from experiments._common.sweep import resolve_model, run

MODULE = "src.activations.extract"


def _index_args(cfg: dict) -> list[str]:
    argv: list[str] = []
    if cfg.get("start_idx") is not None:
        argv += ["--start_idx", str(cfg["start_idx"])]
    if cfg.get("end_idx") is not None:
        argv += ["--end_idx", str(cfg["end_idx"])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.activations.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_stage_config(args.config, args.overrides)

    layers = cfg["layers"]
    if isinstance(layers, str):
        layers = [layers]

    data_root = Path(cfg["data_root"])
    reps_root = paths.reps_root(cfg)

    for model in cfg["models"]:
        model_path = resolve_model(cfg, model)
        for subset in cfg["subsets"]:
            run(MODULE, [
                "--model",      model_path,
                "--input",      str(data_root / subset),
                "--output",     str(reps_root / model / subset),
                "--layers",     *layers,
                "--pool",       cfg["pool"],
                "--max_tokens", str(cfg.get("max_tokens", 8192)),
                "--device",     cfg["device"],
                "--dtype",      cfg.get("dtype", "bfloat16"),
                *_index_args(cfg),
            ], args.dry_run)


if __name__ == "__main__":
    main()
