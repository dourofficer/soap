"""Attention sweep — extract per-step attention mass into predecessors.

Thin driver on experiments/_common: shells out to ``src.attention.streaming``
once per (model, subset), resolving the model path from the dataset manifest.

    CUDA_VISIBLE_DEVICES=0 python -m experiments.attention.sweep \
        --config experiments/attention/configs/correct-error.yaml --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common.config import load_stage_config
from experiments._common import paths
from experiments._common.sweep import CONSOLE, resolve_model, run

MODULE = "src.attention.streaming"


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.attention.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_stage_config(args.config, args.overrides)

    data_root = str(cfg["data_root"])
    attn_root = str(paths.attn_root(cfg))

    for model in cfg["models"]:
        model_path = resolve_model(cfg, model)
        for subset in cfg["subsets"]:
            CONSOLE.rule(f"[bold]model={model} | subset={subset}[/]")
            run(MODULE, [
                "--model",       model,
                "--model-path",  model_path,
                "--subset",      subset,
                "--input",       data_root,
                "--output-root", attn_root,
                "--max_tokens",  str(cfg.get("max_tokens", 8192)),
                "--query-pool",  cfg.get("query_pool", "mean"),
                "--device",      cfg.get("device", "auto"),
                "--dtype",       cfg.get("dtype", "bfloat16"),
            ], args.dry_run, continue_on_error=True)

    CONSOLE.rule("[bold green]All runs complete[/]")


if __name__ == "__main__":
    main()
