"""
CUDA_VISIBLE_DEVICES=0 python -m experiments.svd.sweep \
    --config experiments/svd/configs/default.yaml

CUDA_VISIBLE_DEVICES=1 python -m experiments.svd.sweep \
    --config experiments/svd/configs/default.yaml \
    --set models=[qwen3-8b] --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console

CONSOLE = Console()
MODULE  = "experiments.svd.run_all_positions"


def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def format_command(module: str, argv: list[str]) -> str:
    head = f"{sys.executable} -m {module}"
    if not argv:
        return head
    groups, current = [], []
    for token in argv:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    groups.append(current)
    args = " \\\n    ".join(" ".join(shlex.quote(t) for t in g) for g in groups)
    return f"{head} \\\n    {args}"


def run(argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", MODULE, *argv]
    CONSOLE.print(format_command(MODULE, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.svd.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            for pooling in cfg["poolings"]:
                for seed in cfg["seeds"]:
                    run([
                        "--reps-root",    cfg["reps_root"],
                        "--data-root",    cfg["data_root"],
                        "--outputs-root", cfg["outputs_root"],
                        "--model",        model,
                        "--subset",       subset,
                        "--pooling",      pooling,
                        "--positions",    *cfg.get("positions", ["all"]),
                        "--seed",         str(seed),
                        "--device",       cfg["device"],
                    ], args.dry_run)


if __name__ == "__main__":
    main()