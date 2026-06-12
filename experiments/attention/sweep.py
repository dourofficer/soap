"""
CUDA_VISIBLE_DEVICES=0 python -m experiments.attention.sweep \
    --config experiments/attention/configs/default.yaml
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
MODULE  = "src.attention.streaming"


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


def run(argv: list[str]) -> bool:
    """Returns True on success, False on failure (mirrors the bash || continue pattern)."""
    cmd = [sys.executable, "-m", MODULE, *argv]
    CONSOLE.print(format_command(MODULE, argv), style="green")
    CONSOLE.rule()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        CONSOLE.print(f"[bold red]FAILED[/] (exit {result.returncode}) — continuing...")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.attention.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            CONSOLE.rule(f"[bold]model={model} | subset={subset}[/]")
            argv = [
                "--model",       model,
                "--subset",      subset,
                "--input",       cfg["data_dir"],
                "--output-root", cfg["output_root"],
                "--max_tokens",  str(cfg.get("max_tokens", 8192)),
                "--query-pool",  cfg.get("query_pool", "mean"),
                "--device",      cfg.get("device", "auto"),
                "--dtype",       cfg.get("dtype", "bfloat16"),
            ]
            CONSOLE.rule()
            if not args.dry_run:
                run(argv)

    CONSOLE.rule("[bold green]All runs complete[/]")


if __name__ == "__main__":
    main()