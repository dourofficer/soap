"""
Usage:
CUDA_VISIBLE_DEVICES=0 python -m experiments.activations.sweep \
    --config experiments/activations/configs/default.yaml \
    --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
import yaml

CONSOLE = Console()

def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_model(cfg: dict, model: str) -> str:
    return cfg.get("model_paths", {}).get(model, model)


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


def run(module: str, argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run(cmd, check=True)


def index_args(cfg: dict) -> list[str]:
    argv: list[str] = []
    if cfg.get("start_idx") is not None:
        argv += ["--start_idx", str(cfg["start_idx"])]
    if cfg.get("end_idx") is not None:
        argv += ["--end_idx", str(cfg["end_idx"])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.activations.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    layers = cfg["layers"]
    if isinstance(layers, str):
        layers = [layers]

    for model in cfg["models"]:
        model_path = resolve_model(cfg, model)
        for subset in cfg["subsets"]:
            run("src.activations.extract", [
                "--model",      model_path,
                "--input",      f"{cfg['data_dir']}/{subset}",
                "--output",     f"{cfg['outputs_root']}/{model}/{subset}",
                "--layers",     *layers,
                "--pool",       cfg["pool"],
                "--max_tokens", str(cfg.get("max_tokens", 8192)),
                "--device",     cfg["device"],
                "--dtype",      cfg.get("dtype", "bfloat16"),
                *index_args(cfg),
            ], args.dry_run)


if __name__ == "__main__":
    main()