"""
Sequential sweep runner over MODELS x SUBSETS x POOLINGS x SEEDS.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.classifier.run \
        --config experiments/classifier/configs/default.yaml \
        --dry-run

    CUDA_VISIBLE_DEVICES=0 python -m experiments.classifier.run \
        --config experiments/classifier/configs/hidden_full.yaml \
        --set models=[qwen3-8b] --dry-run

    CUDA_VISIBLE_DEVICES=0 python -m experiments.classifier.run \
        --config experiments/classifier/configs/hidden_full.yaml \
        --set models=[llama-3.1-8b] --dry-run

    CUDA_VISIBLE_DEVICES=0 python -m experiments.classifier.run \
        --config experiments/classifier/configs/hidden_full.yaml \
        --set models=[llama-3.1-8b] \
              subsets=[hand-crafted] \
              outputs_root="outputs/classifier-full-noval-trunc" \
              seeds=[1] \
         --dry-run

    CUDA_VISIBLE_DEVICES=0 python -m experiments.classifier.run \
        --config experiments/classifier/configs/hidden_full.yaml \
        --set models=[qwen3-8b] \
              outputs_root="outputs/classifier" \
              seeds=[1] \
         --dry-run
"""

import argparse
import itertools
import shlex
import subprocess
import sys
import yaml
from pathlib import Path

from rich.console import Console

CONSOLE = Console()
MODULE  = "attribscope.classifier.run_all_positions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    formatted_groups = [" ".join(shlex.quote(t) for t in g)
                        for g in groups]
    args = " \\\n    ".join(formatted_groups)
    return f"{head} \\\n    {args}"


def run(module: str, argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Arg building
# ---------------------------------------------------------------------------

def build_argv(model: str, subset: str, pooling: str, seed: int, cfg: dict) -> list[str]:
    return [
        "--model",        model,
        "--subset",       subset,
        "--pooling",      pooling,
        "--seed",         str(seed),
        "--reps-root",    str(cfg["reps_root"]),
        "--data-root",    str(cfg["data_root"]),
        "--outputs-root", str(cfg["outputs_root"]),
        "--device",       str(cfg.get("device", "cuda")),
        "--thresholds",   *[str(t) for t in cfg["thresholds"]],
        "--positions",    *cfg.get("positions", ["all"]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        key, _, value = item.partition("=")
        # try to parse as YAML so lists/ints work: models=qwen3,gpt4 or seed=42
        cfg[key] = yaml.safe_load(value)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.set)

    combos = list(itertools.product(
        cfg["models"],
        cfg["subsets"],
        cfg["poolings"],
        cfg["seeds"],
    ))
    CONSOLE.print(f"Total combos: {len(combos)}\n")

    failed = []
    for i, (model, subset, pooling, seed) in enumerate(combos):
        label = f"[{i+1}/{len(combos)}] {model} | {subset} | pooling={pooling} | seed={seed}"
        CONSOLE.rule(label)

        argv = build_argv(model, subset, pooling, seed, cfg)
        try:
            run(MODULE, argv, dry_run=args.dry_run)
        except subprocess.CalledProcessError as e:
            CONSOLE.print(f"[FAILED rc={e.returncode}] {label}", style="bold red")
            failed.append(label)

    CONSOLE.rule()
    if failed:
        CONSOLE.print(f"{len(failed)} job(s) failed:", style="bold red")
        for lbl in failed:
            CONSOLE.print(f"  {lbl}", style="red")
        sys.exit(1)
    else:
        CONSOLE.print("All jobs completed successfully.", style="bold green")


if __name__ == "__main__":
    main()