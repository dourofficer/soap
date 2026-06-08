"""
experiments/reps/run.py

Minimal YAML-driven runner. Builds argv from config and shells out to
attribscope.reps.extract_grads / extract_hidden unchanged.

    python -m experiments.reps.run grads  --config experiments/reps/configs/default.yaml
    python -m experiments.reps.run hidden --config experiments/reps/configs/default.yaml

    # override anything, one-shot:
    python -m experiments.reps.run grads --config ... --set shared.models=[qwen3-8b]
    python -m experiments.reps.run grads --config ... --set shared.start_idx=0 --set shared.end_idx=50

    # see the commands that would run, without running them:
    python -m experiments.reps.run hidden --config ... --dry-run

CUDA_VISIBLE_DEVICES=0 python -m experiments.reps.run \
    --config experiments/reps/configs/hidden.yaml \
    --set shared.context=dependency shared.outputs_root="outputs/2025-05-25"

CUDA_VISIBLE_DEVICES=1 python -m experiments.reps.run hidden \
    --config experiments/reps/configs/hidden_all.yaml \
    --set shared.models=[deepseek-8b]
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
    """Return the full path / HF name for a model shorthand."""
    paths = cfg["shared"].get("model_paths", {})
    return paths.get(model, model)  # fall back to the shorthand itself


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase runners
# ─────────────────────────────────────────────────────────────────────────────

def index_args(s: dict) -> list[str]:
    """Return --start_idx / --end_idx argv fragments from shared config."""
    argv: list[str] = []
    if s.get("start_idx") is not None:
        argv += ["--start_idx", str(s["start_idx"])]
    if s.get("end_idx") is not None:
        argv += ["--end_idx", str(s["end_idx"])]
    return argv


def run_grads(cfg: dict, dry_run: bool) -> None:
    """
    File structure, each of the leaf directories
    contains */reps/{subset}/*.safetensors files.
    .
    ├── grads
    │   ├── llama-3.1-8b
    │   ├── llama-3.1-8b-kl
    │   │   ├── temp_1.x
    │   │   └── uniform
    │   ├── qwen3-8b
    │   └── qwen3-8b-kl
    │       ├── temp_1.x
    │       └── uniform
    └── hidden
        ├── llama-3.1-8b
        └── qwen3-8b
    """
    s, g = cfg["shared"], cfg["grads"]

    target_params = g["target_params"]
    if isinstance(target_params, str):
        target_params = [target_params]

    loss = g.get("loss", "ntp")

    # For kl_temp, loop over every requested temperature and place each run
    # in its own sub-directory so outputs don't collide.
    if loss == "kl_temp":
        temperatures: list[float] = g.get("temperatures", [1.0])
        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]
    else:
        temperatures = [None]   # sentinel: no temperature axis

    for model in s["models"]:
        model_path = resolve_model(cfg, model)
        for subset in s["subsets"]:
            for temp in temperatures:
                if temp is not None:
                    temp_tag  = f"temp_{temp}"
                    out_dir   = f"{s['outputs_root']}/grads/{model}-kl/{temp_tag}/reps/{subset}/"
                    temp_argv = ["--temperature", str(temp)]
                else:
                    out_dir   = f"{s['outputs_root']}/grads/{model}/{subset}"
                    if loss == "kl_uniform": 
                        out_dir   = f"{s['outputs_root']}/grads/{model}-kl/uniform/reps/{subset}"
                    temp_argv = []

                run("attribscope.reps.extract_grads", [
                    "--model",         model_path,
                    "--input",         f"{s['data_dir']}/{subset}",
                    "--output",        out_dir,
                    "--target_params", *target_params,
                    "--loss",          loss,
                    "--max_tokens",    str(s.get("max_tokens", 8192)),
                    "--device",        s["device"],
                    "--dtype",         s.get("dtype", "bfloat16"),
                    "--context",       s.get("context", "dependency"),
                    *temp_argv,
                    *index_args(s),
                ], dry_run)


def run_hidden(cfg: dict, dry_run: bool) -> None:
    s, h = cfg["shared"], cfg["hidden"]

    layers = h["layers"]
    if isinstance(layers, str):
        layers = [layers]

    for model in s["models"]:
        model_path = resolve_model(cfg, model)
        for subset in s["subsets"]:
            run("attribscope.reps.extract_hidden", [
                "--model",      model_path,
                "--input",      f"{s['data_dir']}/{subset}",
                "--output",     f"{s['outputs_root']}/hidden/{model}/reps/{subset}",
                "--layers",     *layers,
                "--pool",       h["pool"],
                "--max_tokens", str(s.get("max_tokens", 8192)),
                "--device",     s["device"],
                "--dtype",      s.get("dtype", "bfloat16"),
                "--context",    s.get("context", "dependency"),
                *index_args(s),
            ], dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PHASES = {
    "grads":  run_grads,
    "hidden": run_hidden,
}

def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.reps.run")
    sub = p.add_subparsers(dest="phase", required=True)
    for name in PHASES:
        sp = sub.add_parser(name)
        sp.add_argument("--config", type=Path, required=True)
        sp.add_argument(
            "--set", dest="overrides", action="append", default=[],
            metavar="KEY=VALUE",
            help="e.g. --set shared.models=[qwen3-8b]",
        )
        sp.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    PHASES[args.phase](cfg, args.dry_run)


if __name__ == "__main__":
    main()