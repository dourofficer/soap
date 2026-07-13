"""Generic sweep engine shared by every stage driver.

Consolidates the ``format_command`` / ``run`` helpers that were duplicated across
the activations, attention and svd drivers, plus the grid expansion. A stage
driver now only declares its axes and an ``argv`` builder; this module handles
pretty-printing, dry-run, and continue-on-error.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Callable, Iterable, Sequence

from rich.console import Console

CONSOLE = Console()


def resolve_model(cfg: dict, model: str) -> str:
    """Map a model shorthand to its HF path/name via the manifest ``model_paths``."""
    return cfg.get("model_paths", {}).get(model, model)


def format_command(module: str, argv: Sequence[str]) -> str:
    head = f"{sys.executable} -m {module}"
    if not argv:
        return head
    groups: list[list[str]] = []
    current: list[str] = []
    for token in argv:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    groups.append(current)
    args = " \\\n    ".join(" ".join(shlex.quote(t) for t in g) for g in groups)
    return f"{head} \\\n    {args}"


def run(module: str, argv: Sequence[str], dry_run: bool,
        continue_on_error: bool = False) -> int:
    """Shell out to ``python -m <module> <argv>``; return the exit code.

    dry_run → print only (returns 0). continue_on_error → log a failure and keep
    going instead of raising (mirrors the old attention driver's ``|| continue``).
    """
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if dry_run:
        return 0
    result = subprocess.run([sys.executable, "-m", module, *list(argv)])
    if result.returncode != 0:
        if continue_on_error:
            CONSOLE.print(
                f"[bold red]FAILED[/] (exit {result.returncode}) — continuing...")
        else:
            raise subprocess.CalledProcessError(result.returncode,
                                                [module, *argv])
    return result.returncode


def run_grid(module: str,
             combos: Iterable[tuple],
             argv_fn: Callable[[tuple], Sequence[str]],
             dry_run: bool,
             continue_on_error: bool = False) -> None:
    """Run ``module`` once per combo, building argv via ``argv_fn(combo)``."""
    combos = list(combos)
    for i, combo in enumerate(combos, 1):
        CONSOLE.print(f"[{i}/{len(combos)}] {combo}", style="bold cyan")
        run(module, argv_fn(combo), dry_run, continue_on_error)
