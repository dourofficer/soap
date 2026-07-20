"""Per-invocation run record: what ran, with what config, producing what.

Every runner calls :func:`write_run_record` on completion, dropping one JSON under
``outputs/<ds>/<stage>/runs/<stage>-<utcstamp>.json`` with argv, resolved config,
git sha of the v2 tree, torch/cuda/device info, input roots + counts, wall time,
and the list of outputs written. This is the "comprehensive intermediate logging"
the pipeline relies on for reproducibility.

Standalone (rare): usually called from within a runner, not the CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import paths


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _torch_info() -> dict:
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
        }
    except Exception:
        return {}


class RunTimer:
    """Context manager collecting wall time + written outputs for a run record.

        with RunTimer(cfg, "score") as run:
            ...
            run.add_output(path)
            run.note(extra_key=value)
    """

    def __init__(self, cfg: dict, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.outputs: list[str] = []
        self.extras: dict = {}
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def add_output(self, path) -> None:
        self.outputs.append(str(path))

    def note(self, **kw) -> None:
        self.extras.update(kw)

    def __exit__(self, exc_type, exc, tb):
        if self.cfg.get("dry_run"):
            return False
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        write_run_record(self.cfg, self.stage, self.outputs,
                         wall_s=round(elapsed, 3),
                         failed=exc_type is not None, **self.extras)
        return False


def write_run_record(cfg: dict, stage: str, outputs: list[str],
                     wall_s: float | None = None, **extras) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = paths.runs_dir(cfg, stage)
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "stage": stage,
        "utc": stamp,
        "argv": sys.argv,
        "config": cfg,
        "git_sha": _git_sha(),
        "wall_s": wall_s,
        "outputs": outputs,
        **_torch_info(),
        **extras,
    }
    path = d / f"{stage}-{stamp}.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path
