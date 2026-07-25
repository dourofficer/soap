"""Trajectory dataclass + dataset loading.

    from src.data import load_dataset
    trajs = load_dataset("data/correct-full", subset="magentic")

Each JSON has: history (ordered turns; step t == history[t], 0-indexed), question_ID,
mistake_agent, mistake_step (string index into history), level, subset. Turn roles are
agent/system names; mistake_agent matches a turn's role.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


# ── directory helpers (ported from src/utils/common.py) ─────────────────────
def _sorted_json_files(directory: Path) -> list[str]:
    """JSON filenames sorted numerically by their digits (1.json, 2.json, ...)."""
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    return sorted(files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── data structure ──────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    """One failure instance (a Who&When / CORRECT / TraceElephant-style trace)."""
    filename:      str
    question_id:   str
    history:       list[dict]
    mistake_agent: str
    mistake_step:  int           # 0-indexed
    level:         int
    subset:        str
    question:      str
    system:        str | None


def load_dataset(path: str | Path, subset: str | None = None) -> list[Trajectory]:
    """Load ``<path>/<subset>/*.json`` into a list of Trajectory (numerically sorted)."""
    root = Path(path) / subset
    trajectories: list[Trajectory] = []
    for filename in _sorted_json_files(root):
        item = _load_json(root / filename)
        system_description = None
        if subset == "algorithm-generated":
            prefix = "## Your role\n"
            system_description = "\n\n".join(
                f"{name}: {desc[len(prefix):].strip()}"
                for name, desc in item.get("system_prompt", {}).items()
            )
        trajectories.append(Trajectory(
            filename      = filename,
            question_id   = item["question_ID"],
            history       = item["history"],
            mistake_agent = item["mistake_agent"],
            mistake_step  = int(item["mistake_step"]),
            level         = item.get("level", -1),
            subset        = subset,
            question      = item.get("question", ""),
            system        = system_description,
        ))
    return trajectories


def extract_metadata(traj: Trajectory) -> dict:
    """Trajectory-metadata header written into every .safetensors payload."""
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
    }
