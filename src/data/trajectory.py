"""
data.py ― Dataset loading, step parsing, and context construction
           for Who&When GradNorm evaluation.

Public API
----------
Trajectory          dataclass holding one failure instance
load_dataset()      load a JSON file → list[Trajectory]
select_context()    ← PLACEHOLDER: return which history indices serve as context
build_context()     tokenise one (context, step) pair via apply_chat_template
custom_build_context()  ← PLACEHOLDER: drop-in replacement for build_context
iter_scoreable_steps()  steps that should receive a GradNorm score
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from ..utils.common import _get_sorted_json_files, _load_json_data

from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """One Who&When failure instance.

    Attributes
    ----------
    question_id   : unique ID string from the dataset.
    history       : raw history list; step t == history[t] (0-indexed).
    mistake_agent : ground-truth agent name (matches a history[t]["role"]).
    mistake_step  : ground-truth step index (0-indexed into history).
    level         : difficulty level.
    subset        : "algo" | "handcrafted" (or "unknown" if absent in JSON).
    question      : original user question string.
    """
    filename:      str
    question_id:   str
    history:       list[dict]
    mistake_agent: str
    mistake_step:  int           # 0-indexed
    level:         int
    subset:        str
    question:      str
    system:        str


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    path: str | Path = "ww",
    subset: str | None = None,
) -> list[Trajectory]:
    """Load the Who&When dataset from a JSON file.

    Parameters
    ----------
    path   : path to the data directory
    subset : optional filter — "algo" | "handcrafted".
             Pass None to return everything.

    Returns
    -------
    list[Trajectory]

    Expected JSON schema per item
    ------------------------------
    {
        "history":       [{"role": str, "content": str}, ...],
        "mistake_agent": str,
        "mistake_step":  str | int,   # parsed to int; 0-indexed
        "question_ID":   str,
        "level":         int,          # optional
        "subset":        str,          # optional; "algorithm-generated" | "hand-crafted"
        "question":      str           # optional
    }

    Notes
    -----
    If the JSON file does not contain a "subset" key (e.g., separate files per
    subset), supply the subset label via the `subset` argument *as a filter*
    only, or pre-tag the items before calling this function.
    """

    path = Path(path) / subset
    # import pdb; pdb.set_trace()
    filenames = _get_sorted_json_files(path)
    raw = [(filename, _load_json_data(path / filename)) for filename in filenames]

    trajectories: list[Trajectory] = []
    for filename, item in raw:
        system_description = None
        if subset == "algorithm-generated":
            system = item.get("system_prompt")
            prefix = "## Your role\n"
            agents = [
                f"{name}: {description[len(prefix):].strip()}"
                for name, description
                in system.items()
            ]
            system_description = "\n\n".join(agents)
        elif subset == "hand-crafted":
            pass
        traj = Trajectory(
            filename      = filename,
            question_id   = item["question_ID"],
            history       = item["history"],
            mistake_agent = item["mistake_agent"],
            mistake_step  = int(item["mistake_step"]),
            level         = item.get("level", -1),
            subset        = subset,
            question      = item.get("question", ""),
            system        = system_description,
        )
        trajectories.append(traj)

    return trajectories
