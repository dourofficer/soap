"""Shared trajectory-metadata header written into every .safetensors payload.

Previously copy-pasted identically in activations/extract.py, attention/streaming.py
and attention/eager.py. Single definition here keeps the on-disk header stable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # [data_v2 swap] rollback: uncomment the src.data line, delete the src.data_v2 one
    # from src.data.trajectory import Trajectory
    from src.data_v2.trajectory import Trajectory


def extract_metadata(traj: "Trajectory") -> dict:
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
    }
