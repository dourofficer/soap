"""Trajectory representation + step-context construction."""
from .trajectory import Trajectory, load_dataset
from .context import (
    build_context,
    separate_steps,
    iter_scoreable_steps,
    select_context,
)

__all__ = [
    "Trajectory", "load_dataset",
    "build_context", "separate_steps", "iter_scoreable_steps", "select_context",
]
