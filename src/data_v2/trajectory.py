"""src/data_v2/trajectory.py — re-export the unchanged trajectory layer.

Only the context builders differ in data_v2; the Trajectory dataclass and dataset
loading are identical, so we re-export them from :mod:`src.data.trajectory`. Fork
this module if you want to hack the trajectory representation independently.
"""
from src.data.trajectory import Trajectory, load_dataset

__all__ = ["Trajectory", "load_dataset"]
