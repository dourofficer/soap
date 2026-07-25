"""Shared config, path and target-mapping helpers for the synthetic cross-fit.

Kept tiny and self-contained: the whole point of ``src/xfit/`` is that the cross-fit is
just the core primitives wired to a synthetic fit source, so this module only resolves
(a) where the synthetic corpus/reps live and (b) the matched-harness source->target map,
then hands ordinary ``cfg`` dicts to the core ``paths``/``reduce`` machinery.

    from src.xfit.common import load_config, iter_jobs, target_cfg
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..common.config import load_yaml, load_manifest

XFIT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = XFIT_DIR / "config.yaml"

# synthetic corpus + reps roots (reps are an extract-stage OUTPUT, hence under outputs/).
SYNTH_DATA_ROOT = Path("data/synthetic")
SYNTH_REPS_ROOT = Path("outputs/synthetic/activations")


def load_config(overrides: list[str] | None = None) -> dict:
    """Load ``src/xfit/config.yaml`` (with optional dot-path ``--set`` overrides)."""
    return load_yaml(CONFIG_PATH, overrides)


def source_tag(source: str) -> str:
    """Split-tag naming every xfit-derived root, e.g. ``captain-qwen9b`` -> ``xfit-captain-qwen9b``."""
    return f"xfit-{source}"


def synth_reps_dir(proxy: str, source: str) -> Path:
    return SYNTH_REPS_ROOT / proxy / source


def synth_data_dir(source: str) -> Path:
    return SYNTH_DATA_ROOT / source


# ── fit-pool filtering (gaia + assistantbench only) ──────────────────────────
def kept_files(source: str, pools: list[str]) -> list[str]:
    """Filenames in a synthetic source whose ``pool`` is in ``pools`` (from filename_map.csv)."""
    fmap = synth_data_dir(source) / "filename_map.csv"
    keep = []
    with fmap.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["pool"] in pools:
                keep.append(row["file"])
    return keep


# ── source <-> target resolution ─────────────────────────────────────────────
def iter_sources(cfg: dict):
    """Yield (harness, generator, source_name) over every generator in every harness."""
    for harness, gens in cfg["sources"].items():
        for gen, source in gens.items():
            yield harness, gen, source


def targets_for(cfg: dict, harness: str) -> list[dict]:
    """The matched target ``[{dataset, subset}, ...]`` for a harness."""
    return cfg["targets"][harness]


def iter_jobs(cfg: dict):
    """Yield one (harness, gen, source, dataset, subset) per scoreable cross-fit cell."""
    for harness, gen, source in iter_sources(cfg):
        for tgt in targets_for(cfg, harness):
            yield harness, gen, source, tgt["dataset"], tgt["subset"]


# ── cfg builders for the core paths/reduce machinery ─────────────────────────
def target_cfg(dataset: str, subset: str, cfg: dict, split_tag: str | None = None,
               **extra) -> dict:
    """A resolved cfg for ONE (dataset, subset) usable by ``src.common.paths`` and the
    reduce functions. Merges the dataset manifest (splits, seeds, model_paths), then pins
    proxies as ``models`` and the single subset. ``split_tag`` overrides the 325 tag
    (e.g. ``xfit-<source>``); None keeps the manifest's 325 tag for reading in-dist tables.
    """
    m = dict(load_manifest(dataset))
    m["dataset"] = dataset
    m["models"] = list(cfg["proxies"])
    m["subsets"] = [subset]
    if split_tag is not None:
        m["split_tag"] = split_tag
    m.update(extra)
    return m
