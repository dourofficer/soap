"""Shared plumbing for exp-synthetic-correct: config loading, dataset-root
derivation, the two split helpers, and results-path builders.

Import this first in every entry script — it puts the repo root on sys.path so
`src.…` / `experiments.…` imports resolve when scripts are invoked by path
(`python exp-synthetic-correct/<script>.py` from the repo root; the directory
name is hyphenated, so `python -m` is not available).
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._common import paths  # noqa: E402
from experiments._common.config import load_manifest, load_yaml  # noqa: E402
from src.svd.reproduce import _validate_and_derive_split_ratios  # noqa: E402
from src.utils.utils import split_data  # noqa: E402

DEFAULT_CONFIG = EXP_DIR / "config.yaml"

# The 7 columns that uniquely identify one scoring config within a
# (model, subset, pooling, seed) run — shared by selection, persistence,
# rescoring and the cross-checks.
CONFIG_KEYS = ["position", "pooling", "method",
               "c_begin", "c_end", "centered", "weighted"]


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(path: Path | str | None = None,
             overrides: list[str] | None = None) -> dict:
    """Load config.yaml (+ --set overrides) and derive per-dataset roots.

    Adds cfg["datasets"][ds] = {"data_root", "reps_root", "attn_root"} for the
    source dataset and every target dataset, all read from the dataset
    manifests / paths module.
    """
    cfg = load_yaml(path or DEFAULT_CONFIG, overrides)

    ds_names = set(cfg["targets"]) | {cfg["source"]["dataset"]}
    cfg["datasets"] = {}
    for ds in sorted(ds_names):
        manifest = dict(load_manifest(ds))
        manifest["dataset"] = ds
        cfg["datasets"][ds] = {
            "data_root": Path(manifest["data_root"]),
            "reps_root": paths.reps_root(manifest),
            "attn_root": paths.attn_root(manifest),
        }
    return cfg


def iter_targets(cfg: dict):
    """Yield (dataset, subset) pairs in config order."""
    for ds, subsets in cfg["targets"].items():
        for subset in subsets:
            yield ds, subset


# ── Splits ────────────────────────────────────────────────────────────────────

def list_traj_files(rep_dir: Path) -> list[str]:
    """The one canonical trajectory-file lister: names sorted by int stem.

    Any other ordering breaks seed-reproducibility of the splits.
    """
    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, f"No .safetensors files in {rep_dir}"
    return [f.name for f in files]


def split_source(files: list[str], splits: dict, seed: int):
    """Nested train/val/test split — identical mechanism to src/svd/score.py."""
    r_trval_test, r_train_val = _validate_and_derive_split_ratios(
        splits["train"], splits["val"], splits["test"])
    trval_files, test_files = split_data(files, r_trval_test, seed)
    train_files, val_files = split_data(trval_files, r_train_val, seed)
    return train_files, val_files, test_files


def split_target(files: list[str], val_ratio: float, seed: int):
    """Train-less val/test split: one split_data pass, val = first part.

    Deliberately does NOT go through _validate_and_derive_split_ratios, which
    rejects any zero split.
    """
    val_files, test_files = split_data(files, val_ratio, seed)
    return val_files, test_files


# ── Results paths (everything under results_root) ─────────────────────────────

def _results(cfg: dict) -> Path:
    return Path(cfg["results_root"])


def svd_dir(cfg, model, ds, subset) -> Path:
    return _results(cfg) / "svd" / model / ds / subset


def scores_dir(cfg, model, ds, subset) -> Path:
    return _results(cfg) / "base-scores" / model / ds / subset


def undisc_dir(cfg, model, ds, subset) -> Path:
    return _results(cfg) / "undiscounted" / model / ds / subset


def sweep_dir(cfg, model, ds, subset) -> Path:
    return _results(cfg) / "rescore" / "sweep" / model / ds / subset


def reduced_dir(cfg, model, ds, subset) -> Path:
    return _results(cfg) / "rescore" / "reduced" / model / ds / subset


def summary_dir(cfg) -> Path:
    return _results(cfg) / "summary"


def svd_tsv(cfg, model, ds, subset, pooling, seed) -> Path:
    return svd_dir(cfg, model, ds, subset) / f"svd_pooling-{pooling}_seed-{seed}.tsv"


def scores_pt(cfg, model, ds, subset, pooling, seed) -> Path:
    return scores_dir(cfg, model, ds, subset) / f"selected_pooling-{pooling}_seed-{seed}.pt"
