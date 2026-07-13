"""Canonical config loading for the SVD + CRR sweep drivers.

This replaces the ``load_cfg`` that was copy-pasted verbatim in every sweep
driver and both v2 report builders. It also introduces the **dataset manifest**
(``experiments/datasets/<dataset>.yaml``) as the single source of truth for the
knobs shared across stages (models, model_paths, subsets, data_root, max_tokens,
split ratios). A stage config declares ``dataset: <name>`` and only its own
sweep axes; :func:`resolve` merges the manifest underneath it.

Precedence (lowest → highest): manifest < stage config < ``--set`` overrides.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "experiments" / "datasets"


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.subkey=value`` dot-path overrides (values parsed as YAML)."""
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_yaml(path: Path | str | None, overrides: list[str] | None = None) -> dict:
    """Load a YAML file and apply dot-path overrides. ``None`` path → ``{}``."""
    cfg = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = cfg or {}
    return _apply_overrides(cfg, overrides or [])


def load_manifest(dataset: str) -> dict:
    """Read ``experiments/datasets/<dataset>.yaml`` (the per-dataset manifest)."""
    path = DATASETS_DIR / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset manifest for {dataset!r} at {path}. "
            f"Create it (see experiments/datasets/correct-error.yaml)."
        )
    return yaml.safe_load(path.read_text())


def resolve(stage_cfg: dict) -> dict:
    """Merge the named dataset manifest underneath a stage config.

    The stage config must carry a ``dataset`` key. Manifest keys fill in any the
    stage config does not set; the stage config (and, upstream, ``--set``
    overrides already applied to it) win on conflict.
    """
    dataset = stage_cfg.get("dataset")
    if dataset is None:
        # No manifest requested — return as-is (back-compat with flat configs).
        return stage_cfg
    merged = dict(load_manifest(dataset))
    merged.update(stage_cfg)  # stage config / overrides win
    merged["dataset"] = dataset
    return merged


def load_stage_config(path: Path | str, overrides: list[str] | None = None) -> dict:
    """Full load: read stage YAML, apply overrides, then merge the manifest."""
    return resolve(load_yaml(path, overrides))


def split_tag(splits: dict, override: str | int | None = None) -> str:
    """Split-ratio tag baked into output paths, e.g. 0.3/0.2/0.5 → ``"325"``.

    Derived as the train/val/test fractions in tenths, concatenated. An explicit
    manifest ``split_tag`` wins (needed for legacy ratios like 0.25/0.25/0.5 →
    ``"112"`` that are not expressible in whole tenths).
    """
    if override is not None:
        return str(override)
    t, v, s = splits["train"], splits["val"], splits["test"]
    return f"{int(round(t * 10))}{int(round(v * 10))}{int(round(s * 10))}"
