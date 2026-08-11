"""Canonical config loading: dataset manifest + thin stage config + --set overrides.

The dataset manifest (``configs/datasets/<dataset>.yaml``) is the single source of
truth for knobs shared across stages (models, model_paths, subsets, max_tokens,
split ratios, seeds, ks). A stage config declares ``dataset: <name>`` and only its
own axes; :func:`resolve` merges the manifest underneath it.

Precedence (lowest -> highest): manifest < stage config < ``--set`` overrides.

Standalone use:
    python -c "from src.common.config import load_stage_config as L; \
               print(L('configs/score/correct-error.yaml'))"
"""
from __future__ import annotations

from pathlib import Path

import yaml

# config.py -> common -> src -> v2/  (parents[2] == v2 root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "configs" / "datasets"


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
    """Load a YAML file and apply dot-path overrides. ``None`` path -> ``{}``."""
    cfg = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = cfg or {}
    return _apply_overrides(cfg, overrides or [])


def load_manifest(dataset: str) -> dict:
    """Read ``configs/datasets/<dataset>.yaml``."""
    path = DATASETS_DIR / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset manifest for {dataset!r} at {path}. "
            f"Create it (see configs/datasets/correct-error.yaml)."
        )
    return yaml.safe_load(path.read_text())


def resolve(stage_cfg: dict) -> dict:
    """Merge the named dataset manifest underneath a stage config."""
    dataset = stage_cfg.get("dataset")
    if dataset is None:
        return stage_cfg
    merged = dict(load_manifest(dataset))
    merged.update(stage_cfg)  # stage config / --set win
    merged["dataset"] = dataset
    return merged


def load_stage_config(path: Path | str, overrides: list[str] | None = None) -> dict:
    """Read stage YAML, apply overrides, then merge the manifest."""
    return resolve(load_yaml(path, overrides))


def split_tag(splits: dict, override: str | int | None = None) -> str:
    """Split-ratio tag baked into output paths, e.g. 0.3/0.2/0.5 -> ``"325"``.

    An explicit ``split_tag`` (manifest or --set) wins.
    """
    if override is not None:
        return str(override)
    t, v, s = splits["train"], splits["val"], splits["test"]
    return f"{int(round(t * 10))}{int(round(v * 10))}{int(round(s * 10))}"
