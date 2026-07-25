"""Shared plumbing for the datagen pipeline: repo-root bootstrap, config
loading, and the endpoint registry.

Import this first in every datagen entry script — it puts the repo root on
sys.path so `src.…` / `experiments.…` imports resolve when scripts are invoked
by path (`python datagen/<script>.py` from the repo root).

Pattern follows exp-synthetic-correct/common.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

DATAGEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATAGEN_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The v2 restructure moved the shared config loader from experiments/_common
# to src/common; the function is unchanged.
from src.common.config import load_yaml  # noqa: E402

CONFIGS_DIR = DATAGEN_DIR / "configs"
POOLS_DIR = DATAGEN_DIR / "pools" / "data"
RUNS_DIR = DATAGEN_DIR / "runs"
TRACE_ELEPHANT = DATAGEN_DIR / "TraceElephant"


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(name: str, overrides: list[str] | None = None) -> dict:
    """Load `datagen/configs/<name>.yaml` (+ `--set key.sub=value` overrides).

    `name` may be a bare stem ("serve") or a path to a YAML file.
    """
    path = Path(name)
    if not path.suffix:
        path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no datagen config at {path}")
    return load_yaml(path, overrides)


# ── Endpoint registry ─────────────────────────────────────────────────────────

def endpoints(cfg: dict | None = None) -> dict[str, dict]:
    """The `{shorthand: {base_url, model, ...}}` registry from serve.yaml.

    Every model role in the pipeline (agent backbone, judge, injector,
    verifier, replay oracle) resolves its endpoint through here, so swapping in
    a larger served model is a one-line config change.
    """
    cfg = cfg or load_cfg("serve")
    reg = {}
    for name, spec in cfg["models"].items():
        reg[name] = {
            "base_url": f"http://{spec.get('host', '127.0.0.1')}:{spec['port']}/v1",
            "model": spec.get("served_model_name", name),
            "api_key": spec.get("api_key", "EMPTY"),
        }
    return reg


def resolve_endpoint(ref: str | dict, cfg: dict | None = None) -> dict:
    """Resolve a role's endpoint: either a registry shorthand or an inline dict.

    >>> resolve_endpoint("qwen3.5-9b")
    {'base_url': 'http://127.0.0.1:8001/v1', 'model': 'qwen3.5-9b', 'api_key': 'EMPTY'}
    """
    if isinstance(ref, dict):
        return {"api_key": "EMPTY", **ref}
    reg = endpoints(cfg)
    if ref not in reg:
        raise KeyError(f"unknown endpoint {ref!r}; known: {sorted(reg)}")
    return reg[ref]


# ── Pools ─────────────────────────────────────────────────────────────────────

def pool_path(pool: str) -> Path:
    return POOLS_DIR / f"{pool}.jsonl"


def load_pool(pool: str, limit: int | None = None) -> list[dict]:
    """Read a prepared pool jsonl into a list of task dicts."""
    import json

    path = pool_path(pool)
    if not path.exists():
        raise FileNotFoundError(
            f"pool {pool!r} not prepared at {path}. "
            f"Run: python datagen/pools/prepare.py --pool {pool}"
        )
    with path.open() as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
    return tasks[:limit] if limit else tasks
