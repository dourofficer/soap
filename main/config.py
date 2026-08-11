"""Config loading and every output path, in one place.

One YAML per dataset (``configs-main/<ds>.yaml``) owns everything: the corpus, the
backbones, the partition, the frozen seed triples and both sweep grids. There is no
manifest/stage-config merge and no split-tag directory level — with the seeds pinned in
the config, a second partition would be a different experiment, not a sibling tree.

That makes ``run_stamp.json`` load-bearing: it records the knobs a stage's outputs
depend on, is written next to them the first time, and is VERIFIED on every later run.
Editing ``splits`` or ``seeds`` and re-running therefore fails loudly instead of
silently mixing rows computed under two different partitions into one table.

    from main.config import load_config, results_root, seeds_for
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

N_COMPONENTS = 20
KS = (1, 3)
STRATEGIES = ("backprop", "succ-strong", "succ-near")
SEEDS_PER_SUBSET = 3

# Knobs that change what a stage's numbers MEAN. Stamped and verified; anything else
# (device, dtype, tqdm) can change freely between runs.
STAMP_KEYS = ("splits", "seeds", "n_components", "positions", "ensemble",
              "n_ranges", "gammas", "ws", "strategies", "gt")


# ── loading ─────────────────────────────────────────────────────────────────
def _apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """``--set a.b=value`` dot-paths; values parsed as YAML so [1,2]/true/3 work."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _apply_overrides(cfg, overrides)
    cfg.setdefault("gt", False)
    cfg.setdefault("ks", list(KS))
    cfg.setdefault("n_components", N_COMPONENTS)
    cfg.setdefault("strategies", list(STRATEGIES))
    if "dataset" not in cfg:
        raise SystemExit(f"{path}: missing `dataset`")
    return cfg


def dataset(cfg: dict) -> str:
    return cfg["dataset"]


# ── roots ───────────────────────────────────────────────────────────────────
def results_root(cfg: dict) -> Path:
    """``results-gt/<ds>`` when ``gt``, else ``results-nogt/<ds>``.

    Two trees, never one with a flag column: the with-GT run sees a different context
    (the pinned [question, answer] block), so its activations and attention are
    different artifacts, not a variant of the same ones.
    """
    base = cfg.get("results_base") or ("results-gt" if cfg["gt"] else "results-nogt")
    return REPO_ROOT / base / dataset(cfg)


def data_root(cfg: dict) -> Path:
    return REPO_ROOT / (cfg.get("data_dir") or f"data/{dataset(cfg)}")


def reps_root(cfg: dict) -> Path:
    return results_root(cfg) / "activations"


def attn_root(cfg: dict) -> Path:
    return results_root(cfg) / "attention"


def sweep_dir(cfg: dict, model: str, subset: str) -> Path:
    return results_root(cfg) / "sweep" / model / subset


def select_dir(cfg: dict) -> Path:
    return results_root(cfg) / "select"


def repro_dir(cfg: dict, model: str, subset: str) -> Path:
    return results_root(cfg) / "reproduce" / model / subset


# ── frozen seeds ────────────────────────────────────────────────────────────
def seeds_for(cfg: dict, subset: str) -> list[int]:
    """The subset's frozen seed triple — the same one for every backbone."""
    seeds = cfg.get("seeds")
    if not isinstance(seeds, dict) or subset not in seeds:
        raise SystemExit(
            f"config has no frozen seeds for subset {subset!r}; expected a mapping like\n"
            f"  seeds:\n    {subset}: [3, 4, 5]")
    got = [int(s) for s in seeds[subset]]
    if len(got) != SEEDS_PER_SUBSET:
        raise SystemExit(f"subset {subset!r}: expected {SEEDS_PER_SUBSET} seeds, got {got}")
    return got


# ── drift guard ─────────────────────────────────────────────────────────────
def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _stamp(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in STAMP_KEYS}


def check_stamp(cfg: dict, out_dir: Path, force: bool = False) -> None:
    """Write ``run_stamp.json`` the first time; afterwards REFUSE a changed stamp.

    Without a split tag in the path, a changed ``splits``/``seeds`` would overwrite a
    table computed on a different partition and nothing would look wrong. ``--force``
    is the explicit "yes, recompute this tree under the new knobs" escape hatch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_stamp.json"
    want = _stamp(cfg)
    if path.exists() and not force:
        have = json.loads(path.read_text()).get("stamp", {})
        drift = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
        if drift:
            lines = "\n".join(f"    {k}: {old!r} -> {new!r}" for k, (old, new) in drift.items())
            raise SystemExit(
                f"config drift under {out_dir}:\n{lines}\n"
                f"These change what the existing rows mean. Re-run with --force to "
                f"recompute, or point `results_base` at a fresh tree.")
    path.write_text(json.dumps(
        {"stamp": want, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "git_sha": _git_sha()}, indent=2, default=str))
