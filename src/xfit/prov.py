"""Provenance + same-tag safety for the xfit stages.

Two mechanisms, both absent from the original xfit (which relied on skip-if-exists and
would silently mix a half-populated tag with new reductions after a knob change):

* ``write_run_record`` (core provenance) — one JSON per stage invocation per dataset,
  under ``outputs/<ds>/xfit-<stage>/runs/``.
* a per-tag config stamp — the score stage writes ``scores/<tag>/xfit_config.json``
  (hash of every setting-relevant knob); every later stage calls ``ensure_tag_config``
  and refuses on a hash mismatch unless ``--force``. Cross-SETTING mixing is already
  impossible (different tag prefixes); this closes same-setting knob drift. Tags
  written before this stamp existed (the legacy runs) are grandfathered with a warning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.provenance import write_run_record
from .common import paper_cfg, setting

# every knob that changes what a tag's TSVs contain.
_HASH_KEYS = ["pools", "proxies", "sources", "methods", "poolings", "n_components",
              "ks", "target_val_ratio", "gammas", "ws", "orients", "score_norms",
              "strategies", "n_ranges"]


def config_hash(cfg: dict) -> str:
    sub = {k: cfg.get(k) for k in _HASH_KEYS}
    sub["setting"] = setting(cfg)
    if setting(cfg) == "paper":
        sub["paper"] = paper_cfg(cfg)
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()


def write_tag_config(scores_root: Path, cfg: dict, **extra) -> None:
    scores_root.mkdir(parents=True, exist_ok=True)
    (scores_root / "xfit_config.json").write_text(json.dumps(
        {"config_hash": config_hash(cfg), "setting": setting(cfg),
         **({"paper": paper_cfg(cfg)} if setting(cfg) == "paper" else {}), **extra},
        indent=2, default=str))


def ensure_tag_config(scores_root: Path, cfg: dict, force: bool = False) -> None:
    p = Path(scores_root) / "xfit_config.json"
    if not p.exists():
        print(f"[warn] no config stamp at {p} (pre-stamp tag) — proceeding")
        return
    stamped = json.loads(p.read_text()).get("config_hash")
    if stamped != config_hash(cfg) and not force:
        raise SystemExit(
            f"config hash mismatch for tag root {scores_root}: the on-disk scores were "
            f"produced under different knobs. Re-run the score stage with --force to "
            f"regenerate, or restore the original config.")


def record(cfg: dict, tcfg: dict, stage: str, outputs: list) -> None:
    """One core run record for this stage under the target dataset's tree."""
    write_run_record(tcfg, f"xfit-{stage}", [str(o) for o in outputs],
                     setting=setting(cfg), config_hash=config_hash(cfg))
