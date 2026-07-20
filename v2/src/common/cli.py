"""Uniform CLI contract for every stage runner.

    python -m src.<stage>.run --config configs/<stage>/<ds>.yaml \\
        [--set key.subkey=value ...]   # dot-path YAML overrides (highest precedence)
        [--model M] [--subset S] [--seed N]   # narrowing: intersect config lists
        [--device cuda|cpu] [--dry-run] [--force]

Precedence: manifest < stage config < --set < narrowing flags. Narrowing flags
restrict the corresponding manifest list to a single element (error if not a
member). ``--force`` re-runs cells whose output already exists.
"""
from __future__ import annotations

import argparse

from .config import load_stage_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Stage YAML (declares dataset: <ds>).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Dot-path override, e.g. --set device=cpu.")
    p.add_argument("--model", default=None, help="Narrow to one model shorthand.")
    p.add_argument("--subset", default=None, help="Narrow to one subset.")
    p.add_argument("--seed", type=int, default=None, help="Narrow to one seed.")
    p.add_argument("--device", default=None, help="Override cfg.device.")
    p.add_argument("--dry-run", action="store_true", help="Print the work plan, do nothing.")
    p.add_argument("--force", action="store_true", help="Re-run cells with existing outputs.")
    return p


def _narrow(cfg: dict, key: str, value) -> None:
    """Restrict cfg[key] (a list) to [value]; error if value is not a member."""
    if value is None:
        return
    have = cfg.get(key, [])
    # seeds are ints, models/subsets strings; compare with coercion tolerance.
    if value not in have and str(value) not in [str(x) for x in have]:
        raise SystemExit(f"--{key[:-1]} {value!r} not in cfg[{key!r}] = {have}")
    cfg[key] = [value]


def load_and_narrow(args: argparse.Namespace) -> dict:
    """Resolve config, apply --device override, then apply model/subset/seed narrowing."""
    cfg = load_stage_config(args.config, args.overrides)
    if args.device is not None:
        cfg["device"] = args.device
    cfg["force"] = bool(getattr(args, "force", False) or cfg.get("force", False))
    cfg["dry_run"] = bool(getattr(args, "dry_run", False))
    _narrow(cfg, "models", args.model)
    _narrow(cfg, "subsets", args.subset)
    _narrow(cfg, "seeds", args.seed)
    return cfg
