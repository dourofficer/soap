"""One CLI for every stage.

    python -m main extract   --config configs-main/ww.yaml [--stage activations]
    python -m main sweep     --config configs-main/ww.yaml
    python -m main select    --config configs-main/ww.yaml
    python -m main reproduce --config configs-main/ww.yaml --row backprop

Shared flags: ``--set key.subkey=value`` (repeatable, YAML-parsed), ``--model``,
``--subset``, ``--device``, ``--force``, ``--dry-run``. With-GT is ``--set gt=true``.

There is deliberately NO ``--seed``: seeds are frozen per subset in the config, and a
reported number is by definition the mean over that triple, so narrowing to one seed
would produce a number that means nothing.
"""
from __future__ import annotations

import argparse

from .config import load_config


def _narrow(cfg: dict, key: str, value) -> None:
    if value is None:
        return
    have = cfg.get(key, [])
    if value not in have and str(value) not in [str(v) for v in have]:
        raise SystemExit(f"{value!r} not in config {key}={have}")
    cfg[key] = [value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    for name in ("extract", "sweep", "select", "reproduce"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--set", action="append", dest="overrides", default=[],
                       metavar="KEY=VALUE", help="dot-path YAML override (repeatable)")
        s.add_argument("--model")
        s.add_argument("--subset")
        s.add_argument("--device")
        s.add_argument("--force", action="store_true")
        s.add_argument("--dry-run", action="store_true")
        if name == "extract":
            s.add_argument("--stage", dest="extract_stage",
                           choices=["activations", "attention"], default=None,
                           help="run only one extraction stage (default: both)")
            s.add_argument("--start-idx", type=int, default=0)
            s.add_argument("--end-idx", type=int, default=None)
        if name == "reproduce":
            s.add_argument("--row", default="all",
                           help="selection row label: svd | backprop | succ-strong | "
                                "succ-near | all")
            s.add_argument("--split", default="test", choices=["val", "test", "all"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.device:
        cfg["device"] = args.device
    cfg["force"] = bool(args.force)
    cfg["dry_run"] = bool(args.dry_run)
    _narrow(cfg, "models", args.model)
    _narrow(cfg, "subsets", args.subset)

    if args.stage == "extract":
        from .extract import run_extract
        cfg["start_idx"] = args.start_idx
        cfg["end_idx"] = args.end_idx
        stages = ((args.extract_stage,) if args.extract_stage
                  else ("activations", "attention"))
        run_extract(cfg, stages)
    elif args.stage == "sweep":
        from .sweep import run_sweep
        run_sweep(cfg)
    elif args.stage == "select":
        from .sweep import run_select
        run_select(cfg)
    elif args.stage == "reproduce":
        from .reproduce import run_reproduce
        run_reproduce(cfg, rows=args.row, split=args.split)
    return 0
