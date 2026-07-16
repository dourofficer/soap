"""Driver for exp-synthetic-correct: expand the grid, shell out per cell.

Stages (in dependency order):
  svd      grid (models × poolings × seeds) → run_svd.py       [GPU]
  tables   build_tables --stage undisc (add --validate to recheck tensors)
  rescore  grid (models × targets)         → run_rescore.py    [CPU-bound]
  disc     build_tables --stage disc
  summary  build_tables --stage summary
  all      everything above in order

    python exp-synthetic-correct/sweep.py --stage all --dry-run
    CUDA_VISIBLE_DEVICES=0 python exp-synthetic-correct/sweep.py --stage svd
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

import common

STAGE_ORDER = ["svd", "tables", "rescore", "disc", "summary"]


def _run(script: str, argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(common.EXP_DIR / script), *argv]
    print("\n\033[32m" + " ".join(shlex.quote(c) for c in cmd) + "\033[0m")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=common.REPO_ROOT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--stage", required=True, choices=STAGE_ORDER + ["all"])
    p.add_argument("--validate", action="store_true",
                   help="tables stage: recompute metrics from stored tensors")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = common.load_cfg(args.config, args.overrides)
    passthrough = ([] if args.config is None else ["--config", str(args.config)])
    for ov in args.overrides:
        passthrough += ["--set", ov]

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "svd":
            for model in cfg["models"]:
                for pooling in cfg["poolings"]:
                    for seed in cfg["seeds"]:
                        _run("run_svd.py",
                             ["--model", model, "--pooling", pooling,
                              "--seed", str(seed), *passthrough],
                             args.dry_run)
        elif stage == "tables":
            extra = ["--validate"] if args.validate else []
            _run("build_tables.py",
                 ["--stage", "undisc", *extra, *passthrough], args.dry_run)
        elif stage == "rescore":
            for model in cfg["models"]:
                for ds, subset in common.iter_targets(cfg):
                    _run("run_rescore.py",
                         ["--model", model, "--dataset", ds,
                          "--subset", subset, *passthrough],
                         args.dry_run)
        elif stage in ("disc", "summary"):
            _run("build_tables.py", ["--stage", stage, *passthrough],
                 args.dry_run)


if __name__ == "__main__":
    main()
