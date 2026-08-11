"""V1: the triple sweep must reproduce the production run exactly.

Every seed triple currently frozen in `configs-main/` is a CONSECUTIVE window, so all 44
production cells are re-swept by the 48-window sweep:

    ww / algorithm-generated      s03-04-05
    ww / hand-crafted             s13-14-15
    traceelephant / magentic      s08-09-10
    traceelephant / captain       s14-15-16
    correct-error / all 7 subsets s17-18-19

x 2 backbones x 2 GT settings = 44 cells, exactly one full pass. Those cells must agree
numerically, row for row, with `results-nogt/` / `results-gt/`. Any drift means injecting
seeds via `--set` is not equivalent to declaring them in the config — which would
invalidate the whole driver.

    python scripts/main/check_sweep_repro.py
    python scripts/main/check_sweep_repro.py --triples s03-04-05,s17-18-19   # early gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
KEY = ["model", "subset", "seed", "position", "c_begin", "c_end",
       "strategy", "layer_range", "gamma", "w"]


def expected_cells():
    """(gt, triple_tag, dataset, model, subset) whose production run used that triple."""
    out = []
    for ds in ("ww", "traceelephant", "correct-error"):
        cfg = yaml.safe_load((REPO / "configs-main" / f"{ds}.yaml").read_text())
        for subset, seeds in cfg["seeds"].items():
            t = "s" + "-".join(f"{x:02d}" for x in seeds)
            for gt in (False, True):
                for model in cfg["models"]:
                    out.append((gt, t, ds, model, subset))
    return out


def compare(new: Path, ref: Path) -> str | None:
    if not new.exists():
        return f"missing swept file {new}"
    if not ref.exists():
        return f"missing reference {ref}"
    a = pd.read_csv(new, sep="\t").sort_values(KEY).reset_index(drop=True)
    b = pd.read_csv(ref, sep="\t").sort_values(KEY).reset_index(drop=True)
    if len(a) != len(b):
        return f"row count {len(a)} != {len(b)}"
    if list(a.columns) != list(b.columns):
        return f"columns differ: {set(a.columns) ^ set(b.columns)}"
    for c in KEY:
        if not a[c].astype(str).equals(b[c].astype(str)):
            return f"key column {c!r} differs"
    metrics = [c for c in a.columns if "_acc_" in c]
    for c in metrics:
        d = (a[c] - b[c]).abs().max()
        if d > 0:
            return f"metric {c} differs by up to {d:.3e}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triples", default="all")
    args = p.parse_args()
    want = None if args.triples == "all" else set(args.triples.split(","))

    problems, checked, skipped = [], 0, 0
    for gt, t, ds, model, subset in expected_cells():
        if want is not None and t not in want:
            continue
        gt_dir = "gt" if gt else "nogt"
        new = REPO / "results-sweep" / gt_dir / t / ds / "sweep" / model / subset / "sweep.tsv"
        ref = (REPO / f"results-{gt_dir}" / ds / "sweep" / model / subset / "sweep.tsv")
        if not new.exists():
            skipped += 1
            continue
        checked += 1
        err = compare(new, ref)
        if err:
            problems.append(f"{gt_dir}/{t}/{ds}/{model}/{subset}: {err}")

    print("=" * 70)
    if skipped:
        print(f"not swept yet, skipped: {skipped}")
    if problems:
        print(f"REPRODUCTION FAILED — {len(problems)} of {checked} cells differ:")
        for x in problems[:20]:
            print("  -", x)
        return 1
    if not checked:
        print("nothing to check yet")
        return 0
    print(f"REPRODUCTION OK: all {checked} cells identical to the production run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
