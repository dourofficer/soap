"""Aggregate weighted=True vs weighted=False comparison across split folders.

Reads weighted_true.tsv and weighted_false.tsv from:
    {root}/{split}/{model}/{subset}/

Outputs one TSV to:
    outputs-1006/weighted-comparison/table.tsv

Columns:
    split, model, subset, pooling, seed,
    step_acc_test_true,  agent_acc_test_true,
    step_acc_test_false, agent_acc_test_false,
    diff_step_acc_test,  diff_agent_acc_test

Usage:
    python compare_weighted.py
    python compare_weighted.py --root path/to/undiscounted-splits --out path/to/out.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT_DEFAULT = Path("outputs-1006/undiscounted-splits")
OUT_DEFAULT  = Path("outputs-1006/weighted-comparison/table.tsv")

ACC_COLS = ["step_acc_test", "agent_acc_test"]


def load_pair(split_dir: Path, model: str, subset: str) -> pd.DataFrame | None:
    base = split_dir / model / subset
    t_path = base / "weighted_true.tsv"
    f_path = base / "weighted_false.tsv"

    if not t_path.exists() or not f_path.exists():
        print(f"  skip (missing files): {base}")
        return None

    t = pd.read_csv(t_path, sep="\t")[ACC_COLS].mean()
    f = pd.read_csv(f_path, sep="\t")[ACC_COLS].mean()

    row = {
        "split":  split_dir.name,
        "model":  model,
        "subset": subset,
        "step_acc_test_true":    t["step_acc_test"],
        "agent_acc_test_true":   t["agent_acc_test"],
        "step_acc_test_false":   f["step_acc_test"],
        "agent_acc_test_false":  f["agent_acc_test"],
        "diff_step_acc_test":    t["step_acc_test"]  - f["step_acc_test"],
        "diff_agent_acc_test":   t["agent_acc_test"] - f["agent_acc_test"],
    }
    return pd.DataFrame([row])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--out",  type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    parts = []
    for split_dir in sorted(args.root.iterdir()):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for subset_dir in sorted(model_dir.iterdir()):
                if not subset_dir.is_dir():
                    continue
                df = load_pair(split_dir, model_dir.name, subset_dir.name)
                if df is not None:
                    parts.append(df)

    if not parts:
        print("No data found — check --root path.")
        return

    table = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False, na_rep="")
    print(f"wrote {args.out}  ({len(table)} rows)")


if __name__ == "__main__":
    main()