"""Aggregate discounted SVD tables across data splits for one (model, subset).

For each split directory found under --results-dir, reads
    {results_dir}/{split}/{model}/{subset}/svd.tsv
and concatenates them into one table, in ascending split order, with a single
empty row between consecutive splits. Drops columns: strategy, threshold, weighted.

Output separator follows the --out extension (.csv -> comma, else tab).

Usage:
    python aggregate_splits.py \
        --results-dir outputs-1006/discounted-splits/reduced \
        --model qwen3-8b \
        --subset hand-crafted \
        --out outputs-1006/discounted-splits/agg/qwen3-8b__hand-crafted.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DROP_COLS = ["strategy", "threshold", "weighted", "method"]

# Columns to surface first (in this order); remaining columns keep their order.
LEAD_COLS = ["split", "seed", "pooling", "position",
             "c_begin", "c_end", "centered"]

# Explicit split-id -> train:val:test ratio labels (edit as needed).
# Any id not listed falls back to the computed rule in split_label().
SPLIT_LABELS = {
    "112": "25:25:50",
    "226": "20:20:60",
    "316": "30:10:60",
    "325": "30:20:50",
    "424": "40:20:40",
}


def split_label(name: str) -> str:
    """Ratio label for a split id: each digit is a part, scaled to sum to 100."""
    if name in SPLIT_LABELS:
        return SPLIT_LABELS[name]
    if name.isdigit() and sum(int(c) for c in name) > 0:
        parts = [int(c) for c in name]
        total = sum(parts)
        return ":".join(str(round(100 * p / total)) for p in parts)
    return name


def _split_sort_key(name: str):
    # numeric split ids (112, 226, ...) sort numerically; anything else, lexically
    return (0, int(name)) if name.isdigit() else (1, name)


def aggregate(results_dir: Path, model: str, subset: str) -> pd.DataFrame:
    splits = sorted((p.name for p in results_dir.iterdir() if p.is_dir()),
                    key=_split_sort_key)

    pieces, cols = [], None
    for split in splits:
        path = results_dir / split / model / subset / "svd.tsv"
        if not path.exists():
            print(f"skip (missing): {path}")
            continue

        df = pd.read_csv(path, sep="\t").drop(columns=DROP_COLS, errors="ignore")
        df.insert(0, "split", split_label(split))
        if cols is None:
            cols = list(df.columns)
        if pieces:                                      # blank separator row
            pieces.append(pd.DataFrame([[""] * len(cols)], columns=cols))
        pieces.append(df)
        print(f"  + {split}: {len(df)} rows")

    if not pieces:
        raise SystemExit(
            f"no svd.tsv found for {model}/{subset} under {results_dir}")
    table = pd.concat(pieces, ignore_index=True)
    ordered = [c for c in LEAD_COLS if c in table.columns] + \
              [c for c in table.columns if c not in LEAD_COLS]
    return table[ordered]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="dir containing split folders, e.g. .../reduced")
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output file path")
    args = ap.parse_args()

    table = aggregate(args.results_dir, args.model, args.subset)

    sep = "," if args.out.suffix.lower() == ".csv" else "\t"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep=sep, index=False, na_rep="")
    print(f"wrote {args.out}  ({len(table)} rows incl. separators)")


if __name__ == "__main__":
    main()