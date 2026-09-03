"""Build the prompt-based rows the manuscript's open-backbone blocks need.

Table~\\ref{tab:main} reports prompt-based methods under a GPT-4o judge and gives
Qwen3.5-9B and DeepSeek-R1-Distill-Llama-8B their own blocks, which today hold
only the training-based methods and SOAP. This writes the twelve rows that fill
the prompting group of those two blocks.

The numbers come from `results-prompting/by_column.tsv`, which
`scripts/prompting/evaluate.py` scores on the same frozen seed triples and test
splits SOAP uses, so a prompting cell and a SOAP cell count the same
trajectories. Percentages, two decimals — the .tex convention. A blank cell means
that cell has no predictions yet.

    python scripts/tables/open_backbone_rows.py
    python scripts/tables/open_backbone_rows.py --gt        # the with-GT twin
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results-prompting" / "by_column.tsv"

COLUMNS = ["WW-AG", "WW-HC", "CE", "TE-Cap", "TE-Mag"]

# (display name, method stem) in the row order of tab:main's prompting group.
ROWS = [("All-at-Once", "all_at_once"), ("Step-by-Step", "step_by_step"),
        ("Binary Search", "binary_search"), ("CORRECT", "correct"),
        ("CHIEF", "chief"), ("RAFFLES", "raffles")]

BACKBONES = [("Qwen3.5-9B", "qwen3.5-9b"),
             ("DeepSeek-R1-Distill-Llama-8B", "deepseek-8b")]


def build(df: pd.DataFrame, gt: bool, metric: str) -> list[list[str]]:
    out = [["Backbone", "Method"] + COLUMNS]
    for display, judge in BACKBONES:
        for name, stem in ROWS:
            cells = []
            for col in COLUMNS:
                hit = df[(df["with_gt"] == gt) & (df["judge"] == judge)
                         & (df["method"] == stem) & (df["column"] == col)]
                cells.append("" if hit.empty
                             else f"{100.0 * float(hit.iloc[0][metric]):.2f}")
            out.append([display, name] + cells)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", action="store_true",
                   help="the with-GT tree instead of the manuscript's without-GT one")
    p.add_argument("--out-dir", default=str(REPO / "tables"))
    args = p.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"no {SOURCE} — run scripts/prompting/evaluate.py first")
    df = pd.read_csv(SOURCE, sep="\t")

    suffix = "with_gt" if args.gt else "without_gt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in (("step_acc", "step"), ("agent_acc", "agent")):
        rows = build(df, args.gt, metric)
        path = out_dir / f"open_backbones_{label}_{suffix}.tsv"
        with path.open("w", newline="") as fh:
            csv.writer(fh, delimiter="\t").writerows(rows)
        print(f"  {path}")
        for r in rows:
            print("    " + "\t".join(f"{c:<28}" if i < 2 else c
                                     for i, c in enumerate(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
