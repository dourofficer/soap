"""Build undiscounted result tables — one TSV per (model, subset).

Per output TSV, 18 rows in three sections of 6 rows each
(2 poolings x 3 seeds):
  - svd                 (best across all SVD configs)
  - classifier_pseudo   (labels == 'pseudo')
  - classifier_oracle   (labels == 'oracle')

Best = max (step_acc_test, agent_acc_test) lexicographically, per
(pooling, seed). Rows within a section are sorted by the same key.


python -m experiments.reports.build_undiscounted_tables
# or with overrides:
python -m attribscope.utils.build_undiscounted_tables \
    --clf-root outputs/classifier/hidden \
    --out-root outputs/tables/undiscounted
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SVD_ROOT_DEFAULT = Path("outputs/projections")
OUT_ROOT_DEFAULT = Path("outputs/fake-tables/undiscounted")

MODELS  = ["deepseek-8b", "llama-3.1-8b", "qwen3-8b", "qwen3-14b"]
SUBSETS = ["algorithm-generated", "hand-crafted"]

SEED_RE = re.compile(r"seed-(\d+)\.tsv$")

OUT_COLS = [
    "strategy", "weight", "pooling", "method", "c_begin", "c_end", "centered",
    "threshold", "seed",
    "step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test",
]

POOLING_ORDER = ["last", "mean"]


def _read_with_seed(path: Path) -> pd.DataFrame:
    m = SEED_RE.search(path.name)
    if not m:
        raise ValueError(f"can't parse seed from {path.name}")
    df = pd.read_csv(path, sep="\t")
    df["seed"] = int(m.group(1))
    return df


def _load_concat(metrics_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(metrics_dir.glob(f"{prefix}_pooling-*_seed-*.tsv"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.tsv under {metrics_dir}")
    return pd.concat([_read_with_seed(p) for p in files], ignore_index=True)


def _best_per_pooling_seed(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values(["step_acc_test", "agent_acc_test"],
                           ascending=False, kind="mergesort")
              .groupby(["pooling", "seed"], as_index=False, sort=False)
              .first())


def _sort_section(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pooling"] = pd.Categorical(df["pooling"],
                                   categories=POOLING_ORDER, ordered=True)
    out = (df.sort_values(["pooling", "seed"], kind="mergesort")
             .reset_index(drop=True))
    out["pooling"] = out["pooling"].astype(str)
    return out


def section_svd(metrics_dir: Path) -> pd.DataFrame:
    raw = _load_concat(metrics_dir, "svd")
    best = _best_per_pooling_seed(raw).assign(strategy="svd", threshold=pd.NA)
    return _sort_section(best[OUT_COLS])


def build_table(metrics_dir: Path) -> pd.DataFrame:
    return pd.concat([
        section_svd(metrics_dir),
    ], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clf-root", type=Path, default=SVD_ROOT_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for subset in SUBSETS:
            metrics_dir = args.clf_root / model / subset
            if not metrics_dir.exists():
                print(f"skip (missing dir): {metrics_dir}")
                continue
            try:
                table = build_table(metrics_dir)
            except FileNotFoundError as e:
                print(f"skip ({e})")
                continue
            dst = args.out_root / f"{model}/{subset}.tsv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            table.to_csv(dst, sep="\t", index=False, na_rep="")
            print(f"wrote {dst}  ({len(table)} rows)")


if __name__ == "__main__":
    main()