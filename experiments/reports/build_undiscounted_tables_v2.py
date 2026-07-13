"""Build undiscounted result tables — one set of TSVs per (model, subset).

Per (model, subset), three output files under out_root/model/subset/:
  - weighted_true.tsv   best configs with weighted=True  fixed
  - weighted_false.tsv  best configs with weighted=False fixed
  - weighted_all.tsv    best configs treating weighted as a hyperparameter

Each file has one row per (pooling, seed), strategy=svd only. Best = argmax
(step_acc_test, agent_acc_test) lexicographically, per (pooling, seed). Rows
sorted (pooling [last, mean], seed).

Roots come from the dataset manifest when the config declares ``dataset:``:
    svd_root = weighted-projections/<tag>,  out_root = undiscounted-splits/<tag>.
Explicit ``svd_root`` / ``out_root`` / ``models`` / ``subsets`` in the config
still override (back-compat with the old flat configs).

Usage:
    python -m experiments.reports.build_undiscounted_tables_v2 \
        --config experiments/reports/configs/undiscounted_v2_correct-error.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments._common.config import load_yaml, resolve
from experiments._common import paths
from experiments.reports._common import load_concat, best_per_group, sort_section

OUT_COLS = [
    "strategy", "position", "pooling", "method", "c_begin", "c_end",
    "centered", "weighted", "threshold", "seed",
    "step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test",
]

SORT_METRICS = ["step_acc_test", "agent_acc_test"]
GROUP_KEYS   = ["pooling", "seed"]


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(strategy="svd", threshold=pd.NA)
    if "weighted" not in df.columns:
        df["weighted"] = pd.NA
    cols = [c for c in OUT_COLS if c in df.columns]
    return sort_section(df[cols])


def build_tables(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """Return dict with keys 'weighted_true', 'weighted_false', 'weighted_all'."""
    raw = load_concat(metrics_dir, "svd")
    if "weighted" not in raw.columns:
        raise KeyError(
            f"'weighted' column missing in SVD TSVs under {metrics_dir}. "
            "Are these july-branch outputs?"
        )

    tables = {}
    for val in (True, False):
        key = f"weighted_{'true' if val else 'false'}"
        subset = raw[raw["weighted"] == val]
        if subset.empty:
            print(f"  warn: no rows with weighted={val} in {metrics_dir}")
            tables[key] = pd.DataFrame(columns=OUT_COLS)
        else:
            tables[key] = _finalise(best_per_group(subset, SORT_METRICS, GROUP_KEYS))
    tables["weighted_all"] = _finalise(best_per_group(raw, SORT_METRICS, GROUP_KEYS))
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = resolve(load_yaml(args.config, args.overrides))

    svd_root = Path(cfg["svd_root"]) if cfg.get("svd_root") else paths.svd_root(cfg)
    out_root = Path(cfg["out_root"]) if cfg.get("out_root") else paths.undisc_root(cfg)
    models   = cfg["models"]
    subsets  = cfg["subsets"]

    out_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        for subset in subsets:
            metrics_dir = svd_root / model / subset
            if not metrics_dir.exists():
                print(f"skip (missing dir): {metrics_dir}")
                continue
            print(f"{model}/{subset}")
            try:
                tables = build_tables(metrics_dir)
            except (FileNotFoundError, KeyError) as e:
                print(f"  skip ({e})")
                continue

            dst_dir = out_root / model / subset
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name, table in tables.items():
                dst = dst_dir / f"{name}.tsv"
                table.to_csv(dst, sep="\t", index=False, na_rep="")
                print(f"  wrote {dst}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
