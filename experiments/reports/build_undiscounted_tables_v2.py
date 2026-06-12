"""Build undiscounted result tables — one set of TSVs per (model, subset).

Per (model, subset), three output files under out_root/model/subset/:
  - weighted_true.tsv   best configs with weighted=True  fixed
  - weighted_false.tsv  best configs with weighted=False fixed
  - weighted_all.tsv    best configs treating weighted as a hyperparameter

Each file has 6 rows (2 poolings x 3 seeds), strategy=svd only.
Best = argmax (step_acc_test, agent_acc_test) lexicographically, per
(pooling, seed). Rows within a file are sorted (pooling [last,mean], seed).

Usage:
    python -m experiments.reports.build_undiscounted_tables_v2
    python -m experiments.reports.build_undiscounted_tables_v2 \
        --config experiments/reports/configs/undiscounted_v2.yaml \
        --set svd_root="outputs-1006/weighted-projections/226" \
        --set out_root="outputs-1006/undiscounted-splits/226"

    # inline overrides:
    python -m experiments.reports.build_undiscounted_tables_v2 \
        --config ... svd_root=outputs/projections out_root=outputs/tables/v2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Defaults (overridable via config / CLI) ──────────────────────────────────
SVD_ROOT_DEFAULT = Path("outputs/projections")
OUT_ROOT_DEFAULT = Path("outputs/tables/undiscounted-v2")

MODELS_DEFAULT  = ["deepseek-8b", "llama-3.1-8b", "qwen3-8b", "qwen3-14b"]
SUBSETS_DEFAULT = ["algorithm-generated", "hand-crafted"]

SEED_RE      = re.compile(r"seed-(\d+)\.tsv$")
POOLING_ORDER = ["last", "mean"]

OUT_COLS = [
    "strategy", "position", "pooling", "method", "c_begin", "c_end",
    "centered", "weighted", "threshold", "seed",
    "step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test",
]


# ── I/O helpers ──────────────────────────────────────────────────────────────

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


# ── Core logic ───────────────────────────────────────────────────────────────

def _best_per_pooling_seed(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["step_acc_test", "agent_acc_test"],
                       ascending=False, kind="mergesort")
          .groupby(["pooling", "seed"], as_index=False, sort=False)
          .first()
    )


def _sort_section(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pooling"] = pd.Categorical(df["pooling"],
                                   categories=POOLING_ORDER, ordered=True)
    out = (df.sort_values(["pooling", "seed"], kind="mergesort")
             .reset_index(drop=True))
    out["pooling"] = out["pooling"].astype(str)
    return out


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(strategy="svd", threshold=pd.NA)
    # Ensure weighted column is present (may be missing in older TSVs)

    if "weighted" not in df.columns:
        df["weighted"] = pd.NA
    # Keep only columns that exist
    cols = [c for c in OUT_COLS if c in df.columns]
    return _sort_section(df[cols])


def build_tables(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """Return dict with keys 'weighted_true', 'weighted_false', 'weighted_all'."""
    raw = _load_concat(metrics_dir, "svd")

    if "weighted" not in raw.columns:
        raise KeyError(
            f"'weighted' column missing in SVD TSVs under {metrics_dir}. "
            "Are these july-branch outputs?"
        )

    tables = {}

    # weighted=True / weighted=False: filter then pick best
    for val in (True, False):
        key = f"weighted_{'true' if val else 'false'}"
        subset = raw[raw["weighted"] == val]
        if subset.empty:
            print(f"  warn: no rows with weighted={val} in {metrics_dir}")
            tables[key] = pd.DataFrame(columns=OUT_COLS)
        else:
            tables[key] = _finalise(_best_per_pooling_seed(subset))

    # weighted as hyperparameter: pick best across both
    tables["weighted_all"] = _finalise(_best_per_pooling_seed(raw))

    return tables


# ── Config / CLI ─────────────────────────────────────────────────────────────
def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config file (all keys optional)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    svd_root = Path(cfg.get("svd_root", SVD_ROOT_DEFAULT))
    out_root = Path(cfg.get("out_root", OUT_ROOT_DEFAULT))
    models   = cfg.get("models",  MODELS_DEFAULT)
    subsets  = cfg.get("subsets", SUBSETS_DEFAULT)

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