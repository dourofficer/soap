"""Shared table-reduction helpers for the report builders.

Consolidates the seed-regex TSV loading, best-config selection and pooling-order
sorting that were copy-pasted across the undiscounted / discounted builders.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SEED_RE = re.compile(r"seed-(\d+)\.tsv$")
POOLING_ORDER = ["last", "mean"]


# ── seed-tagged TSV loading (undiscounted svd_*.tsv inputs) ──────────────────

def read_with_seed(path: Path) -> pd.DataFrame:
    """Read a ``*_seed-<n>.tsv`` and attach the parsed seed as a column."""
    m = SEED_RE.search(path.name)
    if not m:
        raise ValueError(f"can't parse seed from {path.name}")
    df = pd.read_csv(path, sep="\t")
    df["seed"] = int(m.group(1))
    return df


def load_concat(metrics_dir: Path, prefix: str) -> pd.DataFrame:
    """Concat every ``{prefix}_pooling-*_seed-*.tsv`` under ``metrics_dir``."""
    files = sorted(metrics_dir.glob(f"{prefix}_pooling-*_seed-*.tsv"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.tsv under {metrics_dir}")
    return pd.concat([read_with_seed(p) for p in files], ignore_index=True)


# ── reduction ────────────────────────────────────────────────────────────────

def best_per_group(df: pd.DataFrame,
                   sort_metrics: list[str],
                   group_keys: list[str]) -> pd.DataFrame:
    """Pick the row maximizing ``sort_metrics`` (lexicographic) per group.

    Stable (mergesort) so ties resolve by original order — matches the previous
    per-builder ``_best_per_*`` helpers exactly.
    """
    return (df.sort_values(sort_metrics, ascending=False, kind="mergesort")
              .groupby(group_keys, as_index=False, sort=False)
              .first())


def sort_section(df: pd.DataFrame,
                 order_cols: list[str] = ("pooling", "seed")) -> pd.DataFrame:
    """Sort rows by pooling (last, mean) then seed; return pooling as str."""
    df = df.copy()
    df["pooling"] = pd.Categorical(df["pooling"],
                                   categories=POOLING_ORDER, ordered=True)
    out = (df.sort_values(list(order_cols), kind="mergesort")
             .reset_index(drop=True))
    out["pooling"] = out["pooling"].astype(str)
    return out
