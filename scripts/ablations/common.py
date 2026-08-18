"""Shared plumbing for the ablation runners.

Every ablation starts from the same place: the four without-GT cells (both backbones
x {WW-AG, WW-HC, TE-Cap, TE-Mag}), the anchor rows in the frozen-triple selection
tables, and the sweep's exact base score. This module holds that plumbing once, so
each runner contains only the axis it varies.

The scoring helpers mirror `main/sweep.py` bit for bit — any drift is a bug, and the
runners assert their anchor points against the selection tables to prove it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main import config as C                                        # noqa: E402
from main.score import (ENSEMBLE_POSITION, ens_score_steps, fit_svd,  # noqa: E402
                        member_positions, score_steps)
from main.stores import list_rep_files, rep_names                   # noqa: E402
from main.sweep import norm_val, select_config                      # noqa: E402,F401

POOLING = "mean"
# The without-GT ablation coverage: both backbones on these four cells.
CONFIGS_NOGT = ["configs-main/ww.yaml", "configs-main/traceelephant.yaml"]
RESULTS_DIR = REPO / "results-ablations"


def iter_cells(config_paths=CONFIGS_NOGT):
    """Yield (cfg, model, subset) over every cell the ablations cover."""
    for cfg_path in config_paths:
        cfg = C.load_config(REPO / cfg_path)
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                yield cfg, model, subset


def load_selection(cfg) -> pd.DataFrame:
    return pd.read_csv(C.select_dir(cfg) / "selection.tsv", sep="\t")


def anchor_rows(sel: pd.DataFrame, model: str, subset: str) -> tuple[pd.Series, pd.Series]:
    """The svd (base) and backprop (SOAP) selection rows for one cell."""
    cell = sel[(sel["model"] == model) & (sel["subset"] == subset)]
    svd = cell[cell["row"] == "svd"]
    bp = cell[cell["row"] == "backprop"]
    assert len(svd) == 1 and len(bp) == 1, f"incomplete selection for {model}/{subset}"
    return svd.iloc[0], bp.iloc[0]


def cell_paths(cfg, model: str, subset: str):
    """(rep_dir, data_dir, files) for one cell."""
    rep_dir = C.reps_root(cfg) / model / subset
    data_dir = C.data_root(cfg) / subset
    return rep_dir, data_dir, list_rep_files(rep_dir)


def position_load_names(rep_dir, files, position):
    """(members, weight_names) needed to score ``position``.

    ``members`` is the middle third of the FULL position list. It must be computed
    here, from the unrestricted file, and passed through: recomputing it from an
    already-restricted store would take the middle third twice.
    """
    if position == ENSEMBLE_POSITION:
        members = member_positions(rep_names(rep_dir / files[0]))
        return members, members
    return None, [position]


def base_scores(cfg, position, cb, ce, train, split, members=None):
    """The sweep's base score for one split, bit-identical to ``_base_pass``."""
    if position == ENSEMBLE_POSITION:
        fits = {p: fit_svd(train.stores[(POOLING, p)].R, cfg["n_components"])
                for p in members}
        tr = {p: train.stores[(POOLING, p)].R for p in members}
        ev = {p: split.stores[(POOLING, p)].R for p in members}
        return ens_score_steps(cb, ce, members, fits, tr, ev)
    V = fit_svd(train.stores[(POOLING, position)].R, cfg["n_components"])
    return score_steps(split.stores[(POOLING, position)].R, V, cb, ce)


def anchor_filter(df: pd.DataFrame, row: pd.Series, axes) -> pd.DataFrame:
    """Rows of a sweep table matching ``row`` on ``axes`` (string-normalized)."""
    for ax in axes:
        df = df[df[ax].astype(str).map(norm_val) == norm_val(row[ax])]
    return df


def seed_mean(df: pd.DataFrame, seeds, group_cols) -> pd.DataFrame:
    """Mean metric over the frozen triple; a group must appear in EVERY seed."""
    df = df[df["seed"].isin(seeds)]
    metric = [c for c in df.columns if "_acc_" in c]
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        if len(g) != len(seeds):
            continue
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)),
                     **{c: float(g[c].mean()) for c in metric}})
    return pd.DataFrame(rows)


def assert_close(got: float, want: float, what: str, tol: float = 1e-9) -> None:
    assert abs(got - want) < tol, f"{what}: {got:.12f} != {want:.12f}"
