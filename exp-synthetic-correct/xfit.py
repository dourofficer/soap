"""Cross-dataset SVD fit/score — the one genuinely new capability.

`precompute_svd` hardwires fitting and scoring on the same dataset; here the
fit (source train) and the scored store-sets (any target's val/test) are
decoupled. Everything algorithmic is imported from src/.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import torch

import common  # noqa: F401  (sys.path bootstrap)
from src.svd.computation import SCORING_FNS, fit_one, score_all  # noqa: E402
from src.svd.reproduce import N_COMPONENTS, _select_svd_scores  # noqa: E402
from src.utils.utils import (  # noqa: E402
    RepresentationStores,
    gather_configs_and_metrics,
)
from experiments.reports._common import best_per_group  # noqa: E402

MERGE_KEYS = ["position", "pooling", "method", "c_begin", "c_end",
              "centered", "weighted", "direction", "k"]


def fit_source_svd(train_reps: RepresentationStores,
                   n_components: int = N_COMPONENTS,
                   device: str = "cpu") -> dict:
    """fit_all with per-store device control: stores stay where they were
    loaded; each is moved to `device` only for its own SVD, so GPU peak is one
    store, not the whole train set. Returned tensors are CPU (fit_one)."""
    out: dict[str, dict] = defaultdict(dict)
    for s in sorted(train_reps.stores.values(), key=lambda s: (s.pooling, s.name)):
        R_orig = s.R
        s.R = R_orig.to(device)
        out[s.pooling][s.name] = fit_one(s, n_components)
        s.R = R_orig
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dict(out)


def score_target(reps: RepresentationStores, svd_components: dict,
                 n_components: int = N_COMPONENTS,
                 device: str = "cuda") -> list[dict]:
    """score_all against a foreign fit, with a hard key-coverage assert.

    score_all silently `continue`s on any (pooling, layer) key missing from
    the fit — in a cross-dataset run that would drop stores invisibly, so we
    fail loudly instead.
    """
    missing = [
        f"{s.pooling}.{s.name}" for s in reps.stores.values()
        if s.pooling not in svd_components or s.name not in svd_components[s.pooling]
    ]
    assert not missing, (
        "target stores missing from the source SVD fit (would be silently "
        f"skipped by score_all): {missing}"
    )
    return score_all(reps.stores, svd=svd_components,
                     n_components=n_components, scoring_fns=SCORING_FNS,
                     device=device)


def tabulate_val_test(val_records: list[dict], val_keeper,
                      test_records: list[dict], test_keeper) -> pd.DataFrame:
    """Exactly the tail of precompute_svd + the production TSV filtering:
    metrics per config on val and test, merged with _val/_test suffixes,
    direction=='asc' rows only, sorted by step_acc_test desc."""
    val_df = gather_configs_and_metrics(val_records, keeper=val_keeper, ks=[1])
    test_df = gather_configs_and_metrics(test_records, keeper=test_keeper, ks=[1])
    merged = pd.merge(val_df, test_df, suffixes=("_val", "_test"), on=MERGE_KEYS)
    merged = merged[merged["direction"] == "asc"]
    return merged.sort_values("step_acc_test", ascending=False)


def select_configs(table: pd.DataFrame,
                   weighted_flags: list[bool],
                   by: list[str],
                   top_k: int = 1) -> pd.DataFrame:
    """Best config(s) per (pooling, weighted-flag, sel_by convention).

    Mirrors build_undiscounted_tables_v2's weighted_<flag> tables:
    argmax lexicographic on (step_acc_<sel>, agent_acc_<sel>), stable
    mergesort. Seed is fixed within one runner invocation, so grouping is on
    pooling only. Returns the winning rows with a `sel_by` column added.
    """
    picked = []
    for flag in weighted_flags:
        rows = table[table["weighted"] == flag]
        if rows.empty:
            raise ValueError(f"no rows with weighted={flag} to select from")
        for sel in by:
            metrics = [f"step_acc_{sel}", f"agent_acc_{sel}"]
            best = (rows.sort_values(metrics, ascending=False, kind="mergesort")
                        .groupby(["pooling"], as_index=False, sort=False)
                        .head(top_k))
            picked.append(best.assign(sel_by=sel))
    return pd.concat(picked, ignore_index=True)


def reselect_check(table: pd.DataFrame, selected: pd.DataFrame,
                   top_k: int = 1) -> None:
    """Re-derive the winners with best_per_group (the production reducer) and
    assert they match `selected` — guards the selection step. Only valid for
    top_k == 1 (best_per_group takes .first())."""
    if top_k != 1:
        return
    for _, sel_row in selected.iterrows():
        sel = sel_row["sel_by"]
        rows = table[table["weighted"] == sel_row["weighted"]]
        ref = best_per_group(rows, [f"step_acc_{sel}", f"agent_acc_{sel}"],
                             ["pooling"])
        ref_row = ref[ref["pooling"] == sel_row["pooling"]].iloc[0]
        for col in common.CONFIG_KEYS + ["step_acc_val", "step_acc_test",
                                         "agent_acc_val", "agent_acc_test"]:
            assert ref_row[col] == sel_row[col], (
                f"selection mismatch (sel_by={sel}, col={col}): "
                f"{ref_row[col]!r} != {sel_row[col]!r}"
            )


def extract_selected_scores(selected: pd.DataFrame,
                            val_records: list[dict],
                            test_records: list[dict]) -> list[dict]:
    """Per selected config, pull the raw per-step val/test score vectors
    (SVD convention: lower = error, un-oriented) via _select_svd_scores."""
    rows = []
    for _, row in selected.iterrows():
        rows.append({
            **{k: row[k] for k in common.CONFIG_KEYS},
            "sel_by": row["sel_by"],
            "step_acc_val": float(row["step_acc_val"]),
            "agent_acc_val": float(row["agent_acc_val"]),
            "step_acc_test": float(row["step_acc_test"]),
            "agent_acc_test": float(row["agent_acc_test"]),
            "val_scores": _select_svd_scores(val_records, row).cpu(),
            "test_scores": _select_svd_scores(test_records, row).cpu(),
        })
    return rows
