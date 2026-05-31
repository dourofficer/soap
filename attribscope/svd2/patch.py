from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from typing import Callable
import numpy as np
import random

from safetensors import safe_open

from attribscope.svd2.utils import (
    get_mistake_meta, standardize_role,
    load_representations,
    RepresentationStore,
    RepresentationStores,
    StoreKeeper,
)
from typing import Callable
from collections import defaultdict
from itertools import product as iproduct


def compute_metrics(
    scores: np.ndarray,
    keeper: StoreKeeper,
    ks: list[int],
    direction: str,
) -> dict:
    ascending    = (direction == "asc")
    total_trajs  = len(keeper.traj_ranges)
    step_hits    = {k: 0 for k in ks}
    agent_hits   = {k: 0 for k in ks}
    mistake_indices, mistake_roles = get_mistake_meta(keeper)

    for (start, end), mistake_step, mistake_role in zip(
        keeper.traj_ranges, mistake_indices, mistake_roles
    ):
        if mistake_step is None:
            continue

        # Pair each entry with its score, then rank by score
        traj_entries = keeper.index[start:end]
        traj_scores  = scores[start:end]
        step_scores  = [(entry.step_idx, entry.role, score) 
                        for entry, score in zip(traj_entries, traj_scores)]
        step_scores.sort(key=lambda x: x[2], reverse=not ascending)

        ranked_steps  = [step_idx for step_idx, _, _ in step_scores]
        ranked_roles  = [standardize_role(role).lower() for _, role, _ in step_scores]
        mistake_rank  = ranked_steps.index(mistake_step) + 1  # 1-based ranking.

        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role.lower() in ranked_roles[:k]:
                agent_hits[k] += 1

    return {
        **{f"step@{k}_{direction}":  step_hits[k]  / total_trajs for k in ks},
        **{f"agent@{k}_{direction}": agent_hits[k] / total_trajs for k in ks},
    }


def gather_configs_and_metrics(
    score_records: list[dict],   # output of score_all
    keeper:        StoreKeeper,
    ks:            list[int],
) -> pd.DataFrame:
    """Evaluate score records against ground-truth mistake steps.

    Expands each score record across (direction × k), calling compute_metrics
    once per (record, direction) since it computes all ks in a single pass.

    `score_records` are the results from running `score_all`
    Each row in score_records has the format:
    {
        "weight":   store.name,
        "pooling":  store.pooling,
        "method":   method,
        "c_begin":  c_begin,
        "c_end":    c_end,
        "centered": centered,
        "scores":   scores,      # (T,) np.ndarray
    })

    Returns
    -------
    Flat DataFrame, one row per (weight × method × c × centered × direction × k).
    """
    rows = []
    for rec in score_records:
        for direction in ("asc", "desc"):
            m = compute_metrics(rec["scores"], keeper, ks, direction)
            rows.extend([{
                "weight":    rec["weight"],
                "pooling":   rec["pooling"],
                "method":    rec["method"],
                "c_begin":   rec["c_begin"],
                "c_end":     rec["c_end"],
                "centered":  rec["centered"],
                "direction": direction,
                "k":         k,
                "step_acc":  m[f"step@{k}_{direction}"],
                "agent_acc": m[f"agent@{k}_{direction}"],
            } for k in ks])
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------
# SVD Patch.
# In the current implementation, we select the top-K FIRST principal components.
# However, the first few components may only bakes the information about the 
# agent style, prose, etc. To remove that bias, now we select K components from 
# a begin index to and end index (like 3:10).
# ------------------------------------------------------------------------------
def truncated_projection_svd(
    R: torch.Tensor, # (T, d) matrix of row reps to score
    V: torch.Tensor, # (d, c) matrix of top-c right singular vectors from G
    c_begin: int = 0,
    c_end:   int = 20,
    ref: torch.Tensor | None = None, # (d,) mean gradient for centering, if desired
) -> torch.Tensor:
    assert 0 <= c_begin < c_end <= V.shape[1], \
        f"Invalid range [{c_begin}:{c_end}] for V with {V.shape[1]} components"
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    scores = (R_f @ V[:, c_begin:c_end]).square().mean(dim=1).to(R.dtype)
    return scores


def score_one(
    store:              RepresentationStore,
    svd_entry:          dict,
    n_components:       int,
    scoring_fns:        dict[str, Callable],
    device:             torch.device = torch.device("cuda"),
) -> list[dict]:
    R    = store.R.float().to(device)
    rows = []

    # Projection-based scores over (centered × method × (c_begin, c_end))
    get_bounds = lambda N: [(a, b) for a in range(N) for b in range(a + 1, N + 1)]
    combos = iproduct((True, False), scoring_fns.items(), get_bounds(n_components))
    for centered, (method, fn), (c_begin, c_end) in combos:
        V      = svd_entry["V_centered" if centered else "V_raw"].to(device)
        ref    = svd_entry["ref"].to(device) if centered else None
        scores = fn(R, V, c_begin, c_end, ref).cpu().numpy()
        rows.append({
            "weight":   store.name,
            "pooling":  store.pooling,
            "method":   method,
            "c_begin":  c_begin,
            "c_end":    c_end,
            "centered": centered,
            "scores":   scores,
        })

    # Representation norm baseline (no SVD components involved)
    for centered in (True, False):
        ref    = svd_entry["ref"].to(device) if centered else None
        R_eval = R - ref if ref is not None else R
        norms  = R_eval.norm(dim=1).cpu().numpy()
        rows.append({
            "weight":   store.name,
            "pooling":  store.pooling,
            "method":   "norm",
            "c_begin":  0,
            "c_end":    0,
            "centered": centered,
            "scores":   norms,
        })

    return rows

def score_all(
    stores:             dict,
    svd:                dict,
    n_components:       int,
    scoring_fns:        dict[str, Callable],
    device:             torch.device,
) -> list[dict]:
    """Compute anomaly scores for every store.

    Returns
    -------
    Flat list of score records (see score_one).
    """
    records = []
    for s in stores.values():
        if s.pooling not in svd or s.name not in svd[s.pooling]:
            continue
        records.extend(score_one(
            store        = s,
            svd_entry    = svd[s.pooling][s.name],
            n_components = n_components,
            scoring_fns  = scoring_fns,
            device       = device,
        ))
    return records