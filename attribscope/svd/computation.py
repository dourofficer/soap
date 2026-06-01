"""fit SVD + score in one call, return DataFrame.

attribscope.svd.computation
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product as iproduct
from pathlib import Path
from typing import Callable

import json
import pandas as pd
import torch
from tqdm import tqdm
from safetensors import safe_open

from attribscope.svd.core import _run_svd, projection_svd, ranged_projection_svd
from attribscope.svd.utils import (
    load_representations,
    RepresentationStore,
    RepresentationStores,
    StoreKeeper,
)
# Sweep helpers — assumed to live alongside the other utils. Move/import
# from wherever they actually reside in your project.

# ── Tunables ──────────────────────────────────────────────────────────────────

SCORING_FNS: dict[str, Callable] = {
    # "proj": projection_svd,
    "proj": ranged_projection_svd
}

DEFAULT_POOLING = {
    "hidden": ["mean", "last"],
    "grads":  ["grad"],
}


def fit_one(store: RepresentationStore, n_components: int) -> dict:
    """Fit raw + centered SVD for a single RepresentationStore.

    Returns
    -------
    {
      "V_raw":      Tensor(d, n_components),   # CPU
      "V_centered": Tensor(d, n_components),   # CPU
      "ref":        Tensor(d,),                # per-weight mean, CPU
    }
    """
    G    = store.R.float()
    mean = G.mean(dim=0)
    return {
        "V_raw":      _run_svd(G,        n_components).cpu(),
        "V_centered": _run_svd(G - mean, n_components).cpu(),
        "ref":        mean.cpu(),
    }


def fit_all(stores: dict, n_components: int) -> dict:
    """Fit raw + centered SVD for every store.

    Returns
    -------
    {pooling: {weight_name: {"V_raw", "V_centered", "ref"}}}
    """
    out: dict[str, dict] = defaultdict(dict)
    for s in tqdm(
        sorted(stores.values(), key=lambda s: (s.pooling, s.name)),
        desc="SVD fit",
    ):
        out[s.pooling][s.name] = fit_one(s, n_components)
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_one(
    store:              RepresentationStore,
    svd_entry:          dict,
    n_components:       int,
    scoring_fns:        dict[str, Callable],
    device:             torch.device = torch.device("cuda"),
) -> list[dict]:
    """Compute anomaly scores for one store across all scoring configurations.

    Iterates over the Cartesian product of:
      - centered vs. raw representations
      - scoring functions (e.g. projection-based)
      - component range (c_begin, c_end) pairs from [0, n_components]

    Also appends L1 and L2 norm baselines (no SVD involved).

    Parameters
    ----------
    store        : holds the (T, d) representation matrix and metadata.
    svd_entry    : {"V_raw", "V_centered", "ref"} — precomputed SVD tensors.
    n_components : number of SVD components; determines (c_begin, c_end) combos.
    scoring_fns  : mapping from method name to scoring callable.
    device       : device to run computation on.

    Returns
    -------
    List of dicts, one per (centered × method × (c_begin, c_end)) combo plus
    norm baselines. Each dict has keys:
        weight, pooling, method, c_begin, c_end, centered, scores (np.ndarray).
    """
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

    # Representation norm baselines (no SVD components involved)
    for centered in (True, False):
        ref    = svd_entry["ref"].to(device) if centered else None
        R_eval = R - ref if ref is not None else R
        for norm_p, method_name in ((2, "norm_l2"), (1, "norm_l1")):
            norms = R_eval.norm(p=norm_p, dim=1).cpu().numpy()
            rows.append({
                "weight":   store.name,
                "pooling":  store.pooling,
                "method":   method_name,
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