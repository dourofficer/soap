"""SVD fitting + grid scoring.

fit_one factorises the TRAIN step matrix (raw + mean-centered), keeping the top-k
right singular vectors AND the full singular spectrum (used by weighted proj). score_config
computes any single scorer config's per-step scores — it doubles as the "reproduce a
base-table row" primitive used by the rescore stage. score_position runs the full
grid for one (pooling, position) and returns metric rows.

    from src.score.svd import fit_one, score_config, N_COMPONENTS
"""
from __future__ import annotations

import torch

from .scorers import SCORERS, native_directions, weighted_options
from ..metrics import compute_metrics_batch, KeeperContext

N_COMPONENTS = 20


# ── SVD fit ─────────────────────────────────────────────────────────────────
def _run_svd(G: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (V_k = top-k right singular vectors (d,k), S_full = all singular values)."""
    _, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return Vh[:k].T.contiguous(), S.contiguous()


def fit_one(R_train: torch.Tensor, n_components: int = N_COMPONENTS) -> dict:
    """Fit raw + centered SVD on the train matrix; keep top-k V and full spectrum S."""
    G = R_train.float()
    mean = G.mean(dim=0)
    V_raw, S_raw = _run_svd(G, n_components)
    V_cen, S_cen = _run_svd(G - mean, n_components)
    return {
        "V_raw": V_raw, "S_raw": S_raw,           # S_* are FULL spectra
        "V_centered": V_cen, "S_centered": S_cen,
        "ref": mean,
    }


# ── grid helpers ────────────────────────────────────────────────────────────
def band_bounds(n: int) -> list[tuple[int, int]]:
    """All (c_begin, c_end) bands with 0 <= c_begin < c_end <= n."""
    return [(a, b) for a in range(n) for b in range(a + 1, n + 1)]


def score_config(R, svd_entry, method, c_begin, c_end, centered, weighted,
                 device=None) -> torch.Tensor:
    """Per-step scores for ONE scorer config (also the rescore reproducer primitive).

    R is floated FIRST so scorers
    return fp32 — passing fp16 R would round scores to fp16 and flip near-ties.
    """
    V = svd_entry["V_centered" if centered else "V_raw"]
    S = svd_entry["S_centered" if centered else "S_raw"]
    ref = svd_entry["ref"] if centered else None
    R = R.float()
    if device is not None:
        R, V, S = R.to(device), V.to(device), S.to(device)
        ref = ref.to(device) if ref is not None else None
    return SCORERS[method](R, V, c_begin, c_end, ref, singular_values=S, weighted=weighted)


def config_grid(methods, weighted_cfg, n_components=N_COMPONENTS,
                centered_cfg=(True, False)):
    """Yield every (method, c_begin, c_end, centered, weighted) config in the grid.

    ``centered_cfg`` is an axis like any other, driven by the ``centered:`` key of a
    score config. It defaults to BOTH arms so an omitted key reproduces the historical
    grid exactly; the production configs pin it to ``[false]`` (the uncentered fit the
    protocol selects on), which halves the grid without removing the arm from the code.
    """
    for centered in centered_cfg:
        for method in methods:
            combos = [(0, 0)] if method in ("norm_l1", "norm_l2") else band_bounds(n_components)
            for (cb, ce) in combos:
                for weighted in weighted_options(method, weighted_cfg):
                    yield method, cb, ce, centered, weighted


def score_from_entry(pooling, position, svd_entry, val_R, test_R,
                     val_ctx, test_ctx, methods, weighted_cfg, ks,
                     n_components=N_COMPONENTS, device=None,
                     centered_cfg=(True, False)) -> list[dict]:
    """Score the full grid for one (pooling, position) given a fitted svd_entry.

    All config score vectors are stacked and metrics for BOTH directions are computed
    in two batched passes; each config then emits rows only for its native direction(s).
    """
    configs = list(config_grid(methods, weighted_cfg, n_components, centered_cfg))
    val_stack, test_stack = [], []
    for method, cb, ce, centered, weighted in configs:
        val_stack.append(score_config(val_R, svd_entry, method, cb, ce, centered, weighted, device))
        test_stack.append(score_config(test_R, svd_entry, method, cb, ce, centered, weighted, device))
    val_stack = torch.stack(val_stack)      # (C, N_val)
    test_stack = torch.stack(test_stack)    # (C, N_test)

    m = {}                                  # (split, direction) -> metric dict of (C,) arrays
    for direction in ("asc", "desc"):
        m[("val", direction)] = compute_metrics_batch(val_stack, None, ks, direction, ctx=val_ctx)
        m[("test", direction)] = compute_metrics_batch(test_stack, None, ks, direction, ctx=test_ctx)

    rows: list[dict] = []
    for i, (method, cb, ce, centered, weighted) in enumerate(configs):
        base = {"pooling": pooling, "position": position, "method": method,
                "c_begin": cb, "c_end": ce, "centered": centered, "weighted": weighted}
        for direction in native_directions(method):
            for k in ks:
                rows.append({
                    **base, "direction": direction, "k": k,
                    "step_acc_val":  float(m[("val", direction)][f"step@{k}_{direction}"][i]),
                    "agent_acc_val": float(m[("val", direction)][f"agent@{k}_{direction}"][i]),
                    "step_acc_test":  float(m[("test", direction)][f"step@{k}_{direction}"][i]),
                    "agent_acc_test": float(m[("test", direction)][f"agent@{k}_{direction}"][i]),
                })
    return rows
