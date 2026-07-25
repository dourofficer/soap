"""Score reorientation and the discount pass.

The discount pass implements

    tilde{S}(v_t) = S(v_t) - gamma * sum_{i in top-w}  w_{i,t} S(v_i)

single-pass (reads original S, not tilde{S}). Scores must already be in
the "higher = error" convention; callers are responsible for orienting
SVD outputs (lower = error) before calling.

Metrics are computed via `attribscope.svd.utils.compute_metrics` from the
caller side.
"""
from __future__ import annotations

from typing import Union

import torch


_EPS = 1e-12


# ── SVD orientation ──────────────────────────────────────────────────────────

def orient_svd_scores(scores: torch.Tensor, strategy: str) -> torch.Tensor:
    """Flip SVD projection scores (lower = error) to "higher = error" convention."""
    if strategy == "negate":
        return -scores
    if strategy == "inverse":
        return 1.0 / (scores + _EPS)
    if strategy == "sigmoid":
        return torch.sigmoid(-scores)
    raise ValueError(f"unknown orient strategy: {strategy!r}")


# ── Discount pass ────────────────────────────────────────────────────────────

def apply_discount(
    scores: torch.Tensor,
    keeper,
    weighting: dict,
    gamma: float,
    w: Union[int, str],
) -> torch.Tensor:
    """Single-pass discount with optional top-w restriction.

    Parameters
    ----------
    scores    : (N_total,) — flat across all trajectories in keeper order.
    keeper    : exposes .traj_ranges (list[(start, end)]) and .index (list of
                StepIndex). The trajectory's filename-as-key for the weighting
                dict is `str(entries[0].traj_idx)`, matching the safetensors
                stem naming used by aggregate_attn.
    weighting : dict[traj_stem -> {step_idx: {"ctx_indices", "weights"}}].
    gamma     : float in [0, 1].
    w         : int (keep top-w predecessors, renormalize their weights to
                sum to 1) or "all" (use full predecessor set as given).
    """
    out = scores.clone()
    device = scores.device

    for start, end in keeper.traj_ranges:
        entries = list(keeper.index[start:end])
        if not entries:
            continue

        traj_idx = entries[0].traj_idx
        traj_w   = weighting.get(str(traj_idx))
        if traj_w is None:
            continue

        step_to_global = {e.step_idx: start + i for i, e in enumerate(entries)}

        for offset, entry in enumerate(entries):
            t = entry.step_idx
            ctx = traj_w.get(t)
            if ctx is None:
                continue
            ctx_ids = ctx["ctx_indices"]
            ctx_w   = ctx["weights"].to(device)
            n_ctx   = ctx_w.shape[0]
            if n_ctx == 0:
                continue

            # Top-w slicing
            if w == "all" or (isinstance(w, int) and w >= n_ctx):
                kept_w   = ctx_w
                kept_ids = ctx_ids
            else:
                kept_vals, top_idx = torch.topk(ctx_w, int(w))
                kept_w   = kept_vals / (kept_vals.sum() + _EPS)
                kept_ids = ctx_ids[top_idx]

            # Resolve predecessor scores; drop any ctx ids not in this split.
            pred_scores, aligned_w = [], []
            for i, ci in enumerate(kept_ids.tolist()):
                pos = step_to_global.get(int(ci))
                if pos is not None:
                    pred_scores.append(scores[pos])
                    aligned_w.append(kept_w[i])
            if not pred_scores:
                continue
            pred_scores = torch.stack(pred_scores)
            aligned_w   = torch.stack(aligned_w)
            if aligned_w.numel() != kept_w.numel():
                aligned_w = aligned_w / (aligned_w.sum() + _EPS)

            discount = gamma * (aligned_w * pred_scores).sum()
            out[start + offset] = scores[start + offset] - discount

    return out