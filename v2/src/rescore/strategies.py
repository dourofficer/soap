"""Orientation, per-trajectory normalization, and the rescoring strategies.

THE PROBLEM BEING SOLVED
------------------------
A base score rates each step's anomaly in isolation. But a failed trajectory is not a
bag of independent steps: once a decisive error happens, everything downstream inherits
its damage, so late steps look anomalous merely by being descendants of the real cause.
Ranking raw base scores therefore tends to surface a SYMPTOM rather than the cause.

Both strategies here correct that using the same causal signal — the attention mass
``w_{i,t}``, the fraction of step t's query attention that lands in predecessor i —
but they push blame in OPPOSITE directions:

    discount   S~_t = s_t - gamma * sum_{i<t} w_{i,t} s_i        = s - gamma (W s)
    backprop   S~_i = s_i + gamma * (sum_{t>i} w_{i,t} s_t)
                                    / (sum_{t>i} w_{i,t})        = s + gamma (Wᵀs)/(Wᵀ1)

``discount`` SUBTRACTS inherited anomaly from descendants: a step that merely echoes an
already-anomalous predecessor gets marked down. It can only help when the true cause
already scores competitively; it never moves score toward a cause that looks locally
innocuous.

``backprop`` is the transpose: each step COLLECTS the anomaly of the steps that attended
to it. A decisive error whose own representation looks unremarkable still gets promoted
if everything downstream of it is broken — which is closer to what "decisive error"
means. The hub normalisation (dividing by total received attention ``Wᵀ1``) is what
keeps this honest: without it, any heavily-attended step — a plan, a task restatement —
accumulates blame simply for being popular. Dividing by attention received turns the
quantity into "mean anomaly per unit of attention", so a step is promoted only when its
dependents are anomalous *relative to how much they leaned on it*.

Both are SINGLE-PASS: they read the ORIGINAL ``s`` on the right-hand side, never the
partially-updated ``S~``, so corrections do not cascade through the trajectory.

WHY ORIENT AND NORMALIZE FIRST
------------------------------
* ``orient`` exists only because ``proj`` is "lower = error"; the arithmetic above
  assumes "higher = error". Distance scorers are already in that convention, so
  ``allowed_orients`` gives them ``["none"]`` and the axis disappears. Beware ``sigmoid``
  on large-magnitude scores: it saturates to 0 for every step, collapsing the ranking to
  a tie — which is why the undiscounted reference is taken from the BASE metric rather
  than recomputed from an oriented score.
* ``score_norm`` addresses a real defect: the discount is NOT shift-invariant. Adding a
  constant c to every score moves step 1 (which has no predecessors, so no correction)
  by c, but every other step by c(1-gamma). The absolute level of the score therefore
  leaks into step-1-versus-rest comparisons. Centering (or z-scoring) within each
  trajectory removes that arbitrary offset. It is an affine per-trajectory map, so it
  cannot change the BASE ranking — only the discount arithmetic.

``discount_loop`` is the explicit per-step reference implementation; ``discount_vec`` /
``backprop_vec`` are its matrix form and evaluate ALL gammas in one broadcast. The two
are invariant-tested to agree, including on tie-saturated inputs.

    from src.rescore.strategies import orient, normalize_scores, discount_vec
"""
from __future__ import annotations

import torch

from .weights import coerce_w
from ..score.scorers import METHOD_DIRECTION

_EPS = 1e-12


# ── orientation ─────────────────────────────────────────────────────────────
def orient(scores: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "negate":
        return -scores
    if strategy == "inverse":
        return 1.0 / (scores + _EPS)
    if strategy == "sigmoid":
        return torch.sigmoid(-scores)
    if strategy == "none":
        return scores.clone()
    raise ValueError(f"unknown orient strategy: {strategy!r}")


def allowed_orients(method: str, configured: list[str]) -> list[str]:
    """Native-desc methods (distance family) -> ['none']; asc methods -> configured list."""
    return ["none"] if METHOD_DIRECTION.get(method) == "desc" else list(configured)


# ── per-trajectory score normalization ──────────────────────────────────────
def normalize_scores(scores: torch.Tensor, keeper, mode: str) -> torch.Tensor:
    if mode == "none":
        return scores
    out = scores.clone()
    for start, end in keeper.traj_ranges:
        seg = out[start:end]
        mu = seg.mean()
        if mode == "center":
            out[start:end] = seg - mu
        elif mode == "zscore":
            sd = seg.std(unbiased=False)
            out[start:end] = (seg - mu) / (sd + _EPS)
        else:
            raise ValueError(f"unknown score_norm {mode!r}")
    return out


# ── reference discount (explicit per-step form) ─────────────────────────────
def discount_loop(scores, keeper, weighting, gamma, w):
    """Single-pass discount, written out per step. The readable definition that
    ``discount_vec`` must match (invariant-tested). Scores must already be
    'higher = error'."""
    w = coerce_w(w)
    out = scores.clone()
    device = scores.device
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        if not entries:
            continue
        traj_w = weighting.get(str(entries[0].traj_idx))
        if traj_w is None:
            continue
        step_to_global = {e.step_idx: start + i for i, e in enumerate(entries)}
        for offset, e in enumerate(entries):
            ctx = traj_w.get(e.step_idx)
            if ctx is None:
                continue
            ctx_ids, ctx_w = ctx["ctx_indices"], ctx["weights"].to(device)
            n_ctx = ctx_w.shape[0]
            if n_ctx == 0:
                continue
            if w == "all" or (isinstance(w, int) and w >= n_ctx):
                kept_w, kept_ids = ctx_w, ctx_ids
            else:
                vals, idx = torch.topk(ctx_w, int(w))
                kept_w = vals / (vals.sum() + _EPS)
                kept_ids = ctx_ids[idx]
            pred, aligned = [], []
            for j, ci in enumerate(kept_ids.tolist()):
                pos = step_to_global.get(int(ci))
                if pos is not None:
                    pred.append(scores[pos])
                    aligned.append(kept_w[j])
            if not pred:
                continue
            pred = torch.stack(pred)
            aligned = torch.stack(aligned)
            if aligned.numel() != kept_w.numel():
                aligned = aligned / (aligned.sum() + _EPS)
            out[start + offset] = scores[start + offset] - gamma * (aligned * pred).sum()
    return out


# ── vectorized strategies (all gammas at once) ──────────────────────────────
def _discount_column(s, keeper, Wmats) -> torch.Tensor:
    """D = W s assembled over trajectories (per-step discount sum). (N,)."""
    D = torch.zeros_like(s)
    for (start, end), W in zip(keeper.traj_ranges, Wmats):
        D[start:end] = W.to(s) @ s[start:end]
    return D


def _backprop_column(s, keeper, Wmats) -> torch.Tensor:
    """(W^T s)/(W^T 1) assembled over trajectories (per-step hub-normalized blame). (N,)."""
    B = torch.zeros_like(s)
    for (start, end), W in zip(keeper.traj_ranges, Wmats):
        Wt = W.to(s).T
        num = Wt @ s[start:end]
        den = Wt.sum(dim=1)
        B[start:end] = num / (den + _EPS)
    return B


def discount_vec(s, keeper, Wmats, gammas) -> torch.Tensor:
    """S~ for every gamma: (N, G). S~ = s - gamma * (W s)."""
    D = _discount_column(s, keeper, Wmats)
    g = torch.as_tensor(gammas, dtype=s.dtype, device=s.device)
    return s[:, None] - g[None, :] * D[:, None]


def backprop_vec(s, keeper, Wmats, gammas) -> torch.Tensor:
    """S~ for every gamma: (N, G). S~ = s + gamma * (W^T s)/(W^T 1)."""
    B = _backprop_column(s, keeper, Wmats)
    g = torch.as_tensor(gammas, dtype=s.dtype, device=s.device)
    return s[:, None] + g[None, :] * B[:, None]


STRATEGIES = {"discount": discount_vec, "backprop": backprop_vec}
