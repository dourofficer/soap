"""Orientation, per-trajectory normalization, and the rescoring strategies.

THE PROBLEM BEING SOLVED
------------------------
A base score rates each step's anomaly in isolation. But a failed trajectory is not a
bag of independent steps: once a decisive error happens, everything downstream inherits
its damage, so late steps look anomalous merely by being descendants of the real cause.
Ranking raw base scores therefore tends to surface a SYMPTOM rather than the cause.

All strategies here are the same correction — each step COLLECTS the anomaly of the
steps that attended to it, through the attention mass ``w_{i,t}`` (the fraction of step
t's query attention landing in predecessor i):

    S~_i = s_i + gamma * (sum_t M[t,i] s_t) / (sum_t M[t,i])     = s + gamma (Mᵀs)/(Mᵀ1)

A decisive error whose own representation looks unremarkable still gets promoted if
everything downstream of it is broken — which is what "decisive error" means. The hub
normalisation (dividing by total received attention ``Mᵀ1``) keeps this honest: without
it, any heavily-attended step — a plan, a task restatement — accumulates blame simply
for being popular. The correction is SINGLE-PASS (the right-hand side reads the
ORIGINAL ``s``), so corrections do not cascade through the trajectory.

The three strategies differ only in which matrix M they aggregate through, i.e. WHERE
the top-w sparsification lives (matrices come from ``weights.strategy_mats``):

- ``backprop``    (SOAP) — predecessor-side: each step t keeps its w strongest
  predecessors (rows of W trimmed and renormalized by ``build_W``). Selective: a step
  is lifted only if some successor ranked it among its top-w dependencies; steps that
  win no slot pass through untouched. This selectivity is empirically the strongest.
- ``succ-strong`` — successor-side: W stays full; each step i collects from its w
  strongest successors (columns masked by weight). Lifts nearly every step at any w.
- ``succ-near``   — successor-side: each step i collects from its w NEAREST scored
  successors (columns masked by position); attention still weights the average.

At w="all" the three coincide exactly (M = full W). gamma=0 recovers the base scorer.

WHY ORIENT AND NORMALIZE FIRST
------------------------------
* ``orient`` exists only because ``proj`` is "lower = error"; the arithmetic above
  assumes "higher = error". Distance scorers are already in that convention, so
  ``allowed_orients`` gives them ``["none"]`` and the axis disappears. Beware
  ``sigmoid`` on large-magnitude scores: it saturates to 0 for every step, collapsing
  the ranking to a tie — which is why the undiscounted reference is taken from the
  BASE metric rather than recomputed from an oriented score.
* ``score_norm`` removes the arbitrary per-trajectory offset of the score level. It is
  an affine per-trajectory map, so it cannot change the BASE ranking — only the
  correction arithmetic.

``backprop_succ_loop`` is the explicit per-step reference implementation for the succ
variants (reading the ragged attention dicts directly, independent of build_W and the
masks); the vectorized path is invariant-tested against it, and ``backprop`` against a
by-hand transpose. Vectorized fns evaluate ALL gammas in one broadcast.

    from src.rescore.strategies import orient, normalize_scores, STRATEGIES
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


# ── the shared vectorized correction ─────────────────────────────────────────
def _backprop_column(s, keeper, Wmats) -> torch.Tensor:
    """(M^T s)/(M^T 1) assembled over trajectories (per-step hub-normalized blame). (N,)."""
    B = torch.zeros_like(s)
    for (start, end), W in zip(keeper.traj_ranges, Wmats):
        Wt = W.to(s).T
        num = Wt @ s[start:end]
        den = Wt.sum(dim=1)
        B[start:end] = num / (den + _EPS)
    return B


def backprop_vec(s, keeper, Wmats, gammas) -> torch.Tensor:
    """S~ for every gamma: (N, G). S~ = s + gamma * (M^T s)/(M^T 1)."""
    B = _backprop_column(s, keeper, Wmats)
    g = torch.as_tensor(gammas, dtype=s.dtype, device=s.device)
    return s[:, None] + g[None, :] * B[:, None]


# ── strategy registry ────────────────────────────────────────────────────────
# Every entry shares backprop_vec; the strategy name only picks WHICH matrices out of
# a ``weights.strategy_mats`` dict the correction aggregates through.
def _entry(name: str):
    def fn(s, keeper, mats: dict, gammas):
        return backprop_vec(s, keeper, mats[name], gammas)
    fn.__name__ = f"vec_{name.replace('-', '_')}"
    return fn


STRATEGIES = {name: _entry(name) for name in ("backprop", "succ-strong", "succ-near")}


# ── reference implementation for the succ variants ──────────────────────────
def backprop_succ_loop(scores, keeper, weighting, gamma, w, variant):
    """Successor-side backprop written out per step, reading the ragged attention
    dicts directly (independent of build_W / the masks) — the readable definition the
    vectorized path must match. ``variant`` is 'strongest' or 'nearest'. Scores must
    already be 'higher = error'.

    Row logic mirrors build_W's w="all" path: every predecessor kept, out-of-split /
    unscored context buckets dropped, survivors renormalized ONLY IF something was
    actually dropped."""
    if variant not in ("strongest", "nearest"):
        raise ValueError(f"unknown variant {variant!r}")
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
        step_to_global = {e.step_idx: start + k for k, e in enumerate(entries)}
        # incoming[i] = [(global successor position t, w_{i,t}), ...] in step order
        incoming: dict[int, list] = {start + k: [] for k in range(len(entries))}
        for t_off, e in enumerate(entries):
            ctx = traj_w.get(e.step_idx)
            if ctx is None:
                continue
            ctx_ids, ctx_w = ctx["ctx_indices"], ctx["weights"].to(device)
            if ctx_w.shape[0] == 0:
                continue
            pred, aligned = [], []
            for j, ci in enumerate(ctx_ids.tolist()):
                pos = step_to_global.get(int(ci))
                if pos is not None:
                    pred.append(pos)
                    aligned.append(ctx_w[j])
            if not pred:
                continue
            aligned = torch.stack(aligned)
            if aligned.numel() != ctx_w.numel():
                aligned = aligned / (aligned.sum() + _EPS)
            for pos, wt in zip(pred, aligned):
                incoming[pos].append((start + t_off, wt))
        for gi, succs in incoming.items():
            if not succs:
                continue                              # true sink: score passes through
            if w != "all" and len(succs) > w:
                if variant == "strongest":
                    succs = sorted(succs, key=lambda p: float(p[1]), reverse=True)[:w]
                else:                                 # nearest: smallest successor position
                    succs = sorted(succs, key=lambda p: p[0])[:w]
            wts = torch.stack([wt for _, wt in succs])
            sts = torch.stack([scores[t] for t, _ in succs])
            out[gi] = scores[gi] + gamma * (wts * sts).sum() / (wts.sum() + _EPS)
    return out
