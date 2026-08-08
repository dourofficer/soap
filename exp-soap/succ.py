"""Successor-side top-w for SOAP (backprop) — the only new math in exp-soap.

THE CHANGE BEING TESTED
-----------------------
In the main pipeline top-w lives on the PREDECESSOR side: ``build_W`` trims each row t
to t's w strongest predecessors and renormalizes, and backprop merely transposes those
matrices. Selection and aggregation therefore sit on different axes: whether step i
collects from successor t depends on how i ranked inside t's row. A common artifact is
a step that is nobody's top-w predecessor — its column is all zero and backprop
silently degrades to a pass-through (num=0, den~eps -> correction 0).

Here the dependency weights stay FULL (``build_W(..., "all")``: normalized over all of
t's scored predecessors; split filtering and the human/GT-bucket drops come for free)
and top-w is applied per COLUMN — each step i selects which w successors it collects
blame from:

    strongest:  K_i = argtop-w_{t>i} w_{i,t}          (largest dependency weight)
    nearest:    K_i = the w smallest t>i with w_{i,t} > 0

    S~_i = s_i + gamma * ( sum_{t in K_i} w_{i,t} s_t ) / ( sum_{t in K_i} w_{i,t} )

Matrix form: with M = colmask_w(W_full), S~ = s + gamma (M^T s)/(M^T 1) — identical
arithmetic to ``src.rescore.strategies.backprop_vec``, just fed column-masked matrices.
So the strategy function is UNCHANGED; only the cache that builds the matrices differs
(``SuccWCache`` below, a drop-in for ``src.rescore.weights.WCache``).

Properties (invariant-tested in test_succ.py):
* gamma=0 is the identity (inherited from backprop_vec).
* w="all": both variants coincide EXACTLY with the existing backprop at w="all"
  (M = W_full) — the parity anchor against exp-august's frozen sweeps.
* No renormalization after masking: the hub ratio is a weighted mean over K_i, so a
  common per-column rescale cancels. Cross-row normalization differences deliberately
  remain — a successor that concentrated more of its attention on i counts more.
* A step with no attending successors still passes through; that case is a true sink,
  unlike the selection artifact above.
"""
from __future__ import annotations

import torch

from src.rescore.weights import build_W, coerce_w

_EPS = 1e-12

# sweep strategy name -> selection rule
STRATEGY_VARIANTS = {"succ-strong": "strongest", "succ-near": "nearest"}


# ── column masks ─────────────────────────────────────────────────────────────
def mask_columns_strongest(W: torch.Tensor, w) -> torch.Tensor:
    """Keep, in each column i, the w largest nonzero entries (ties: torch.topk order)."""
    w = coerce_w(w)
    if w == "all":
        return W.clone()
    M = torch.zeros_like(W)
    for i in range(W.shape[1]):
        nz = W[:, i].nonzero(as_tuple=True)[0]
        if nz.numel() == 0:
            continue
        _, order = torch.topk(W[nz, i], min(w, nz.numel()))
        keep = nz[order]
        M[keep, i] = W[keep, i]
    return M


def mask_columns_nearest(W: torch.Tensor, w) -> torch.Tensor:
    """Keep, in each column i, the w nonzero entries with the smallest row index —
    the earliest scored successors that actually have i in context (zeros are
    structurally absent: truncated context, other split, human/GT buckets)."""
    w = coerce_w(w)
    if w == "all":
        return W.clone()
    M = torch.zeros_like(W)
    for i in range(W.shape[1]):
        nz = W[:, i].nonzero(as_tuple=True)[0]       # ascending row index = step order
        keep = nz[:w]
        M[keep, i] = W[keep, i]
    return M


MASKS = {"strongest": mask_columns_strongest, "nearest": mask_columns_nearest}


# ── drop-in cache (same interface as src.rescore.weights.WCache) ─────────────
class SuccWCache:
    """Per-(model, subset, split) cache of column-masked FULL W matrices.

    ``build_W(..., "all")`` is built once per layer range; every w derives from it by
    column masking, for BOTH variants at once (masking is cheap; the expensive per-seed
    representation loading and SVD refits in ``run_pair`` are then paid only once for
    the two variants instead of once each). ``run_pair`` consumes this through the
    identical ``.mats(r_idx, w)`` interface — the returned value is a variant->mats
    dict, and only the ``STRATEGIES_SUCC`` closures below ever index it.
    """

    def __init__(self, weightings: list[dict], keeper, ws, device="cpu"):
        self.keeper = keeper
        self._mats: dict[tuple[int, str], dict[str, list[torch.Tensor]]] = {}
        for r_idx, weighting in enumerate(weightings):
            full = build_W(keeper, weighting, "all", device)
            for w in ws:
                self._mats[(r_idx, str(w))] = {
                    variant: [mask(Wj, w) for Wj in full]
                    for variant, mask in MASKS.items()}

    def mats(self, r_idx: int, w) -> dict[str, list[torch.Tensor]]:
        return self._mats[(r_idx, str(w))]


def _succ_strategy(variant: str):
    """A STRATEGIES-compatible fn: unchanged backprop arithmetic on this variant's
    column-masked matrices (picked out of the SuccWCache dict)."""
    from src.rescore.strategies import backprop_vec

    def fn(s, keeper, mats, gammas):
        return backprop_vec(s, keeper, mats[variant], gammas)
    fn.__name__ = f"backprop_{variant}"
    return fn


# strategy name -> vectorized fn, for patching into src.rescore.run.STRATEGIES
STRATEGIES_SUCC = {name: _succ_strategy(variant)
                   for name, variant in STRATEGY_VARIANTS.items()}


# ── reference implementation (explicit per-step form) ────────────────────────
def backprop_succ_loop(scores, keeper, weighting, gamma, w, variant):
    """Successor-side backprop written out per step, reading the ragged attention
    dicts directly (independent of build_W / the masks) — the readable definition the
    vectorized path must match. Scores must already be 'higher = error'.

    Row logic mirrors build_W's w="all" path: every predecessor kept, out-of-split /
    unscored context buckets dropped, survivors renormalized ONLY IF something was
    actually dropped."""
    if variant not in MASKS:
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
