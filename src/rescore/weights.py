"""Attention-mass loading + dense per-trajectory weight matrices, per strategy.

``aggregate_attn`` reads the attention safetensors
and, for each layer-range R=[lo,hi), builds

    m^R_{i,t} = mean_{l in R} raw_attn[l, i];   w_{i,t} = m^R_{i,t} / sum_j m^R_{j,t}

as dicts ``weighting[traj_stem][step_idx] = {ctx_indices, weights}``.

``build_W`` turns one such dict + a keeper into per-trajectory dense matrices W where
``W[t_local, i_local] = w_{i,t}`` after PREDECESSOR-side top-w slicing and
out-of-split predecessor dropping (see its docstring for the exact order) — the form
``backprop`` consumes. The SUCCESSOR-side strategies instead column-mask the FULL
matrix (``mask_columns_strongest`` / ``mask_columns_nearest``): each step i keeps only
the w successors it collects blame from. ``strategy_mats`` bundles all three forms for
one w; ``WCache`` pre-builds them for a whole sweep (full W built once per range).

    from src.rescore.weights import aggregate_attn, build_W, strategy_mats, WCache
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

_EPS = 1e-12


def layer_ranges(L: int, n_ranges: int) -> list[tuple[int, int]]:
    """Half-open [lo, hi) ranges partitioning [0, L); last absorbs remainder."""
    return [(i * L // n_ranges, (i + 1) * L // n_ranges) for i in range(n_ranges)]


def _normalize(m: torch.Tensor) -> torch.Tensor:
    return m / (m.sum() + _EPS)


def aggregate_attn(attn_root, model, subset, n_ranges=4, device="cpu"):
    """Load attention safetensors and aggregate per layer-range. Returns (weightings, bounds)."""
    root = Path(attn_root) / model / subset
    paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no .safetensors under {root}")

    weightings = [defaultdict(dict) for _ in range(n_ranges)]
    bounds = None
    for path in paths:
        stem = path.stem
        with safe_open(path, framework="pt") as f:
            grouped: dict[int, dict[str, str]] = defaultdict(dict)
            for k in f.keys():
                step_str, name = k.split(".", 1)
                grouped[int(step_str)][name] = k
            for step_idx, name_to_key in grouped.items():
                if "raw_attn" not in name_to_key or "ctx_indices" not in name_to_key:
                    continue
                raw = f.get_tensor(name_to_key["raw_attn"]).to(device)        # (L, n_ctx)
                ctx = f.get_tensor(name_to_key["ctx_indices"]).to(device)     # (n_ctx,)
                if bounds is None:
                    bounds = layer_ranges(raw.shape[0], n_ranges)
                for r, (lo, hi) in enumerate(bounds):
                    weightings[r][stem][step_idx] = {
                        "ctx_indices": ctx,
                        "weights": _normalize(raw[lo:hi].mean(dim=0)),
                    }
    if bounds is None:
        raise RuntimeError(f"no usable step entries under {root}")
    return [dict(d) for d in weightings], bounds


def coerce_w(w):
    """Sweep values for w are ints or the literal 'all' (survives a TSV round-trip)."""
    return "all" if str(w) == "all" else int(w)


def build_W(keeper, weighting: dict, w, device="cpu") -> list[torch.Tensor]:
    """Per-trajectory dense (T,T) weight matrices; row t = predecessor weights of step t.

    This is the bridge from the ragged attention dicts to linear algebra. Once each
    trajectory's dependency structure is a matrix, a rescoring strategy is one matmul
    (``W @ s`` for discount, ``Wᵀ @ s`` for backprop) and every gamma can be evaluated
    by a single broadcast — which is what makes the sweep tractable.

    W is strictly lower-triangular in step order (a step only attends backwards), and
    row t sums to 1 over the predecessors kept for t. Three subtleties, in the order
    they are applied — all three are load-bearing, and getting the order or the
    conditional wrong changes the weights:

      1. **top-w selection**: keep only the w highest-mass predecessors, then RENORMALISE
         those w so the row still sums to 1. Restricting to the few strongest
         dependencies is what makes the correction targeted rather than a diffuse
         recency-weighted average.
      2. **split filtering**: a predecessor may not exist in this split's keeper (the
         eval store holds a subset of trajectories, and steps outside it have no score
         to borrow). Those entries are dropped. Unscored context buckets go the same
         way — the ``human`` question turn of hand-crafted trajectories and, in with-GT
         extractions, the pinned GT block (``ctx_indices`` id ``GT_STEP = -1``). Note
         both CAN claim a top-w slot in step 1 before being dropped here; that is the
         established turn-0 behaviour and is kept identical for the GT bucket.
      3. **conditional renormalisation**: the surviving weights are renormalised ONLY IF
         something was actually dropped. If nothing was dropped, the row keeps the
         normalisation it already has from step 1 (or, for ``w == "all"``, from
         ``aggregate_attn``). Renormalising unconditionally would be a no-op
         mathematically, but the conditional is kept explicit because it documents that a
         dropped predecessor is the ONLY reason a row gets rescaled a second time.

    Rows for steps with no usable predecessors stay all-zero, so their score passes
    through unchanged.
    """
    w = coerce_w(w)
    mats = []
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        T = len(entries)
        W = torch.zeros(T, T, dtype=torch.float32, device=device)
        traj_w = weighting.get(str(entries[0].traj_idx)) if entries else None
        if traj_w is not None:
            step_to_local = {e.step_idx: i for i, e in enumerate(entries)}
            for t_local, e in enumerate(entries):
                ctx = traj_w.get(e.step_idx)
                if ctx is None:
                    continue
                ctx_ids = ctx["ctx_indices"]
                ctx_w = ctx["weights"].to(device)
                n_ctx = ctx_w.shape[0]
                if n_ctx == 0:
                    continue
                if w == "all" or (isinstance(w, int) and w >= n_ctx):
                    kept_w, kept_ids = ctx_w, ctx_ids
                else:
                    vals, idx = torch.topk(ctx_w, int(w))
                    kept_w = vals / (vals.sum() + _EPS)
                    kept_ids = ctx_ids[idx]
                locals_, weights_ = [], []
                for j, ci in enumerate(kept_ids.tolist()):
                    loc = step_to_local.get(int(ci))
                    if loc is not None:
                        locals_.append(loc)
                        weights_.append(kept_w[j])
                if not locals_:
                    continue
                aligned = torch.stack(weights_)
                if aligned.numel() != kept_w.numel():
                    aligned = aligned / (aligned.sum() + _EPS)
                W[t_local, torch.tensor(locals_, device=device)] = aligned.to(device)
        mats.append(W)
    return mats


# ── successor-side column masks ──────────────────────────────────────────────
# Applied to the FULL W (build_W with w="all"): column i of W holds w_{i,t} for every
# successor t that kept i in context, so masking a column selects which successors
# step i collects blame from. Zeros are structurally absent predecessors (truncated
# context, other split, human/GT buckets), so only nonzero entries are candidates.
# No renormalization afterwards: backprop's hub ratio is a weighted mean over the
# kept set, so a common per-column rescale would cancel anyway.

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
    the earliest scored successors that actually have i in context."""
    w = coerce_w(w)
    if w == "all":
        return W.clone()
    M = torch.zeros_like(W)
    for i in range(W.shape[1]):
        nz = W[:, i].nonzero(as_tuple=True)[0]       # ascending row index = step order
        keep = nz[:w]
        M[keep, i] = W[keep, i]
    return M


COLUMN_MASKS = {"succ-strong": mask_columns_strongest, "succ-near": mask_columns_nearest}


def strategy_mats(keeper, weighting: dict, w, device="cpu",
                  full: list[torch.Tensor] | None = None) -> dict[str, list[torch.Tensor]]:
    """The per-strategy matrices for one (weighting, w): what STRATEGIES fns consume.

    ``backprop`` MUST come from ``build_W(w)`` directly — its top-w slot selection
    happens BEFORE unscored buckets (turn-0 human, GT block) are dropped, and that
    slot-consumption order is part of the recorded numbers. Deriving the row trim from
    the already-filtered full W would silently change it. The succ variants, by
    contrast, are defined as column masks of the full W. ``full`` lets a caller pass a
    cached ``build_W(..., "all")`` so it is not rebuilt per w."""
    if full is None:
        full = build_W(keeper, weighting, "all", device)
    pred = [Wj.clone() for Wj in full] if coerce_w(w) == "all" \
        else build_W(keeper, weighting, w, device)
    return {"backprop": pred,
            **{name: [mask(Wj, w) for Wj in full] for name, mask in COLUMN_MASKS.items()}}


class WCache:
    """Per-(model, subset, split) cache of per-strategy W matrices, keyed (range_idx, w).

    Weights are independent of the base-score row / orient / gamma / score_norm, so
    they are built ONCE per (model, subset, seed split) and reused across the whole
    sweep; the full W is built once per range and shared by every w's column masks.
    """

    def __init__(self, weightings: list[dict], keeper, ws, device="cpu"):
        self.keeper = keeper
        self._mats: dict[tuple[int, str], dict[str, list[torch.Tensor]]] = {}
        for r_idx, weighting in enumerate(weightings):
            full = build_W(keeper, weighting, "all", device)
            for w in ws:
                self._mats[(r_idx, str(w))] = strategy_mats(keeper, weighting, w,
                                                            device, full=full)

    def mats(self, r_idx: int, w) -> dict[str, list[torch.Tensor]]:
        return self._mats[(r_idx, str(w))]
