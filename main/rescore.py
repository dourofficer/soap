"""Attention -> dependency weights -> the rescoring correction.

THE PROBLEM. A base score rates each step in isolation, but a failed trajectory is not a
bag of independent steps: once the decisive error happens, everything downstream inherits
its damage, so late steps look anomalous merely by being descendants of the real cause.
Ranking raw base scores therefore surfaces a SYMPTOM rather than the cause.

THE CORRECTION. Every step COLLECTS the anomaly of the steps that attended to it:

    S~(s_i) = S(s_i) + gamma * (sum_t M[t,i] S(s_t)) / (sum_t M[t,i])
            = s + gamma (M^T s) / (M^T 1)

A decisive error whose own representation looks unremarkable is still promoted if
everything downstream of it is broken — which is what "decisive" means. The hub
normalisation (dividing by attention RECEIVED) keeps it honest: without it, any heavily
attended step — a plan, a task restatement — would accumulate blame just for being
popular. The correction is SINGLE-PASS (the right-hand side reads the ORIGINAL ``s``), so
corrections never cascade.

The three strategies are the same arithmetic through a different matrix ``M``, i.e. they
differ only in WHERE the top-w sparsification lives:

  backprop     (SOAP) predecessor-side: each step t keeps its w strongest predecessors
               (rows trimmed and renormalized by build_W). Selective — a step is lifted
               only if some successor ranked it top-w. Empirically strongest.
  succ-strong  successor-side: W stays full; each step i collects from its w strongest
               successors (columns masked by weight).
  succ-near    successor-side: each step i collects from its w NEAREST scored successors.

At ``w="all"`` all three coincide exactly; at ``gamma=0`` all three return ``s``.

    from main.rescore import aggregate_attn, WCache, apply_strategy
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

EPS = 1e-12
STRATEGIES = ("backprop", "succ-strong", "succ-near")


# ── attention -> per-layer-band weights ─────────────────────────────────────
def layer_ranges(L: int, n_ranges: int) -> list[tuple[int, int]]:
    """Half-open [lo, hi) ranges partitioning [0, L); the last absorbs the remainder."""
    return [(i * L // n_ranges, (i + 1) * L // n_ranges) for i in range(n_ranges)]


def _normalize(m: torch.Tensor) -> torch.Tensor:
    return m / (m.sum() + EPS)


def aggregate_attn(attn_root: Path | str, model: str, subset: str,
                   n_ranges: int = 4, device: str = "cpu"):
    """Load attention safetensors and aggregate per layer band.

    For each band R = [lo, hi):  m^R_{i,t} = mean_{l in R} raw_attn[l, i], then
    w_{i,t} = m^R_{i,t} / sum_j m^R_{j,t}.

    Returns ``(weightings, bounds)`` where ``weightings[r][traj_stem][step] =
    {ctx_indices, weights}``. NOTE ``L`` indexes ATTENTION blocks, not the activation
    stage's layer positions — 8 for Qwen3.5 (hybrid, full-attention layers only) vs 32
    for DeepSeek.
    """
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


# ── ragged weights -> dense per-trajectory matrices ─────────────────────────
def build_W(keeper, weighting: dict, w, device: str = "cpu") -> list[torch.Tensor]:
    """Per-trajectory dense (T,T) matrices; row t = predecessor weights of step t.

    Once each trajectory's dependency structure is a matrix, a strategy is one matmul and
    every gamma is a single broadcast. W is strictly lower-triangular in step order, and
    row t sums to 1 over the predecessors kept for t.

    THREE SUBTLETIES, IN THIS ORDER — all load-bearing:

      1. top-w selection: keep the w highest-mass predecessors, then RENORMALISE those w
         so the row still sums to 1. Restricting to the strongest few is what makes the
         correction targeted rather than a diffuse recency-weighted average.
      2. split filtering: a predecessor may not exist in this split's keeper, and some
         context buckets are never scored — the ``human`` question turn of hand-crafted
         trajectories and, in with-GT mode, the pinned block under ``GT_STEP = -1``.
         Those are dropped. Note both CAN claim a top-w slot in step 1 and then vanish
         here; that slot-consumption order is part of the recorded numbers.
      3. conditional renormalisation: survivors are renormalised ONLY IF something was
         dropped. Unconditional renormalisation would be mathematically identical, but
         the conditional documents that a dropped predecessor is the only reason a row is
         rescaled twice.

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
                ctx_ids, ctx_w = ctx["ctx_indices"], ctx["weights"].to(device)
                n_ctx = ctx_w.shape[0]
                if n_ctx == 0:
                    continue
                if w == "all" or (isinstance(w, int) and w >= n_ctx):
                    kept_w, kept_ids = ctx_w, ctx_ids
                else:
                    vals, idx = torch.topk(ctx_w, int(w))
                    kept_w = vals / (vals.sum() + EPS)
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
                    aligned = aligned / (aligned.sum() + EPS)
                W[t_local, torch.tensor(locals_, device=device)] = aligned.to(device)
        mats.append(W)
    return mats


# ── successor-side column masks ─────────────────────────────────────────────
# Applied to the FULL W (build_W with w="all"): column i holds w_{i,t} for every
# successor t that kept i in context, so masking a column selects which successors step
# i collects blame from. Zeros are structurally absent predecessors (truncated context,
# other split, human/GT buckets), so only nonzero entries are candidates. No
# renormalization afterwards: the hub ratio is a weighted mean over the kept set, so a
# common per-column rescale would cancel anyway.

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
        nz = W[:, i].nonzero(as_tuple=True)[0]      # ascending row index = step order
        keep = nz[:w]
        M[keep, i] = W[keep, i]
    return M


COLUMN_MASKS = {"succ-strong": mask_columns_strongest, "succ-near": mask_columns_nearest}


def strategy_mats(keeper, weighting: dict, w, device: str = "cpu",
                  full: list[torch.Tensor] | None = None) -> dict[str, list[torch.Tensor]]:
    """The per-strategy matrices for one (weighting, w).

    ``backprop`` MUST come from ``build_W(w)`` directly — its top-w slot selection happens
    BEFORE unscored buckets are dropped, and that order is part of the recorded numbers.
    Deriving the row trim from the already-filtered full W would silently change it. The
    succ variants, by contrast, are DEFINED as column masks of the filtered full W.
    """
    if full is None:
        full = build_W(keeper, weighting, "all", device)
    pred = ([Wj.clone() for Wj in full] if coerce_w(w) == "all"
            else build_W(keeper, weighting, w, device))
    return {"backprop": pred,
            **{name: [mask(Wj, w) for Wj in full] for name, mask in COLUMN_MASKS.items()}}


class WCache:
    """Per-(model, subset, split) cache of per-strategy W matrices, keyed (range, w).

    Weights depend on none of the base score / gamma, so they are built ONCE per split
    and reused across the whole sweep; the full W is built once per range and shared by
    every w's column masks.
    """

    def __init__(self, weightings: list[dict], keeper, ws, device: str = "cpu"):
        self.keeper = keeper
        self._mats: dict[tuple[int, str], dict[str, list[torch.Tensor]]] = {}
        for r_idx, weighting in enumerate(weightings):
            full = build_W(keeper, weighting, "all", device)
            for w in ws:
                self._mats[(r_idx, str(w))] = strategy_mats(keeper, weighting, w,
                                                            device, full=full)

    def mats(self, r_idx: int, w) -> dict[str, list[torch.Tensor]]:
        return self._mats[(r_idx, str(w))]


# ── the correction ──────────────────────────────────────────────────────────
def apply_strategy(s: torch.Tensor, keeper, mats: dict, strategy: str,
                   gammas) -> torch.Tensor:
    """S~ for EVERY gamma at once: (N, G). ``mats`` is one ``strategy_mats`` dict."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    B = torch.zeros_like(s)
    for (start, end), W in zip(keeper.traj_ranges, mats[strategy]):
        Wt = W.to(s).T
        num = Wt @ s[start:end]
        den = Wt.sum(dim=1)
        B[start:end] = num / (den + EPS)
    g = torch.as_tensor(list(gammas), dtype=s.dtype, device=s.device)
    return s[:, None] + g[None, :] * B[:, None]
