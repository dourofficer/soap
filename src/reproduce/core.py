"""Reproduce one frozen config's per-step scores, predictions and metrics.

WHY THIS EXISTS
---------------
Every sweep row on disk is a *summary*: one accuracy number standing in for a full
scoring pass over thousands of steps. That is enough to pick a winner and useless for
understanding it. This module re-runs a single chosen config and hands back the
intermediate signal at every stage of the pipeline:

    base  ->  oriented  ->  normalized  ->  final (after the rescoring strategy)

with per-step rows carrying trajectory id, step index, role, gold flag and rank. From
that you can plot a trajectory's score curve, see whether the discount moved the
argmax and where, and diff two methods step by step on the same trajectory.

FAITHFULNESS
------------
The reproduction path calls the SAME primitives the sweep calls (``score_config`` /
``ens_score_vec`` for the base, ``orient`` / ``normalize_scores`` / ``STRATEGIES`` for
the rescore), so a reproduced row's metrics must equal the sweep row's metrics. The
runner asserts exactly that — a mismatch means the pipeline is not reproducible and is
a bug, not a rounding artifact.

SPLITS
------
``val`` / ``test`` reproduce the sweep's evaluation exactly. ``all`` re-uses the SAME
train-fitted SVD to score EVERY trajectory in the subset — the "apply the frozen model"
mode, for inspecting trajectories that were never in an eval split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from ..common import paths
from ..metrics import compute_metrics, get_mistake_meta, standardize_role
from ..stores import load_representations, split_files, list_rep_files
from ..score.svd import fit_one, score_config, N_COMPONENTS
from ..score.ensemble import member_positions, ens_score_vec, ENSEMBLE_POSITION
from ..score.scorers import native_direction
from ..rescore.weights import aggregate_attn, strategy_mats
from ..rescore.strategies import orient, normalize_scores, STRATEGIES


@dataclass
class Reproduction:
    """One frozen config, re-run. Tensors are in keeper (trajectory-major) order."""
    config:      dict                  # the frozen config, as reproduced
    keeper:      Any
    base:        torch.Tensor          # scorer output, in the method's NATIVE convention
    oriented:    torch.Tensor          # after orient (== base when orient='none')
    normalized:  torch.Tensor          # after per-trajectory score_norm
    final:       torch.Tensor          # after the rescoring strategy (== normalized if none)
    direction:   str                   # ranking direction for `base` ('asc' | 'desc')
    metrics:     dict
    per_step:    pd.DataFrame = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)


class ReproContext:
    """Caches the expensive per-(model, subset) and per-seed work across many rows.

    Reproducing N rows of a reduced table would otherwise reload representations and
    re-aggregate attention N times. Attention aggregation is per (model, subset);
    representation loading + SVD fitting is per seed. Both are cached here so a whole
    table reproduces in roughly the cost of one row.
    """

    def __init__(self, cfg: dict, model: str, subset: str, n_ranges: int = 4):
        self.cfg, self.model, self.subset = cfg, model, subset
        self.device = cfg.get("device", "cuda")
        self.poolings = cfg.get("poolings", ["mean", "last"])
        self.rep_dir = paths.reps_root(cfg) / model / subset
        self.data_dir = paths.data_root(cfg) / subset
        self.files = list_rep_files(self.rep_dir)
        self._weightings = None
        self._range_labels = None
        self._n_ranges = n_ranges
        self._seed_cache: dict = {}

    @property
    def weightings(self):
        """Attention mass per layer-range; loaded once, reused by every row."""
        if self._weightings is None:
            self._weightings, bounds = aggregate_attn(
                paths.attn_root(self.cfg), self.model, self.subset,
                n_ranges=self._n_ranges, device=self.device)
            self._range_labels = [f"{lo}-{hi}" for lo, hi in bounds]
        return self._weightings

    def range_index(self, label: str) -> int:
        _ = self.weightings                       # ensure labels are populated
        if str(label) not in self._range_labels:
            raise ValueError(f"layer_range {label!r} not in {self._range_labels} "
                             f"(n_ranges mismatch?)")
        return self._range_labels.index(str(label))

    def seed_bundle(self, seed: int, split: str) -> dict:
        """Train store (for the SVD fit) + the eval store/keeper for `split`."""
        key = (seed, split)
        if key in self._seed_cache:
            return self._seed_cache[key]
        parts = split_files(self.files, self.cfg["splits"], seed)
        load = lambda fl: load_representations(
            self.rep_dir, self.data_dir, poolings=self.poolings, files=fl, device=self.device)
        train = load(parts["train"])
        # 'all' scores every trajectory with the SAME train-fitted SVD (apply mode).
        eval_files = self.files if split == "all" else parts[split]
        ev = load(eval_files)
        bundle = {"train": train, "eval": ev, "fits": {}}
        self._seed_cache[key] = bundle
        return bundle

    def fit(self, bundle: dict, pooling: str, position: str) -> dict:
        if (pooling, position) not in bundle["fits"]:
            bundle["fits"][(pooling, position)] = fit_one(
                bundle["train"].stores[(pooling, position)].R, N_COMPONENTS)
        return bundle["fits"][(pooling, position)]


# ── base score (single position or the layer ensemble) ──────────────────────
def _base_scores(ctx: ReproContext, bundle: dict, row: dict) -> tuple[torch.Tensor, str]:
    """Per-step base scores + the direction they should be ranked in."""
    pooling, position, method = row["pooling"], row["position"], row["method"]
    cb, ce = int(row["c_begin"]), int(row["c_end"])
    cen, wt = bool(row["centered"]), bool(row["weighted"])
    train, ev = bundle["train"], bundle["eval"]

    if position == ENSEMBLE_POSITION:
        # The ensemble is a derived pseudo-position: rebuild it from its members using
        # the same train-split z-statistics the score stage used. Its output is already
        # oriented "higher = error", hence direction 'desc' regardless of `method`.
        members = member_positions(train.positions())
        fits = {p: ctx.fit(bundle, pooling, p) for p in members}
        s = ens_score_vec(method, cb, ce, cen, wt, members, fits,
                          {p: train.stores[(pooling, p)].R for p in members},
                          {p: ev.stores[(pooling, p)].R for p in members})
        return s, "desc"

    entry = ctx.fit(bundle, pooling, position)
    s = score_config(ev.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
    return s, native_direction(method, row.get("direction"))


# ── the reproduction itself ─────────────────────────────────────────────────
def reproduce_row(ctx: ReproContext, row: dict, split: str = "test",
                  ks: list[int] | None = None) -> Reproduction:
    """Re-run one frozen config; return per-step signal at every pipeline stage.

    ``row`` is a reduced-table row (or any dict with the same keys). Base-only rows
    (from ``base_*.tsv``) carry no rescoring columns and stop after the base score;
    rows from ``crr_*``/``backprop_*`` carry orient / score_norm / strategy / layer_range
    / gamma / w and are taken all the way through.
    """
    ks = ks or ctx.cfg.get("ks", [1])
    seed = int(row["seed"])
    bundle = ctx.seed_bundle(seed, split)
    keeper = bundle["eval"].keeper

    base, direction = _base_scores(ctx, bundle, row)

    # A rescoring strategy is present only on crr_/backprop_ rows.
    strategy = row.get("strategy")
    has_rescore = isinstance(strategy, str) and strategy in STRATEGIES

    if has_rescore:
        # Same order as the sweep: orient -> per-trajectory normalize -> strategy.
        orient_name = row.get("orient", "none")
        snorm = row.get("score_norm", "none")
        oriented = orient(base, orient_name)
        normalized = normalize_scores(oriented, keeper, snorm)
        mats = strategy_mats(keeper, ctx.weightings[ctx.range_index(row["layer_range"])],
                             row["w"], device=normalized.device)
        gamma = float(row["gamma"])
        final = STRATEGIES[strategy](normalized, keeper, mats, [gamma])[:, 0]
        rank_dir = "desc"          # post-orientation scores are always "higher = error"
    else:
        orient_name, snorm, gamma = "none", "none", None
        oriented = normalized = final = base
        rank_dir = direction

    metrics = compute_metrics(final, keeper, ks, rank_dir)
    per_step, predictions = _tabulate(
        ctx, row, split, keeper, base, oriented, normalized, final, rank_dir)

    config = {k: row[k] for k in (
        "seed", "pooling", "position", "method", "c_begin", "c_end", "centered",
        "weighted", "orient", "score_norm", "strategy", "layer_range", "gamma", "w",
    ) if k in row}
    config.update(model=ctx.model, subset=ctx.subset, split=split,
                  rank_direction=rank_dir, base_direction=direction)
    return Reproduction(config=config, keeper=keeper, base=base, oriented=oriented,
                        normalized=normalized, final=final, direction=direction,
                        metrics=metrics, per_step=per_step, predictions=predictions)


def _tabulate(ctx, row, split, keeper, base, oriented, normalized, final, rank_dir):
    """Long per-step frame (plot-ready) + one prediction row per trajectory.

    ``rank`` is 1-based within its trajectory under ``rank_dir``, with ties broken
    toward the EARLIER step — the same convention the metrics use, so ``rank == 1``
    marks exactly the step the metric counts as the prediction.
    """
    b, o, n, f = (x.detach().float().cpu() for x in (base, oriented, normalized, final))
    m_steps, m_roles = get_mistake_meta(keeper)
    rows, preds = [], []
    ascending = rank_dir == "asc"

    for (start, end), true_step, true_role in zip(keeper.traj_ranges, m_steps, m_roles):
        entries = keeper.index[start:end]
        seg = f[start:end]
        order = sorted(range(len(entries)),
                       key=lambda i: (seg[i].item(), -i), reverse=not ascending)
        rank_of = {i: r + 1 for r, i in enumerate(order)}
        best = order[0]
        pred_step = entries[best].step_idx
        pred_role = standardize_role(entries[best].role)

        for i, e in enumerate(entries):
            g = start + i
            rows.append({
                "model": ctx.model, "subset": ctx.subset, "seed": int(row["seed"]),
                "split": split, "traj_idx": e.traj_idx, "n_steps": len(entries),
                "step_idx": e.step_idx, "role": e.role,
                "is_mistake": bool(e.is_mistake),
                "base": b[g].item(), "oriented": o[g].item(),
                "normalized": n[g].item(), "final": f[g].item(),
                "rank": rank_of[i], "is_pred": i == best,
            })
        preds.append({
            "model": ctx.model, "subset": ctx.subset, "seed": int(row["seed"]),
            "split": split, "traj_idx": entries[0].traj_idx, "n_steps": len(entries),
            "pred_step": pred_step, "pred_agent": pred_role,
            "true_step": true_step, "true_agent": true_role,
            "step_correct": true_step is not None and pred_step == true_step,
            "agent_correct": (true_role is not None
                              and pred_role.lower() == standardize_role(true_role).lower()),
            "true_step_rank": (rank_of[[i for i, e in enumerate(entries)
                                        if e.step_idx == true_step][0]]
                               if true_step is not None and
                               any(e.step_idx == true_step for e in entries) else None),
        })
    return pd.DataFrame(rows), pd.DataFrame(preds)
