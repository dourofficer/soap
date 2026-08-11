"""Re-run ONE frozen config and expose the per-step signal.

Every sweep row is a summary: one accuracy standing in for a scoring pass over thousands
of steps. That is enough to pick a winner and useless for understanding it. This module
re-runs a chosen config and hands back, per step, the base score, the final score after
the strategy, the within-trajectory rank, the prediction flag and the gold flag.

FAITHFULNESS. It calls the SAME primitives the sweep calls (``score_steps`` /
``ens_score_steps``, ``strategy_mats``, ``apply_strategy``), so a reproduced row's
metrics must equal the recorded ones — ``run_reproduce`` asserts exactly that. A
mismatch is a bug, not a rounding artifact.

SPLITS. ``val``/``test`` reproduce the sweep's evaluation exactly. ``all`` re-uses the
SAME train-fitted V to score EVERY trajectory in the subset — the "apply the frozen
model" mode, for inspecting trajectories that were never in an eval split.

    python -m main reproduce --config configs-main/ww.yaml --row backprop
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from . import config as C
from .metrics import compute_metrics, get_mistake_meta, standardize_role
from .rescore import aggregate_attn, apply_strategy, strategy_mats
from .score import ENSEMBLE_POSITION, ens_score_steps, fit_svd, member_positions, score_steps
from .stores import list_rep_files, load_representations, split_files
from .sweep import BASE_STRATEGY, POOLING


@dataclass
class Reproduction:
    config:      dict
    keeper:      Any
    base:        torch.Tensor      # S = 1/(pi+eps); already "higher = error"
    final:       torch.Tensor      # after the strategy (== base for the svd row)
    metrics:     dict
    per_step:    pd.DataFrame = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)


class ReproContext:
    """Per (model, subset) caches: the attention aggregation and the loaded splits.

    Constructed from explicit paths rather than a config so other callers (e.g. the
    figure scripts in ``src/analysis``) can point it at any tree.
    """

    def __init__(self, rep_dir, data_dir, attn_root, model: str, subset: str,
                 splits: dict, n_ranges: int = 4, n_components: int = 20,
                 device: str = "cpu"):
        self.rep_dir, self.data_dir, self.attn_root = rep_dir, data_dir, attn_root
        self.model, self.subset = model, subset
        self.splits, self.n_ranges = splits, n_ranges
        self.n_components, self.device = n_components, device
        self.files = list_rep_files(rep_dir)
        self._weightings = None
        self._labels: list[str] = []
        self._bundles: dict[tuple[int, str], dict] = {}

    @classmethod
    def from_config(cls, cfg: dict, model: str, subset: str) -> "ReproContext":
        return cls(rep_dir=C.reps_root(cfg) / model / subset,
                   data_dir=C.data_root(cfg) / subset,
                   attn_root=C.attn_root(cfg), model=model, subset=subset,
                   splits=cfg["splits"], n_ranges=cfg["n_ranges"],
                   n_components=cfg["n_components"], device=cfg.get("device", "cpu"))

    @property
    def weightings(self):
        if self._weightings is None:
            self._weightings, bounds = aggregate_attn(
                self.attn_root, self.model, self.subset,
                n_ranges=self.n_ranges, device=self.device)
            self._labels = [f"{lo}-{hi}" for lo, hi in bounds]
        return self._weightings

    def range_index(self, label: str) -> int:
        self.weightings                       # force the labels
        return self._labels.index(str(label))

    def bundle(self, seed: int, split: str) -> dict:
        key = (int(seed), split)
        if key not in self._bundles:
            parts = split_files(self.files, self.splits, int(seed))
            load = lambda fl: load_representations(
                self.rep_dir, self.data_dir, poolings=[POOLING], files=fl,
                device=self.device)
            train = load(parts["train"])
            ev = load(self.files if split == "all" else parts[split])
            self._bundles[key] = {"train": train, "eval": ev, "fits": {}}
        return self._bundles[key]

    def fit(self, bundle: dict, position: str) -> torch.Tensor:
        if position not in bundle["fits"]:
            bundle["fits"][position] = fit_svd(
                bundle["train"].stores[(POOLING, position)].R, self.n_components)
        return bundle["fits"][position]


def _base_scores(ctx: ReproContext, bundle: dict, row: dict) -> torch.Tensor:
    position = row["position"]
    cb, ce = int(row["c_begin"]), int(row["c_end"])
    if position == ENSEMBLE_POSITION:
        members = member_positions(bundle["train"].positions())
        fits = {p: ctx.fit(bundle, p) for p in members}
        tr = {p: bundle["train"].stores[(POOLING, p)].R for p in members}
        ev = {p: bundle["eval"].stores[(POOLING, p)].R for p in members}
        return ens_score_steps(cb, ce, members, fits, tr, ev)
    V = ctx.fit(bundle, position)
    return score_steps(bundle["eval"].stores[(POOLING, position)].R, V, cb, ce)


def _tabulate(ctx, row, split, keeper, base, final):
    """Per-step and per-trajectory frames. Ranking is descending with the earliest step
    winning ties — the same convention the metrics use."""
    m_steps, m_roles = get_mistake_meta(keeper)
    steps, preds = [], []
    for (start, end), mstep, mrole in zip(keeper.traj_ranges, m_steps, m_roles):
        entries = keeper.index[start:end]
        seg = final[start:end]
        n = len(entries)
        # (-score, i) ascending == score descending with earliest-step tie-break.
        order = sorted(range(n), key=lambda i: (-float(seg[i]), i))
        rank = {i: r + 1 for r, i in enumerate(order)}
        best = order[0]
        for i, e in enumerate(entries):
            steps.append({"model": ctx.model, "subset": ctx.subset, "row": row["row"],
                          "seed": row["seed"], "split": split, "traj_idx": e.traj_idx,
                          "n_steps": n, "step_idx": e.step_idx, "role": e.role,
                          "is_mistake": e.is_mistake,
                          "base": float(base[start + i]), "final": float(seg[i]),
                          "rank": rank[i], "is_pred": i == best})
        pred_e = entries[best]
        true_rank = next((rank[i] for i, e in enumerate(entries) if e.step_idx == mstep), None)
        preds.append({"model": ctx.model, "subset": ctx.subset, "row": row["row"],
                      "seed": row["seed"], "split": split, "traj_idx": pred_e.traj_idx,
                      "n_steps": n, "pred_step": pred_e.step_idx,
                      "pred_agent": standardize_role(pred_e.role),
                      "true_step": mstep, "true_agent": mrole,
                      "step_correct": mstep is not None and pred_e.step_idx == mstep,
                      "agent_correct": (mrole is not None
                                        and standardize_role(pred_e.role).lower() == mrole.lower()),
                      "true_step_rank": true_rank})
    return pd.DataFrame(steps), pd.DataFrame(preds)


def reproduce_row(ctx: ReproContext, row: dict, split: str = "test", ks=(1, 3)) -> Reproduction:
    bundle = ctx.bundle(int(row["seed"]), split)
    keeper = bundle["eval"].keeper
    base = _base_scores(ctx, bundle, row)

    strategy = row.get("strategy")
    has_rescore = isinstance(strategy, str) and strategy and strategy != BASE_STRATEGY
    if has_rescore:
        mats = strategy_mats(keeper, ctx.weightings[ctx.range_index(row["layer_range"])],
                             row["w"], device=ctx.device)
        final = apply_strategy(base, keeper, mats, strategy, [float(row["gamma"])])[:, 0]
    else:
        final = base

    metrics = compute_metrics(final, keeper, ks)
    cfg_out = {k: row[k] for k in ("seed", "position", "c_begin", "c_end", "strategy",
                                   "layer_range", "gamma", "w") if k in row}
    cfg_out |= {"model": ctx.model, "subset": ctx.subset, "split": split}
    per_step, preds = _tabulate(ctx, row, split, keeper, base, final)
    return Reproduction(config=cfg_out, keeper=keeper, base=base, final=final,
                        metrics=metrics, per_step=per_step, predictions=preds)


def run_reproduce(cfg: dict, rows="all", split: str = "test") -> None:
    ks = cfg["ks"]
    sel_path = C.select_dir(cfg) / "selection.tsv"
    if not sel_path.exists():
        raise SystemExit(f"no selection table at {sel_path}; run `main select` first")
    sel = pd.read_csv(sel_path, sep="\t")
    if rows != "all":
        want = rows if isinstance(rows, (list, tuple)) else [rows]
        sel = sel[sel["row"].isin(want)]

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            cell = sel[(sel.model == model) & (sel.subset == subset)]
            if cell.empty:
                continue
            ctx = ReproContext.from_config(cfg, model, subset)
            out_dir = C.repro_dir(cfg, model, subset)
            out_dir.mkdir(parents=True, exist_ok=True)
            for _, srow in cell.iterrows():
                seeds = [int(s) for s in str(srow["seeds"]).split(",")]
                accs = []
                for seed in seeds:
                    row = {c: srow[c] for c in srow.index if pd.notna(srow[c])}
                    row["seed"] = seed
                    r = reproduce_row(ctx, row, split=split, ks=ks)
                    accs.append(r.metrics[f"step@{ks[0]}"])
                    stem = f"{srow['row']}_seed-{seed}_{split}"
                    r.per_step.to_csv(out_dir / f"{stem}.steps.tsv", sep="\t", index=False)
                    r.predictions.to_csv(out_dir / f"{stem}.preds.tsv", sep="\t", index=False)
                    (out_dir / f"{stem}.json").write_text(json.dumps(
                        {"config": r.config, "metrics": r.metrics,
                         "n_trajectories": len(r.keeper.traj_ranges),
                         "n_steps": len(r.per_step)}, indent=2, default=str))
                got = sum(accs) / len(accs)
                if split == "test":
                    want = float(srow["step_acc_test"])
                    assert abs(got - want) < 1e-9, (
                        f"{model}/{subset} row={srow['row']}: reproduced {got:.12f} "
                        f"!= recorded {want:.12f}")
                    print(f"  [ok] {model}/{subset} {srow['row']}: {got:.4f} (verified)")
                else:
                    print(f"  [--] {model}/{subset} {srow['row']} split={split}: {got:.4f}")
