"""Full CRR reproducer — apply a frozen (optimal) config end-to-end.

Given ONE row that carries both the SVD base-score config (position, pooling,
method, c_begin, c_end, centered, weighted, seed) and the CRR discount config
(svd_orient, layer_range, gamma, w), re-run the whole scoring pipeline:

    base SVD scores  →  orient ("higher = error")  →  attention-weighted discount

Two modes, selected by ``split``:
  - "val" / "test": score the frozen split and return metrics (validate that they
    match the reduced discounted table).
  - "all": fit the scorer on the row's train split, then score EVERY trajectory in
    the subset and return per-trajectory predictions (apply / inference).

Composes src.svd.reproduce (base scores) + src.rescore.weights (attention) +
src.rescore.discount (the discount pass).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.svd.reproduce import (
    reproduce_svd, _load_reps_and_svd, _select_svd_scores, N_COMPONENTS,
)
from src.svd.computation import score_all, SCORING_FNS
from src.rescore.weights import aggregate_attn
from src.rescore.discount import orient_svd_scores, apply_discount
from src.utils.utils import (
    load_representations, compute_metrics, standardize_role, get_mistake_meta,
)


@dataclass
class CRRResult:
    s_tilde:     torch.Tensor           # (N,) final discounted scores, keeper order
    keeper:      Any                    # StoreKeeper aligned with s_tilde
    predictions: list[dict]             # one per trajectory (argmax decisive step)
    metrics:     dict                   # step@k / agent@k for this split


def _coerce_w(w):
    """Sweep values for w are ints or the literal 'all' (survives a TSV round-trip)."""
    return "all" if str(w) == "all" else int(w)


def _base_scores_for_split(row, model, subset, reps_root, data_root, device,
                           splits: dict, split: str):
    """Return (base_scores, keeper) for the requested split.

    val/test reuse the reproducer's cached train-fit + split scores. "all" reuses
    the SAME train-fitted SVD components to score every trajectory in the subset.
    """
    reps_root, data_root = Path(reps_root), Path(data_root)
    if split in ("val", "test"):
        bundle = reproduce_svd(row, model, subset, reps_root, data_root, device,
                               train_split=splits["train"],
                               val_split=splits["val"],
                               test_split=splits["test"])
        if split == "val":
            return bundle.val_scores, bundle.val_keeper
        return bundle.test_scores, bundle.test_keeper

    if split == "all":
        cached = _load_reps_and_svd(
            model, subset, row["pooling"], int(row["seed"]),
            Path(reps_root), Path(data_root), device,
            splits["train"], splits["val"], splits["test"],
        )
        svd_components = cached["svd"]["svd_components"]  # fitted on train

        rep_dir  = Path(reps_root) / model / subset
        data_dir = Path(data_root) / subset
        files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
        files = [f.name for f in files]
        all_reps = load_representations(
            rep_dir=rep_dir, data_dir=data_dir, pooling=row["pooling"],
            weight_names="all", device=device, files=files,
        )
        all_scores = score_all(all_reps.stores, svd=svd_components,
                               n_components=N_COMPONENTS, scoring_fns=SCORING_FNS,
                               device=device)
        return _select_svd_scores(all_scores, row).cpu(), all_reps.keeper

    raise ValueError(f"unknown split {split!r}; expected val | test | all")


def _predictions(s_tilde: torch.Tensor, keeper) -> list[dict]:
    """Per-trajectory argmax (higher = error) → predicted decisive step + agent."""
    mistake_steps, mistake_roles = get_mistake_meta(keeper)
    preds = []
    for (start, end), true_step, true_role in zip(
        keeper.traj_ranges, mistake_steps, mistake_roles
    ):
        entries = keeper.index[start:end]
        seg = s_tilde[start:end]
        best = int(torch.as_tensor(seg).argmax())
        pred_step = entries[best].step_idx
        pred_role = standardize_role(entries[best].role)
        norm_true_role = standardize_role(true_role) if true_role else None
        preds.append({
            "traj_idx":       entries[0].traj_idx,
            "pred_step":      pred_step,
            "pred_agent":     pred_role,
            "true_step":      true_step,
            "true_agent":     true_role,
            "step_correct":   (true_step is not None and pred_step == true_step),
            "agent_correct":  (norm_true_role is not None
                               and pred_role.lower() == norm_true_role.lower()),
        })
    return preds


def reproduce_crr(
    row,
    model: str, subset: str,
    reps_root, data_root, attn_root,
    device: str = "cuda",
    *,
    splits: dict,
    split: str = "test",
    n_ranges: int = 4,
) -> CRRResult:
    """Re-run SVD→orient→discount for one frozen config; see module docstring."""
    base_scores, keeper = _base_scores_for_split(
        row, model, subset, reps_root, data_root, device, splits, split,
    )

    oriented = orient_svd_scores(base_scores, strategy=row["svd_orient"]).cpu()

    weightings, bounds = aggregate_attn(
        Path(attn_root), model, subset, n_ranges=n_ranges, device="cpu",
    )
    labels = [f"{lo}-{hi}" for (lo, hi) in bounds]
    target = str(row["layer_range"])
    if target not in labels:
        raise ValueError(
            f"layer_range {target!r} not in {labels} (n_ranges={n_ranges} mismatch?)")
    weighting = weightings[labels.index(target)]

    s_tilde = apply_discount(
        oriented, keeper, weighting,
        gamma=float(row["gamma"]), w=_coerce_w(row["w"]),
    )

    metrics = compute_metrics(s_tilde, keeper, ks=[1], direction="desc")
    return CRRResult(
        s_tilde=s_tilde, keeper=keeper,
        predictions=_predictions(s_tilde, keeper), metrics=metrics,
    )
