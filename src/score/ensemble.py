"""Layer-band ensemble scoring (ens-mid3) — one deterministic pseudo-position.

For each scorer config, average the z-scored per-step scores across the MIDDLE THIRD
of the model's ``act/N`` positions (excluding embed / _normed). Members are oriented
to a common "higher = error" (desc) sense first (native-asc methods negated), then
z-scored with TRAIN-split member statistics, then averaged. Emitted as rows with
``position = "ens-mid3"``, ``direction = "desc"`` so they compete in reduction like any
other position — but denoise the fragile per-layer selection.

The same ``ens_score_vec`` reproduces an ens-mid3 base row for the rescore stage.

    from src.score.ensemble import member_positions, ensemble_rows, ens_score_vec
"""
from __future__ import annotations

import torch

from .svd import score_config, config_grid, N_COMPONENTS
from .scorers import METHOD_DIRECTION
from ..metrics import compute_metrics_batch

ENSEMBLE_BAND = "mid3"
ENSEMBLE_POSITION = "ens-mid3"
_EPS = 1e-8
# norms are direction "both" — skip them in the ensemble (baselines, not headline).
ENSEMBLE_METHODS_OK = lambda m: METHOD_DIRECTION.get(m) in ("asc", "desc")


def member_positions(positions: list[str]) -> list[str]:
    """Middle third of act/N positions (exclude embed, *_normed, ens-*).

    Which layer to score is the noisiest hyperparameter in the whole pipeline: adjacent
    layers give similar-but-not-identical rankings, and on small validation splits the
    argmax over ~10-34 positions is mostly sampling noise. Averaging over a BAND instead
    of selecting one position removes that axis from the search entirely.

    The middle third is chosen on the standard depth argument — early layers are still
    largely lexical, the last layers specialise toward next-token prediction, and the
    mid-stack carries the most task-semantic representation. It is deliberately a fixed
    rule rather than a tuned range, since tuning it would reintroduce the selection noise
    the ensemble exists to remove. ``embed`` and ``*_normed`` are excluded as they are
    not comparable residual-stream points.
    """
    acts = sorted((p for p in positions
                   if p.startswith("act/") and not p.endswith("_normed")),
                  key=lambda s: int(s.split("/")[1]))
    P = len(acts)
    return acts[P // 3: P - P // 3]


def ens_score_vec(method, cb, ce, centered, weighted, members,
                  fit_by_pos, train_R_by_pos, eval_R_by_pos) -> torch.Tensor:
    """z-averaged ensemble score for one config on an eval split (train stats).

    Two things have to happen before member scores can be averaged at all:

      1. **Common orientation.** Members must agree on which end means "error", or the
         average cancels. Native-asc methods (``proj``) are negated so every member is
         "higher = error"; the ensemble's output direction is therefore always ``desc``.
      2. **Common scale.** Different layers produce scores on wildly different scales
         (norms grow with depth), so a raw mean is dominated by whichever layer happens
         to have the largest magnitude. Each member is z-scored first.

    The z-statistics come from the TRAIN split, never from the split being scored: using
    eval statistics would leak the evaluation distribution into the score. This is the
    same discipline as fitting the SVD on train only.
    """
    zs = []
    flip = METHOD_DIRECTION.get(method) == "asc"     # orient to desc
    for pos in members:
        entry = fit_by_pos[pos]
        s_tr = score_config(train_R_by_pos[pos], entry, method, cb, ce, centered, weighted)
        s_ev = score_config(eval_R_by_pos[pos], entry, method, cb, ce, centered, weighted)
        if flip:
            s_tr, s_ev = -s_tr, -s_ev
        mu, sd = s_tr.mean(), s_tr.std(unbiased=False)
        zs.append((s_ev - mu) / (sd + _EPS))
    return torch.stack(zs).mean(dim=0)


def ensemble_rows(pooling, members, fit_by_pos, train_R_by_pos, val_R_by_pos, test_R_by_pos,
                  val_ctx, test_ctx, methods, weighted_cfg, ks,
                  n_components=N_COMPONENTS, centered_cfg=(True, False)) -> list[dict]:
    """Emit ens-mid3 metric rows (direction=desc) for every config, batched over configs."""
    if len(members) < 2:
        return []
    configs = [c for c in config_grid(methods, weighted_cfg, n_components, centered_cfg)
               if ENSEMBLE_METHODS_OK(c[0])]
    val_stack, test_stack = [], []
    for method, cb, ce, centered, weighted in configs:
        val_stack.append(ens_score_vec(method, cb, ce, centered, weighted, members,
                                       fit_by_pos, train_R_by_pos, val_R_by_pos))
        test_stack.append(ens_score_vec(method, cb, ce, centered, weighted, members,
                                        fit_by_pos, train_R_by_pos, test_R_by_pos))
    val_stack, test_stack = torch.stack(val_stack), torch.stack(test_stack)
    vm = compute_metrics_batch(val_stack, None, ks, "desc", ctx=val_ctx)
    tm = compute_metrics_batch(test_stack, None, ks, "desc", ctx=test_ctx)

    rows = []
    for i, (method, cb, ce, centered, weighted) in enumerate(configs):
        for k in ks:
            rows.append({
                "pooling": pooling, "position": ENSEMBLE_POSITION, "method": method,
                "c_begin": cb, "c_end": ce, "centered": centered, "weighted": weighted,
                "direction": "desc", "k": k,
                "step_acc_val": float(vm[f"step@{k}_desc"][i]),
                "agent_acc_val": float(vm[f"agent@{k}_desc"][i]),
                "step_acc_test": float(tm[f"step@{k}_desc"][i]),
                "agent_acc_test": float(tm[f"agent@{k}_desc"][i]),
            })
    return rows
