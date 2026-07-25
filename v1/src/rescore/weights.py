"""Attention-mass loading and per-range aggregation.

For each (model, subset), reads the .safetensors files produced by
`extract_attention_qk.py` (or its sibling), then for each layer-range
R = [lo, hi) computes

    m^R_{i,t} = mean over layers l in R of raw_attn[l, i]
    w_{i,t}   = m^R_{i,t} / sum_j m^R_{j,t}      (plain normalization)

producing one `process_weighting`-compatible dict per range:

    weightings[r][traj_stem] = {
        step_idx: {"ctx_indices": LongTensor (n_ctx,),
                   "weights":     FloatTensor (n_ctx,)},
        ...
    }
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def layer_ranges(L: int, n_ranges: int) -> list[tuple[int, int]]:
    """Half-open [lo, hi) ranges partitioning [0, L). Last absorbs any remainder."""
    return [(i * L // n_ranges, (i + 1) * L // n_ranges) for i in range(n_ranges)]


def _normalize(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return m / (m.sum() + eps)


def aggregate_attn(
    weighting_root: Path | str,
    model: str,
    subset: str,
    n_ranges: int = 4,
    device: str = "cpu",
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Load attention safetensors and aggregate per layer-range.

    Returns
    -------
    weightings : list[dict]   (length n_ranges)
    range_bounds : list[(lo, hi)]
    """
    root  = Path(weighting_root) / model / subset
    paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no .safetensors files under {root}")

    weightings: list[dict] = [defaultdict(dict) for _ in range(n_ranges)]
    bounds: list[tuple[int, int]] | None = None

    for path in paths:
        traj_stem = path.stem
        with safe_open(path, framework="pt") as f:
            # Group keys by step_idx
            grouped: dict[int, dict[str, str]] = defaultdict(dict)
            for k in f.keys():
                step_str, name = k.split(".", 1)
                grouped[int(step_str)][name] = k

            for step_idx, name_to_key in grouped.items():
                if "raw_attn" not in name_to_key or "ctx_indices" not in name_to_key:
                    continue
                raw_attn    = f.get_tensor(name_to_key["raw_attn"]).to(device)     # (L, n_ctx)
                ctx_indices = f.get_tensor(name_to_key["ctx_indices"]).to(device)  # (n_ctx,)

                if bounds is None:
                    L = raw_attn.shape[0]
                    bounds = layer_ranges(L, n_ranges)

                for r, (lo, hi) in enumerate(bounds):
                    m = raw_attn[lo:hi].mean(dim=0)        # (n_ctx,)
                    weightings[r][traj_stem][step_idx] = {
                        "ctx_indices": ctx_indices,
                        "weights":     _normalize(m),
                    }

    if bounds is None:
        raise RuntimeError(f"no usable step entries found under {root}")

    return [dict(d) for d in weightings], bounds