"""Attention-extractor helpers shared by the streaming (live) and eager (legacy,
validation-reference) implementations.
"""
from __future__ import annotations

import torch
from torch import Tensor


def build_key_mask(
    ctx_step_ids: list[int],
    step_tokens:  dict[int, list[int]],
    seq_len:      int,
    device:       torch.device,
) -> Tensor:
    """Build the (n_ctx, N) one-hot mask used to sum-over-T_i in one matmul."""
    n_ctx = len(ctx_step_ids)
    M = torch.zeros(n_ctx, seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M
