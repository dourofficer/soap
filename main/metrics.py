"""step@k / agent@k, ranked WITHIN each trajectory.

Ranking is always DESCENDING here: the base score folds the inverse orientation in, so
"higher = error" holds at every stage and there is no direction axis to carry.

Four quirks are preserved on purpose — they define the numbers, and "fixing" any of them
silently changes every result:

  * ``agent@k`` lowercases the GOLD role but ``standardize_role()``s the candidate roles.
  * Trajectories with no gold mistake are SKIPPED but still counted in the divisor.
  * Ties resolve to the EARLIEST step.
  * Hits accumulate as int64 and are divided in float64 at the very end (dividing in
    float32 introduces ~6e-8 error and breaks exact agreement with the loop).

    from main.metrics import compute_metrics, compute_metrics_batch, KeeperContext
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

KS = (1, 3)


def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower():
        return "Orchestrator"
    return role


def get_mistake_meta(keeper) -> tuple[list, list]:
    """Per trajectory (keeper order): (mistake_step or None, mistake_agent or None)."""
    indices, roles = [], []
    for start, end in keeper.traj_ranges:
        entry = next((e for e in keeper.index[start:end] if e.is_mistake), None)
        indices.append(entry.step_idx if entry else None)
        roles.append(keeper.traj_meta[entry.traj_idx].get("mistake_agent") if entry else None)
    return indices, roles


def _to_list(scores) -> list[float]:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().tolist()
    if isinstance(scores, np.ndarray):
        return scores.tolist()
    return list(scores)


def compute_metrics(scores, keeper, ks: Sequence[int] = KS) -> dict:
    """Reference loop. Returns ``{"step@k": float, "agent@k": float}``."""
    scores = _to_list(scores)
    total = len(keeper.traj_ranges)
    step_hits = {k: 0 for k in ks}
    agent_hits = {k: 0 for k in ks}
    mistake_indices, mistake_roles = get_mistake_meta(keeper)

    for (start, end), mistake_step, mistake_role in zip(
            keeper.traj_ranges, mistake_indices, mistake_roles):
        if mistake_step is None:
            continue
        entries = keeper.index[start:end]
        step_scores = [(e.step_idx, e.role, s) for e, s in zip(entries, scores[start:end])]
        step_scores.sort(key=lambda x: x[2], reverse=True)     # stable -> earliest tie wins
        ranked_steps = [s for s, _, _ in step_scores]
        ranked_roles = [standardize_role(r).lower() for _, r, _ in step_scores]
        mistake_rank = ranked_steps.index(mistake_step) + 1
        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role.lower() in ranked_roles[:k]:
                agent_hits[k] += 1

    return {**{f"step@{k}": step_hits[k] / total for k in ks},
            **{f"agent@{k}": agent_hits[k] / total for k in ks}}


class KeeperContext:
    """Precomputed, score-independent ranking context for a keeper (built once).

    ``total`` counts ALL trajectories — mistake-less ones inflate the divisor, matching
    ``compute_metrics``.
    """

    def __init__(self, keeper):
        self.total = len(keeper.traj_ranges)
        self.trajs: list[dict] = []
        m_idx, m_role = get_mistake_meta(keeper)
        for (start, end), mstep, mrole in zip(keeper.traj_ranges, m_idx, m_role):
            if mstep is None:
                continue
            entries = keeper.index[start:end]
            steps = [e.step_idx for e in entries]
            role_match = [standardize_role(e.role).lower() == mrole.lower() for e in entries]
            self.trajs.append({
                "start": start, "end": end, "m": steps.index(mstep),
                "role_match": torch.tensor(role_match, dtype=torch.bool),
            })


def compute_metrics_batch(scores, keeper, ks: Sequence[int] = KS,
                          ctx: KeeperContext | None = None) -> dict:
    """Vectorized step@k / agent@k for MANY score vectors at once.

    ``scores`` is (C, N) — C independent score vectors over the same N steps in keeper
    order — or (N,). Returns ``{"step@k": (C,) ndarray, ...}``.

    WHY RANK-COUNTS INSTEAD OF SORTING
    The reference loop ranks by a stable descending sort, so tied steps keep step order
    and the EARLIEST tied step wins. ``torch.topk`` is NOT tie-stable, so sorting would
    silently disagree on exactly the ties that near-degenerate scores produce. Instead
    each step's rank is computed in closed form:

        rank(i) = 1 + #{j : s_j > s_i} + #{j < i : s_j == s_i}

    The first count places i below everything strictly better; the second breaks ties by
    original position, charging i only for tied steps that come EARLIER. This is not an
    approximation of the stable sort — it is an identity with it, and a bijection onto
    1..T, so ``rank <= k`` selects exactly the sorted top-k.
    """
    if ctx is None:
        ctx = KeeperContext(keeper)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    C = scores.shape[0]
    dev = scores.device
    step_hits = {k: torch.zeros(C, device=dev, dtype=torch.long) for k in ks}
    agent_hits = {k: torch.zeros(C, device=dev, dtype=torch.long) for k in ks}

    for t in ctx.trajs:
        A = scores[:, t["start"]:t["end"]]                # (C, T)
        T = A.shape[1]
        m = t["m"]
        rm = t["role_match"].to(dev)                      # (T,)
        sm = A[:, m:m + 1]
        rank_m = 1 + (A > sm).sum(1) + (A[:, :m] == sm).sum(1)      # (C,)
        Ai, Aip = A.unsqueeze(2), A.unsqueeze(1)          # (C,T,1), (C,1,T)
        gt = (Aip > Ai).sum(2)                            # (C,T): #better than i
        tri = torch.tril(torch.ones(T, T, device=dev, dtype=torch.long), diagonal=-1)
        eq_earlier = ((Aip == Ai).long() * tri).sum(2)    # (C,T)
        rank_i = 1 + gt + eq_earlier                      # a permutation of 1..T
        for k in ks:
            step_hits[k] += (rank_m <= k).long()
            agent_hits[k] += ((rank_i <= k) & rm.unsqueeze(0)).any(dim=1).long()

    return {**{f"step@{k}": step_hits[k].cpu().numpy() / ctx.total for k in ks},
            **{f"agent@{k}": agent_hits[k].cpu().numpy() / ctx.total for k in ks}}
