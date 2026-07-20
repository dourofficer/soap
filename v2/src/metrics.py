"""Per-trajectory ranking metrics.

step@k / agent@k rank steps WITHIN a trajectory (never across trajectories).
``compute_metrics`` is the faithful reference loop; ``compute_metrics_batch``
(added in the vectorization milestone) computes all gammas at once and must agree
with it exactly (invariant-tested).

Quirks preserved on purpose (do NOT "fix" — parity):
  * agent@k lowercases the gold role but standardize_role()s the candidate roles.
  * trajectories with no mistake meta are skipped but still counted in the divisor.
  * ties resolve to the EARLIEST step (stable descending sort; entries are in step order).

    from src.metrics import compute_metrics
    m = compute_metrics(scores, keeper, ks=[1, 3], direction="desc")
"""
from __future__ import annotations

import numpy as np
import torch


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


def compute_metrics(scores, keeper, ks: list[int], direction: str) -> dict:
    """step@k / agent@k for each k. ``direction`` in {asc, desc} sets the ranking order."""
    assert direction in ("asc", "desc"), f"bad direction {direction!r}"
    ascending = direction == "asc"
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
        traj_scores = scores[start:end]
        step_scores = [(e.step_idx, e.role, s) for e, s in zip(entries, traj_scores)]
        step_scores.sort(key=lambda x: x[2], reverse=not ascending)
        ranked_steps = [s for s, _, _ in step_scores]
        ranked_roles = [standardize_role(r).lower() for _, r, _ in step_scores]
        mistake_rank = ranked_steps.index(mistake_step) + 1
        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role.lower() in ranked_roles[:k]:
                agent_hits[k] += 1

    return {
        **{f"step@{k}_{direction}": step_hits[k] / total for k in ks},
        **{f"agent@{k}_{direction}": agent_hits[k] / total for k in ks},
    }


# ── vectorized metrics (all configs / gammas at once) ───────────────────────
class KeeperContext:
    """Precomputed, score-independent ranking context for a keeper (built once).

    Holds, per trajectory that HAS a gold mistake: row slice, local mistake index,
    and a boolean role-match vector (standardize_role(role).lower() == gold.lower()).
    ``total`` counts ALL trajectories (missing-mistake ones inflate the divisor,
    matching compute_metrics).
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
            m_local = steps.index(mstep)
            role_match = [standardize_role(e.role).lower() == mrole.lower() for e in entries]
            self.trajs.append({
                "start": start, "end": end, "m": m_local,
                "role_match": torch.tensor(role_match, dtype=torch.bool),
            })


def compute_metrics_batch(scores, keeper, ks, direction, ctx: KeeperContext | None = None):
    """Vectorized step@k / agent@k for MANY score vectors at once.

    ``scores`` is (C, N) — C independent score vectors (configs, or gammas) over the
    same N steps in keeper order — or (N,) for a single vector. Returns
    ``{f"step@{k}_{direction}": (C,) ndarray, ...}``.

    WHY RANK-COUNTS INSTEAD OF SORTING
    ----------------------------------
    The reference loop ranks by a stable descending sort, so tied steps keep their
    original (step-index) order and the EARLIEST tied step wins. Reproducing that with
    tensor ops is the whole difficulty: ``torch.topk`` is NOT tie-stable (verified — it
    can return tied indices in any order), so sorting would silently disagree with the
    loop on exactly the ties that near-degenerate scores produce.

    Instead we compute each step's rank in closed form:

        rank(i) = 1 + #{j : s_j > s_i} + #{j < i : s_j == s_i}

    The first count places i below everything strictly better; the second breaks ties by
    original position, charging i only for tied steps that come EARLIER. This is not an
    approximation of the stable sort — it is an identity with it, and it is a bijection
    onto 1..T (two tied steps get different ranks because their earlier-tie counts
    differ). So ``rank <= k`` selects exactly the sorted top-k, and ``rank(mistake)``
    is exactly the loop's ``mistake_rank``. Invariant-tested against the loop on
    tie-saturated inputs.

    ``asc`` is handled by negating: ranking ``-s`` descending == ranking ``s`` ascending,
    and equality (hence the tie-break) is unaffected by the sign flip.

    Hits accumulate as int64 and are divided in float64 at the very end — dividing in
    float32 introduces ~6e-8 error and breaks exact agreement with the loop.
    """
    if ctx is None:
        ctx = KeeperContext(keeper)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    C = scores.shape[0]
    dev = scores.device
    s = scores if direction == "desc" else -scores      # rank by higher-is-better
    # Integer hit-counts (divide in float64 at the end for exact parity with the loop).
    step_hits = {k: torch.zeros(C, device=dev, dtype=torch.long) for k in ks}
    agent_hits = {k: torch.zeros(C, device=dev, dtype=torch.long) for k in ks}

    for t in ctx.trajs:
        A = s[:, t["start"]:t["end"]]                    # (C, T)
        T = A.shape[1]
        m = t["m"]
        rm = t["role_match"].to(dev)                     # (T,)
        # mistake-step rank
        sm = A[:, m:m + 1]
        rank_m = 1 + (A > sm).sum(1) + (A[:, :m] == sm).sum(1)   # (C,)
        # all-row ranks (for agent@k top-k set)
        Ai, Aip = A.unsqueeze(2), A.unsqueeze(1)         # (C,T,1),(C,1,T)
        gt = (Aip > Ai).sum(2)                           # (C,T): #better than i
        tri = torch.tril(torch.ones(T, T, device=dev, dtype=torch.long), diagonal=-1)
        eq_earlier = ((Aip == Ai).long() * tri).sum(2)   # (C,T)
        rank_i = 1 + gt + eq_earlier                     # (C,T), a permutation of 1..T
        for k in ks:
            step_hits[k] += (rank_m <= k).long()
            topk = (rank_i <= k) & rm.unsqueeze(0)       # (C,T)
            agent_hits[k] += topk.any(dim=1).long()

    out = {}
    for k in ks:
        out[f"step@{k}_{direction}"] = step_hits[k].cpu().numpy() / ctx.total
        out[f"agent@{k}_{direction}"] = agent_hits[k].cpu().numpy() / ctx.total
    return out
