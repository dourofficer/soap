import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from attribscope.svd.utils import StoreKeeper
import torch.nn.functional as F
from safetensors import safe_open
import torch


def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower(): return "Orchestrator"
    else: return role

def get_mistake_meta(
    keeper: StoreKeeper,
) -> tuple[list[int | None], list[str | None]]:
    indices, roles = [], []
    for start, end in keeper.traj_ranges:
        entry = next((e for e in keeper.index[start:end] if e.is_mistake), None)
        indices.append(entry.step_idx if entry else None)
        roles.append(
            keeper.traj_meta[entry.traj_idx].get("mistake_agent") if entry else None
        )
    return indices, roles

def compute_metrics(
    scores: np.ndarray,
    keeper: StoreKeeper,
    ks: list[int],
    direction: str,
) -> dict:
    ascending    = (direction == "asc")
    total_trajs  = len(keeper.traj_ranges)
    step_hits    = {k: 0 for k in ks}
    agent_hits   = {k: 0 for k in ks}

    mistake_indices, mistake_roles = get_mistake_meta(keeper)

    for (start, end), mistake_step, mistake_role in zip(
        keeper.traj_ranges, mistake_indices, mistake_roles
    ):
        if mistake_step is None:
            continue

        # Pair each entry with its score, then rank by score
        traj_entries = keeper.index[start:end]
        traj_scores  = scores[start:end]
        step_scores  = [(entry.step_idx, entry.role, score) 
                        for entry, score in zip(traj_entries, traj_scores)]
        step_scores.sort(key=lambda x: x[2], reverse=not ascending)

        ranked_steps  = [step_idx for step_idx, _, _ in step_scores]
        ranked_roles  = [standardize_role(role).lower() for _, role, _ in step_scores]
        mistake_rank  = ranked_steps.index(mistake_step) + 1  # 1-based ranking.

        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role.lower() in ranked_roles[:k]:
                agent_hits[k] += 1

    return {
        **{f"step@{k}_{direction}":  step_hits[k]  / total_trajs for k in ks},
        **{f"agent@{k}_{direction}": agent_hits[k] / total_trajs for k in ks},
    }


def resolve_layer(spec: str | int) -> int:
    """Translate shorthand layer specs to the integer index used by process_weighting."""
    if isinstance(spec, int):   return spec
    if spec == "embed":         return 0
    if spec.startswith("act/"): 
        return int(spec.split("/")[1].strip("_normed")) + 1
    raise ValueError(f"Unknown layer spec: {spec!r}")

def load_weighting(
    model:  str,
    subset: str,
    root:   Path | str = "outputs/weighting",
    device: str        = "cpu",
) -> dict[str, dict]:
    """Load per-trajectory raw similarity scores from disk.

    Returns a dict keyed by trajectory filename:
        weighting[fn] = {
            "metadata": {...},
            "steps":    {step_idx: {"raw_cosine": ..., "raw_dot": ..., "ctx_indices": ...}}
        }
    """
    root = Path(root) / model / subset
    weighting: dict[str, dict] = {}

    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            meta = json.loads(f.metadata()["payload_metadata"])

            steps: dict[int, dict] = defaultdict(dict)
            for key in f.keys():
                step_idx_str, name = key.split(".", 1)
                steps[int(step_idx_str)][name] = f.get_tensor(key).to(device)

            weighting[path.stem] = {
                "metadata": meta,
                "steps":    dict(sorted(steps.items())),
            }

    print(f"Loaded {len(weighting)} trajectories from {root}")
    return weighting


def process_weighting(
    weighting: dict[str, dict],
    layer:     int   = -1,
    temp:      float = 0.1,
    sim:       str   = "raw_cosine",
) -> dict[str, dict[int, dict]]:
    """Convert raw per-layer similarities into 1D softmaxed weights per step.

    For each step, picks one layer of the raw (L+1, n_ctx) score matrix,
    applies softmax with temperature, and pairs the result with its
    `ctx_indices` so it's directly usable by `reweight_scores`.

    Parameters
    ----------
    layer : which layer of the raw matrix to use. -1 = last (default).
    temp  : softmax temperature. Lower → sharper distribution.
    sim   : "raw_cosine" or "raw_dot".
    """
    processed: dict[str, dict[int, dict]] = {}
    for fn, traj_data in weighting.items():
        steps_in = traj_data.get("steps", traj_data)
        out: dict[int, dict] = {}
        for step_idx, entry in steps_in.items():
            raw = entry[sim][layer]                       # (n_ctx,)
            w   = F.softmax(raw.float() / temp, dim=-1)   # (n_ctx,)
            out[step_idx] = {
                "ctx_indices": entry["ctx_indices"],
                "weights":     w,
            }
            # breakpoint()
        processed[fn] = out
    return processed


def reweight_scores(
    scores: torch.Tensor,
    keeper: StoreKeeper,
    weighting: dict,
    gamma: float,
    k: int,
) -> torch.Tensor:
    """Dependency-aware reweighting:

        S~(v_t) = S(v_t) - gamma * sum_{i < t} w_{i,t} * S(v_i)

    Each step's uncertainty is discounted by what its weighted predecessors
    already "explain". Single-pass: the discount reads original S(v_i), not
    the reweighted version — so corrections don't cascade forward through
    the trajectory.

    Parameters
    ----------
    scores    : 1D tensor of raw scores, one per (trajectory, step) entry,
                aligned with ``keeper.index``.
    keeper    : exposes ``.traj_ranges`` (list of (start, end) into `scores`),
                ``.index`` (per-entry records with ``step_idx`` / ``traj_idx``),
                and ``.traj_meta`` (per-trajectory metadata).
    weighting : per-trajectory weighting in either layout:
                    weighting[fn][step_idx]           = {ctx_indices, weights}
                    weighting[fn]["steps"][step_idx]  = {ctx_indices, weights}
                ``weights`` is a 1D, length-n_ctx tensor (already softmaxed
                and layer-collapsed); ``ctx_indices`` lists the step indices
                each weight refers to, in matching order.
    gamma     : correction strength in [0, 1]. gamma=0 returns `scores` unchanged.

    Returns
    -------
    Tensor with the same shape, dtype, and device as `scores`.
    """
    # Clone preserves dtype/device and ensures writes don't mutate the input.
    reweighted = scores.clone()

    # Each trajectory occupies a contiguous block [start, end) of the flat
    # arrays (`scores`, `keeper.index`, …). Processing one trajectory at a
    # time lets us build a local step→position map and keeps cross-trajectory
    # bookkeeping out of the discount computation.
    for start, end in keeper.traj_ranges:
        traj_entries = keeper.index[start:end]
        if not traj_entries:
            raise ValueError("A trajectory must contain some steps.")

        # Resolve the trajectory's filename via an entry's traj_idx — more
        # robust than relying on the enumerate index of `traj_ranges`, in
        # case ordering ever drifts.
        traj_idx = traj_entries[0].traj_idx
        # fn       = keeper.traj_meta[traj_idx]["filename"]
        fn       = str(traj_idx)

        # Fetch the weighting block for this trajectory. Tolerate both the
        # flat layout and the nested {"steps": …} layout in one line.
        traj_w = weighting.get(fn)
        if traj_w is None:
            raise ValueError("Why is this empty?")
        traj_w = traj_w.get("steps", traj_w)

        # step_idx → position in the flat `scores` array. Used both to write
        # the reweighted value back and to look up scores of predecessor
        # steps within this same trajectory.
        pos_of = {e.step_idx: start + k for k, e in enumerate(traj_entries)}

        for step_idx, pos in pos_of.items():
            entry = traj_w.get(step_idx)
            if entry is None: continue
                # No recorded weighting (e.g. earliest scoreable step with no
                # in-graph predecessors). Leaving the score as-is matches the
                # base case S~(v_1) = S(v_1) in the paper.
                

            # Weighted sum of predecessor scores. Any ctx_id not in `pos_of`
            # — typically step 0, the human question that's never scored —
            # is silently dropped; there's no S(v_i) to subtract for it.
            discount = sum(
                w_i * scores[pos_of[int(ctx_id)]]
                for ctx_id, w_i in zip(entry["ctx_indices"][-k:], entry["weights"][-k:])
                if int(ctx_id) in pos_of
            )

            reweighted[pos] = scores[pos] - gamma * discount
            # breakpoint()

    return reweighted

