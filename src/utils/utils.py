"""
attribscope.svd.utils
--------------------------------------------
Utility functions and data structures for 
loading, organizing, and saving representations 
and scores in the SVD-based evaluation pipeline.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from typing import Callable
import numpy as np
import random

from safetensors import safe_open


# ─────────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepIndex:
    """Row-level metadata for one entry in a stacked representation matrix."""
    row:        int     # row index in R
    traj_idx:   int     # 1-based index into the loaded data list
    step_idx:   int     # step index within the trajectory
    role:       str     # e.g. "WebSurfer", "Orchestrator (thought)"
    is_mistake: bool    # whether this is the gold mistake step


@dataclass
class RepresentationStore:
    """A single (T, d) matrix for one (name, pooling) pair."""
    R:       torch.Tensor   # (T, d) stacked representations, one row per step
    name:    str            # weight name or layer name, e.g. "v/35"
    pooling: str            # "last" / "mean" for activations; "grad" / "l1_norm" / ... for gradients


@dataclass
class StoreKeeper:
    """Shared metadata across all stores in a RepresentationStores container.

    Every RepresentationStore in a given container indexes the same set of
    (traj_idx, step_idx) pairs in the same order, so book-keeping lives here,
    not per-store.
    """
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   dict[int, dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device


@dataclass
class RepresentationStores:
    """Container for multiple RepresentationStore, keyed by "{pooling}.{name}"."""
    stores: dict[str, RepresentationStore]
    keeper: StoreKeeper

# ─────────────────────────────────────────────────────────────────────────────
# Safetensors key helpers
#   Keys are formatted as "{step_idx}.{pooling}.{weight_name}".
#   Weight-name shorthand uses "/" (e.g. "v/35"), so splitting on "." with
#   maxsplit=2 cleanly separates the three components.
# ─────────────────────────────────────────────────────────────────────────────
def _parse_key(key: str) -> tuple[int, str, str]:
    step_str, pooling, weight_name = key.split(".", 2)
    return int(step_str), pooling, weight_name


def get_all_rep_names(fp: Path) -> list[str]:
    """Inspect the keys of a safetensors file to find all weight names."""
    with safe_open(fp, framework="pt") as f:
        names = set()
        for k in f.keys():
            step_idx, pooling, name = _parse_key(k)
            names.add(name)
    return sorted(names)

# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────
def load_singular_vectors(
    base_dir: Path,
    model: str,
    subset: str,
    pooling: str,
    n_components_fit: int,
    centered: bool,
):
    """Loads the V matrices from a fitted SVD."""
    centered_str =  f"{'centered' if centered else 'raw'}"
    svd_dir = base_dir / f"{model}/svd/{subset}"
    V_file = f"{pooling}_c{n_components_fit}_{centered_str}/V.safetensors"
    fp = svd_dir / V_file
    assert fp.exists(), f"Missing SVD file: {fp}"
    with safe_open(fp, framework="pt") as f:
        data = {k: f.get_tensor(k) for k in f.keys()}
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Modular loader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_tag(model: str, loss: str, temperature: float) -> str:
    if   loss == "ntp":        return f"{model}"
    elif loss == "kl_temp":    return f"{model}-kl/temp_{temperature}"
    elif loss == "kl_uniform": return f"{model}-kl/uniform"
    else: raise ValueError(f"Unsupported loss: {loss}")

def _resolve_dir(
    root_dir:     Path, # e,g., outputs/ or /data/username/attrib/
    model:        str, # llama-3.1-8b | qwen3-8b
    subset:       str, # algorithm-generated | hand-crafted
    rep_type:     str, # grads | hidden
    loss:         str, # ntp | kl_uniform | kl_temp
    temperature:  float | None, # None | None | 1.x
    dir_type:     str, # representations | metrics
):
    """
    Directory structure containing represntations | metrics from {reps_root}
    .
    ├── grads
    │   ├── llama-3.1-8b/
    │   ├── llama-3.1-8b-kl/
    │   │   ├── temp_1.x/
    │   │   └── uniform/
    │   ├── qwen3-8b/
    │   └── qwen3-8b-kl/
    │       ├── temp_1.x/
    │       └── uniform/
    └── hidden
        ├── llama-3.1-8b/
        └── qwen3-8b/
    
    Depends on dir_type, the resolved dir will be `reps` or `metrics`
    """
    assert dir_type in ["representations", "metrics"], \
        f"Unsupported directory type for resolving: {dir_type}"
    assert rep_type in ["grads", "hidden"], f"Unsupported rep_type: {rep_type}"

    if   rep_type == "grads":  model_tag = _resolve_model_tag(model, loss, temperature)
    elif rep_type == "hidden": model_tag = model

    if   dir_type == "representations": dir_tag = "reps"
    elif dir_type == "metrics":         dir_tag = "metrics"

    result = root_dir / rep_type  / model_tag / dir_tag / subset
    result.mkdir(parents=True, exist_ok=True)
    return result

def _resolve_files(rep_dir: Path, files: list[Path] | None) -> list[Path]:
    """Return the sorted list of .safetensors files to load."""
    input_dir = rep_dir
    if files is None:
        resolved = sorted(input_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
        assert resolved, f"No .safetensors files in {input_dir}"
    else:
        resolved = [input_dir / f for f in files]
        assert all(f.exists() for f in resolved), "Invalid file list: one or more paths missing."
        assert all(f.suffix == ".safetensors" for f in resolved), \
            "Only .safetensors files are accepted."
    return resolved


def _resolve_weight_names(weight_names: list[str] | str, ref_file: Path) -> list[str]:
    """Expand 'all' to the full weight-name list inferred from ref_file."""
    if weight_names == "all":
        weight_names = get_all_rep_names(ref_file)
    assert isinstance(weight_names, list) and all(isinstance(w, str) for w in weight_names), \
        "weight_names must be a list of strings or 'all'."
    return weight_names


def _load_trajectory_file(
    tensor_file:  Path, # e.g., rep/to/1.safetensors
    traj_file:    Path, # e.g., traj/to/1.json
    pooling:      str,
    weight_names: list[str],
    row_offset:   int,
) -> tuple[dict[str, list[torch.Tensor]], list[StepIndex], dict[int, dict], int]:
    """
    Load one .safetensors trajectory file and its JSON sidecar.

    Returns
    -------
    tensors     : weight_name -> list of per-step tensors (appended to caller's collections)
    step_indices: list of StepIndex records for each loaded step
    traj_meta   : {traj_idx: metadata_dict}
    n_rows      : number of steps loaded (so the caller can advance its row counter)
    """
    assert tensor_file.stem == traj_file.stem, \
        "Both tensor file and trajectory file should have the same name. " \
        f"Got {tensor_file} and {traj_file}."
    
    traj_idx = int(tensor_file.stem)
    tensors: dict[str, list[torch.Tensor]] = {w: [] for w in weight_names}
    step_indices: list[StepIndex] = []

    # Roles from Who&When JSON sidecar
    history = json.load(open(traj_file))['history']

    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        # Trajectory metadata
        header       = f.metadata()
        metadata     = json.loads(header.get("payload_metadata", "{}"))
        mistake_step = int(metadata.get("mistake_step", -1))

        # Enumerate steps via the first weight as a pivot
        pivot_suffix = f".{pooling}.{weight_names[0]}"
        steps = sorted({
            int(k.split(".", 1)[0]) for k in f.keys() if k.endswith(pivot_suffix)
        })

        for local_row, step_idx in enumerate(steps):
            for w in weight_names:
                tensors[w].append(f.get_tensor(f"{step_idx}.{pooling}.{w}"))
                # breakpoint()
            step_indices.append(StepIndex(
                row        = row_offset + local_row,
                traj_idx   = traj_idx,
                step_idx   = step_idx,
                role       = history[step_idx]["role"],
                is_mistake = (step_idx == mistake_step),
            ))

    return tensors, step_indices, {traj_idx: metadata}, len(steps)


def _build_stores(
    collections:  dict[str, list[torch.Tensor]],
    pooling:      str,
    index:        list[StepIndex],
    lookup:       dict[tuple[int, int], int],
    traj_meta:    dict[int, dict],
    traj_ranges:  list[tuple[int, int]],
    device:       torch.device,
) -> RepresentationStores:
    """Stack accumulated tensors and wrap everything in RepresentationStores."""
    # breakpoint()
    stores = {
        w: RepresentationStore(
            R       = torch.stack(tensors).to(device),
            name    = w,
            pooling = pooling,
        )
        for w, tensors in collections.items()
    }
    keeper = StoreKeeper(
        index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )
    return RepresentationStores(stores=stores, keeper=keeper)


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_representations(
    rep_dir:      Path, # {outputs_root}/grads/{model}/reps/{subset}
                        # with `outputs_root` = /data/username/attrib
                        # only contains .safetensors files
    data_dir:     Path, # accpomanied data dir, `/data/ww/{subset}`
    pooling:      list[str] | str, # grads | hidden
    weight_names: list[str] | str, # default `all``
    device:       torch.device = torch.device("cpu"),
    files:        list[Path] | None = None, 
                        # None to load all, or a list to specify.
                        # list of [1.safetensors, ...]
) -> RepresentationStores:
    """Load per-trajectory safetensors files and build one RepresentationStore
    per (pooling, weight_name) pair, sharing a single StoreKeeper."""

    assert data_dir.parts[-1] == rep_dir.parts[-1], \
        "Representation directory doesn't match subset "
    subset = data_dir.parts[-1]

    files        = _resolve_files(rep_dir, files)
    # breakpoint()
    weight_names = _resolve_weight_names(weight_names, ref_file=files[0])

    collections: dict[str, list[torch.Tensor]] = {w: [] for w in weight_names}
    index:       list[StepIndex]               = []
    lookup:      dict[tuple[int, int], int]    = {}
    traj_meta:   dict[int, dict]               = {}
    traj_ranges: list[tuple[int, int]]         = []
    row = 0

    for tensor_file in tqdm(files, desc=f"Loading representations"):
        traj_file = data_dir / tensor_file.with_suffix(".json").name
        tensors, step_indices, meta, n_rows = _load_trajectory_file(
            tensor_file, traj_file, 
            pooling, weight_names, row_offset=row,
        )

        for w in weight_names:
            collections[w].extend(tensors[w])

        traj_idx = int(tensor_file.stem)
        traj_ranges.append((row, row + n_rows))
        traj_meta.update(meta)
        index.extend(step_indices)
        lookup.update({(si.traj_idx, si.step_idx): si.row for si in step_indices})
        row += n_rows

    return _build_stores(
        collections, pooling, index, lookup, 
        traj_meta, traj_ranges, device
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric builders
# ─────────────────────────────────────────────────────────────────────────────

def split_data(data: list, ratio: float, seed: int):
    data = data.copy()
    random.seed(seed)
    random.shuffle(data)
    i = int(len(data) * ratio)
    return data[:i], data[i:]

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
    assert direction in ["asc", "desc"], \
        "Scores should be ranked either in the asc(ending) order, or " \
        f"the desc(ending) order. The direction `{direction}` is not supported."
    
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


def gather_configs_and_metrics(
    score_records: list[dict],   # output of score_all
    keeper:        StoreKeeper,
    ks:            list[int],
) -> pd.DataFrame:
    """Evaluate score records against ground-truth mistake steps.

    Expands each score record across (direction × k), calling compute_metrics
    once per (record, direction) since it computes all ks in a single pass.

    `score_records` are the results from running `score_all`
    Each row in score_records has the format:
    {
        "weight":   store.name,
        "pooling":  store.pooling,
        "method":   method,
        "c_begin":  c_begin,
        "c_end":    c_end,
        "centered": centered,
        "scores":   scores,      # (T,) np.ndarray
    })

    Returns
    -------
    Flat DataFrame, one row per (weight × method × c × centered × direction × k).
    """
    rows = []
    for rec in score_records:
        for direction in ("asc", "desc"):
            m = compute_metrics(rec["scores"], keeper, ks, direction)
            rows.extend([{
                "weight":    rec["weight"],
                "pooling":   rec["pooling"],
                "method":    rec["method"],
                "c_begin":   rec["c_begin"],
                "c_end":     rec["c_end"],
                "centered":  rec["centered"],
                "direction": direction,
                "k":         k,
                "step_acc":  m[f"step@{k}_{direction}"],
                "agent_acc": m[f"agent@{k}_{direction}"],
            } for k in ks])
    return pd.DataFrame(rows)