"""Representation loading + the seed -> partition mapping.

``split_data`` / ``derive_split_ratios`` / ``split_files`` are carried over LINE FOR
LINE. With the seeds frozen in the config, this mapping IS the experiment's identity:
every cached score and every reported number is tied to the partition these produce.
Changing the cut order or the reseeding would silently make the same seed mean a
different split while still "working". ``tests/test_main.py`` pins them against a
golden literal partition for exactly that reason.

Stores stay keyed ``(pooling, name)`` even though the sweep only ever asks for ``mean``:
the extractor writes both poolings, so keeping the key means switching back is a config
change rather than a re-extraction.

    from main.stores import load_representations, split_files
    reps = load_representations(rep_dir, data_dir, poolings=["mean"])
    R = reps.stores[("mean", "act/15")].R          # (T, d) fp16
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


# ── data structures ─────────────────────────────────────────────────────────
@dataclass
class StepIndex:
    row:        int     # row index in every store's R
    traj_idx:   int     # int(filename stem)
    step_idx:   int     # step index within the trajectory
    role:       str
    is_mistake: bool


@dataclass
class RepresentationStore:
    R:       torch.Tensor   # (T, d)
    pooling: str
    name:    str            # layer shorthand, e.g. "act/15"


@dataclass
class StoreKeeper:
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   dict[int, dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device


@dataclass
class RepresentationStores:
    stores: dict[tuple[str, str], RepresentationStore]
    keeper: StoreKeeper

    def positions(self) -> list[str]:
        return sorted({name for _, name in self.stores})

    def poolings(self) -> list[str]:
        return sorted({p for p, _ in self.stores})


# ── splits (verbatim — do not "simplify") ───────────────────────────────────
def split_data(data: list, ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle a copy under ``seed``, cut at ``int(len*ratio)``."""
    data = data.copy()
    random.seed(seed)
    random.shuffle(data)
    i = int(len(data) * ratio)
    return data[:i], data[i:]


def derive_split_ratios(train: float, val: float, test: float) -> tuple[float, float]:
    """Validate splits sum to 1; return (trval_vs_test, train_vs_val) ratios."""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1, got {train}+{val}+{test}={total}")
    if min(train, val, test) <= 0:
        raise ValueError(f"all splits must be > 0, got {train}/{val}/{test}")
    trval = train + val
    return trval, train / trval


def split_files(files: list[str], splits: dict, seed: int) -> dict[str, list[str]]:
    """Partition trajectories into {train, val, test} filename lists.

    Splitting is by TRAJECTORY, never by step: steps within a trajectory are ranked
    against each other, so splitting mid-trajectory would put a step's competitors in a
    different split and make the metric meaningless.

    Done as two sequential two-way cuts (trval|test, then train|val), each RESEEDED with
    the same ``seed``. That is more convoluted than one three-way cut and is kept
    exactly this way because the seed->partition mapping is the experiment's identity.
    """
    r_trval_test, r_train_val = derive_split_ratios(
        splits["train"], splits["val"], splits["test"])
    trval, test = split_data(files, r_trval_test, seed)
    train, val = split_data(trval, r_train_val, seed)
    return {"train": train, "val": val, "test": test}


def list_rep_files(rep_dir: Path | str) -> list[str]:
    """Filenames under rep_dir, sorted numerically by stem."""
    fs = sorted(Path(rep_dir).glob("*.safetensors"), key=lambda x: int(x.stem))
    assert fs, f"No .safetensors files in {rep_dir}"
    return [f.name for f in fs]


# ── safetensors key helpers ─────────────────────────────────────────────────
def _parse_key(key: str) -> tuple[int, str, str]:
    step_str, pooling, name = key.split(".", 2)      # name may contain "/"
    return int(step_str), pooling, name


def rep_names(fp: Path) -> list[str]:
    """Sorted set of layer names in a safetensors file (pooling-independent)."""
    with safe_open(fp, framework="pt") as f:
        return sorted({_parse_key(k)[2] for k in f.keys()})


# ── loader ──────────────────────────────────────────────────────────────────
def load_representations(
    rep_dir:   Path | str,
    data_dir:  Path | str,
    poolings:  list[str],
    weight_names: list[str] | str = "all",
    device:    torch.device | str = "cpu",
    files:     list[str] | None = None,
) -> RepresentationStores:
    """Load per-trajectory safetensors into one store per (pooling, name), shared keeper.

    Steps are enumerated once per file from the keys ending in the pivot suffix, so the
    keeper's row order is pooling-independent. ``data_dir`` is needed only to read each
    step's ``role`` — roles live in the corpus JSON, not in the safetensors.

    ``R`` comes back as stored (fp16); callers MUST ``.float()`` before arithmetic.
    """
    rep_dir, data_dir = Path(rep_dir), Path(data_dir)
    device = torch.device(device)
    if files is None:
        files = list_rep_files(rep_dir)
    if weight_names == "all":
        weight_names = rep_names(rep_dir / files[0])

    collections: dict[tuple[str, str], list[torch.Tensor]] = {
        (p, w): [] for p in poolings for w in weight_names
    }
    index: list[StepIndex] = []
    lookup: dict[tuple[int, int], int] = {}
    traj_meta: dict[int, dict] = {}
    traj_ranges: list[tuple[int, int]] = []
    row = 0

    for fname in files:
        tensor_file = rep_dir / fname
        traj_idx = int(tensor_file.stem)
        history = json.loads(
            (data_dir / tensor_file.with_suffix(".json").name).read_text())["history"]

        with safe_open(tensor_file, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata().get("payload_metadata", "{}"))
            mistake_step = int(meta.get("mistake_step", -1))
            pivot_suffix = f".{poolings[0]}.{weight_names[0]}"
            steps = sorted({int(k.split('.', 1)[0]) for k in f.keys()
                            if k.endswith(pivot_suffix)})
            for local, step_idx in enumerate(steps):
                for p in poolings:
                    for w in weight_names:
                        collections[(p, w)].append(f.get_tensor(f"{step_idx}.{p}.{w}"))
                index.append(StepIndex(
                    row=row + local, traj_idx=traj_idx, step_idx=step_idx,
                    role=history[step_idx]["role"], is_mistake=(step_idx == mistake_step)))

        n = len(steps)
        traj_ranges.append((row, row + n))
        traj_meta[traj_idx] = meta
        lookup.update({(si.traj_idx, si.step_idx): si.row for si in index[-n:]})
        row += n

    stores = {
        (p, w): RepresentationStore(R=torch.stack(tensors).to(device), pooling=p, name=w)
        for (p, w), tensors in collections.items()
    }
    keeper = StoreKeeper(index=index, lookup=lookup, traj_meta=traj_meta,
                         traj_ranges=traj_ranges, device=device)
    return RepresentationStores(stores=stores, keeper=keeper)
