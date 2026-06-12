"""Reproduce per-step val + test scores for a single undiscounted-table row.

Entry point:
  - reproduce_svd(row, ...)  — direct SVD-projection scores
                               (convention: lower = error; orient before discounting)

CLI: spot-check a single row against the undiscounted table.

python -m src.svd.reproduce \
    --table outputs-1006/undiscounted-splits/112/qwen3-8b/algorithm-generated/weighted_false.tsv \
    --row 0 \
    --model qwen3-8b --subset algorithm-generated \
    --reps-root outputs/activations \
    --data-root data/ww \
    --train-split 0.25 \
    --val-split 0.25 \
    --test-split 0.5 \
    --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.svd.computation import precompute_svd
from src.utils.utils import (
    load_representations, split_data,
    compute_metrics,
)
from src.svd.computation import SCORING_FNS


# ── Hyperparameters / paths ──────────────────────────────────────────────────
N_COMPONENTS = 20

REP_TYPE     = "hidden"
WEIGHT_NAMES = "all"


def _validate_and_derive_split_ratios(
    train_split: float, val_split: float, test_split: float,
) -> tuple[float, float]:
    """Validate that splits sum to 1 and return the two two-way ratios
    (trval_vs_test, train_vs_val) used by sequential split_data calls."""
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test splits must sum to 1, got "
            f"{train_split} + {val_split} + {test_split} = {total}"
        )
    if min(train_split, val_split, test_split) <= 0:
        raise ValueError(
            f"all splits must be > 0, got "
            f"train={train_split}, val={val_split}, test={test_split}"
        )
    trval        = train_split + val_split          # 1 - test_split
    r_trval_test = trval                            # first pass: trval vs test
    r_train_val  = train_split / trval              # second pass: train vs val out of trval
    return r_trval_test, r_train_val


@dataclass
class ScoreBundle:
    val_scores:  torch.Tensor
    val_keeper:  Any
    test_scores: torch.Tensor
    test_keeper: Any


# ── Caching: avoid redoing the data-loading + SVD precompute per row ─────────
_cache: dict[tuple, dict] = {}


def _load_reps_and_svd(
    model: str, subset: str, pooling: str, seed: int,
    reps_root: Path, data_root: Path, device: str,
    train_split: float, val_split: float, test_split: float,
) -> dict:
    """Load + split reps and run precompute_svd. Cached per
    (model, subset, pooling, seed, reps_root, data_root, splits)."""
    r_trval_test, r_train_val = _validate_and_derive_split_ratios(
        train_split, val_split, test_split,
    )

    key = (model, subset, pooling, seed, str(reps_root), str(data_root),
           train_split, val_split, test_split)
    if key in _cache:
        return _cache[key]

    rep_dir  = reps_root / model / subset
    data_dir = data_root / subset

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    files = [file.name for file in files]
    assert files, f"No .safetensors files in {rep_dir}"

    trval_files, test_files = split_data(files,       r_trval_test, seed)
    train_files, val_files  = split_data(trval_files, r_train_val,  seed)

    rk = dict(rep_dir=rep_dir, data_dir=data_dir,
              pooling=pooling, weight_names=WEIGHT_NAMES, device=device)

    train_reps = load_representations(**rk, files=train_files)
    val_reps   = load_representations(**rk, files=val_files)
    test_reps  = load_representations(**rk, files=test_files)

    svd_pre = precompute_svd(
        train_reps, val_reps, test_reps,
        n_components=N_COMPONENTS, scoring_fns=SCORING_FNS, device=device,
    )

    bundle = {
        "train_reps": train_reps, "val_reps": val_reps, "test_reps": test_reps,
        "svd": svd_pre,
    }
    _cache[key] = bundle
    return bundle


def _select_svd_scores(score_records, row) -> torch.Tensor:
    """Pick the SVD per-step score vector matching this row's config."""
    df = pd.DataFrame(score_records) if not isinstance(score_records, pd.DataFrame) else score_records
    mask = (
        (df["position"]   == row["position"]) &
        (df["pooling"]  == row["pooling"]) &
        (df["method"]   == row["method"]) &
        (df["c_begin"]  == int(row["c_begin"])) &
        (df["c_end"]    == int(row["c_end"])) &
        (df["centered"] == bool(row["centered"])) &
        (df["weighted"] == bool(row["weighted"]))
    )
    hits = df[mask]
    if len(hits) == 0:
        raise ValueError(
            f"no SVD score record matching "
            f"{dict((c, row[c]) for c in ['position','pooling','method','c_begin','c_end','centered','weighted'])}"
        )
    return torch.as_tensor(hits.iloc[0]["scores"]).float()


# ── Public entry point ───────────────────────────────────────────────────────

def reproduce_svd(
    row,
    model: str, subset: str,
    reps_root: Path, data_root: Path,
    device: str = "cuda",
    *,
    train_split: float = None,
    val_split:   float = None,
    test_split:  float = None,
) -> ScoreBundle:
    """SVD direct-projection scores. Convention: lower = error.

    Splits default to 40/20/40 (matching the original undiscounted-table
    runs). Pass matching splits if reproducing a row from a non-default sweep.
    """
    bundle = _load_reps_and_svd(
        model, subset, row["pooling"], int(row["seed"]),
        reps_root, data_root, device,
        train_split, val_split, test_split,
    )
    val_scores  = _select_svd_scores(bundle["svd"]["val_scores"],  row).cpu()
    test_scores = _select_svd_scores(bundle["svd"]["test_scores"], row).cpu()
    return ScoreBundle(
        val_scores=val_scores,  val_keeper=bundle["val_reps"].keeper,
        test_scores=test_scores, test_keeper=bundle["test_reps"].keeper,
    )


# ── Spot-check CLI ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Spot-check SVD score reproduction.")
    ap.add_argument("--table", required=True, type=Path,
                    help="Undiscounted TSV (outputs/tables/undiscounted/...).")
    ap.add_argument("--row", required=True, type=int, help="0-based row index.")
    ap.add_argument("--model",  required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--reps-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-split", type=float, default=None, help=f"Train fraction.")
    ap.add_argument("--val-split",   type=float, default=None, help=f"Val fraction.")
    ap.add_argument("--test-split",  type=float, default=None, help=f"Test fraction. "
                         "Must sum with train/val to 1.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df  = pd.read_csv(args.table, sep="\t")
    row = df.iloc[args.row]
    print(f"\nRow {args.row}: strategy={row['strategy']} position={row['position']} "
          f"pooling={row['pooling']} seed={int(row['seed'])}")

    if row["strategy"] != "svd":
        raise ValueError(f"expected strategy='svd', got {row['strategy']!r}")

    bundle = reproduce_svd(
        row, args.model, args.subset,
        args.reps_root, args.data_root, args.device,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    # SVD: lower = error → negate for "higher = error" metric convention.
    val_m  = compute_metrics(-bundle.val_scores,  bundle.val_keeper,  ks=[1], direction="desc")
    test_m = compute_metrics(-bundle.test_scores, bundle.test_keeper, ks=[1], direction="desc")

    val_step_acc,  val_agent_acc  = list(val_m.values())
    test_step_acc, test_agent_acc = list(test_m.values())
    print("\n  reproduced   val:  step={:.4f}  agent={:.4f}".format(val_step_acc,  val_agent_acc))
    print("  table        val:  step={:.4f}  agent={:.4f}".format(row["step_acc_val"],  row["agent_acc_val"]))
    print("  reproduced   test: step={:.4f}  agent={:.4f}".format(test_step_acc, test_agent_acc))
    print("  table        test: step={:.4f}  agent={:.4f}".format(row["step_acc_test"], row["agent_acc_test"]))


if __name__ == "__main__":
    main()