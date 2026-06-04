"""Reproduce per-step val + test scores for a single undiscounted-table row.

Two entry points, both returning a ScoreBundle:
  - reproduce_svd(row, ...)        — direct SVD-projection scores
                                     (convention: lower = error; orient before discounting)
  - reproduce_classifier(row, ...) — classifier head output
                                     (convention: higher = error; no orient needed)

CLI: spot-check a single row against the undiscounted table.

python -m attribscope.discount.reproduce \
    --table outputs/tables/undiscounted/qwen3-8b__hand-crafted.tsv \
    --row 0 \
    --model qwen3-8b --subset hand-crafted \
    --reps-root outputs/representation-full \
    --data-root data/ww \
    --device cuda

The reproduction uses the same helpers that produced the undiscounted TSVs:
prepare_data / precompute_svd / run_one from attribscope.classifier.run_all_positions.
If your run_all_positions uses different split ratios or training hyperparameters,
adjust the constants below to match.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from attribscope.classifier.classifier import (
    seed_everything, MLPClassifier, train, infer,
)
from attribscope.classifier.run_all_positions import (
    prepare_data, precompute_svd,
)
from attribscope.svd.utils import (
    load_representations, split_data, 
    _resolve_dir, compute_metrics
)
from attribscope.svd.computation import SCORING_FNS


# ── Hyperparameters / paths (must match run_all_positions defaults) ──────────
N_COMPONENTS  = 20
EPOCHS        = 500
LR            = 0.02
HIDDEN_DIM    = 1024
WEIGHT_DECAY  = 3e-4
MOMENTUM      = 0.9
LOGGING_STEPS = 100
VAL_METRIC    = "f1"

# 40/20/40 train/val/test split via two passes of split_data.
SPLIT_TRVAL_VS_TEST = 0.6      # 60% train+val, 40% test
SPLIT_TRAIN_VS_VAL  = 2.0 / 3  # of the 60%, 2/3 train (=40%) + 1/3 val (=20%)

REP_TYPE     = "hidden"
WEIGHT_NAMES = "all"


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
) -> dict:
    """Load + split reps and run precompute_svd. Cached per
    (model, subset, pooling, seed, reps_root, data_root)."""
    key = (model, subset, pooling, seed, str(reps_root), str(data_root))
    if key in _cache:
        return _cache[key]

    rep_dir = _resolve_dir(
        root_dir=reps_root,
        model=model,
        subset=subset,
        rep_type=REP_TYPE,
        loss=None,
        temperature=None,
        dir_type="representations"
    )
    data_dir = data_root / subset

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    files = [file.name for file in files] # silent failure if not included
    assert files, f"No .safetensors files in {rep_dir}"

    # # Split 4 : 2 : 4 (train : val : test)
    trval_files, test_files  = split_data(files,         SPLIT_TRVAL_VS_TEST, seed)
    train_files, val_files   = split_data(trval_files,   SPLIT_TRAIN_VS_VAL,  seed)

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
    """Pick the SVD per-step score vector matching this row's config (direction='asc')."""
    df = pd.DataFrame(score_records) if not isinstance(score_records, pd.DataFrame) else score_records
    mask = (
        (df["weight"]    == row["weight"]) &
        (df["pooling"]   == row["pooling"]) &
        (df["method"]    == row["method"]) &
        (df["c_begin"]   == int(row["c_begin"])) &
        (df["c_end"]     == int(row["c_end"])) &
        (df["centered"]  == bool(row["centered"]))
    )
    hits = df[mask]
    if len(hits) == 0:
        raise ValueError(
            f"no SVD score record matching {dict((c, row[c]) for c in ['weight','pooling','method','c_begin','c_end','centered'])}"
        )
    return torch.as_tensor(hits.iloc[0]["scores"]).float()


# ── Public entry points ──────────────────────────────────────────────────────

def reproduce_svd(
    row,
    model: str, subset: str,
    reps_root: Path, data_root: Path,
    device: str = "cuda",
) -> ScoreBundle:
    """SVD direct-projection scores. Convention: lower = error."""
    bundle = _load_reps_and_svd(
        model, subset, row["pooling"], int(row["seed"]),
        reps_root, data_root, device,
    )
    # breakpoint()
    val_scores  = _select_svd_scores(bundle["svd"]["val_scores"],  row).cpu()
    test_scores = _select_svd_scores(bundle["svd"]["test_scores"], row).cpu()
    return ScoreBundle(
        val_scores=val_scores,  val_keeper=bundle["val_reps"].keeper,
        test_scores=test_scores, test_keeper=bundle["test_reps"].keeper,
    )


def reproduce_classifier(
    row,
    model: str, subset: str,
    reps_root: Path, data_root: Path,
    device: str = "cuda",
) -> ScoreBundle:
    """Classifier scores. Convention: higher = error."""
    bundle = _load_reps_and_svd(
        model, subset, row["pooling"], int(row["seed"]),
        reps_root, data_root, device,
    )
    layer     = row["weight"]
    threshold = float(row["threshold"])
    mode      = "oracle" if threshold == 0.0 else "pseudo"
    seed      = int(row["seed"])

    prepared = prepare_data(
        bundle["train_reps"], bundle["val_reps"], bundle["test_reps"],
        bundle["svd"]["train_scores"],
        bundle["svd"]["val_scores"],
        bundle["svd"]["test_scores"],
        layer_idx=layer, threshold=threshold, mode=mode,
        ceiling=False, device=device,
    )

    seed_everything(seed)
    clf = MLPClassifier(
        input_dim=prepared["train"][0].shape[1],
        hidden_dim=HIDDEN_DIM,
    )
    clf, _ = train(
        clf,
        train_loader  = prepared["train_loader"],
        val_loader    = prepared["val_loader"],   # ceiling=False ⇒ val for early stopping
        epochs        = EPOCHS,
        learning_rate = LR,
        weight_decay  = WEIGHT_DECAY,
        momentum      = MOMENTUM,
        pos_weight    = None,
        logging_steps = LOGGING_STEPS,
        val_metric    = VAL_METRIC,
        device        = device,
    )

    X_val  = bundle["val_reps"].stores[layer].R.float().to(device)
    X_test = bundle["test_reps"].stores[layer].R.float().to(device)
    val_scores  = infer(clf, X_val,  return_logits=False, device=device).detach().cpu().float()
    test_scores = infer(clf, X_test, return_logits=False, device=device).detach().cpu().float()
    return ScoreBundle(
        val_scores=val_scores,  val_keeper=bundle["val_reps"].keeper,
        test_scores=test_scores, test_keeper=bundle["test_reps"].keeper,
    )


# ── Spot-check CLI ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Spot-check score reproduction.")
    ap.add_argument("--table", required=True, type=Path,
                    help="Undiscounted TSV (outputs/tables/undiscounted/...).")
    ap.add_argument("--row", required=True, type=int, help="0-based row index.")
    ap.add_argument("--model",  required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--reps-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df  = pd.read_csv(args.table, sep="\t")
    row = df.iloc[args.row]
    print(f"\nRow {args.row}: strategy={row['strategy']} weight={row['weight']} "
          f"pooling={row['pooling']} seed={int(row['seed'])}")

    if row["strategy"] == "svd":
        bundle = reproduce_svd(row, args.model, args.subset,
                                args.reps_root, args.data_root, args.device)
        # SVD: lower = error → negate for "higher = error" metric convention.
        v_for_metric = -bundle.val_scores
        t_for_metric = -bundle.test_scores
    elif row["strategy"].startswith("classifier_"):
        bundle = reproduce_classifier(row, args.model, args.subset,
                                       args.reps_root, args.data_root, args.device)
        v_for_metric = bundle.val_scores
        t_for_metric = bundle.test_scores
    else:
        raise ValueError(f"unknown strategy: {row['strategy']!r}")

    val_m  = compute_metrics(v_for_metric, bundle.val_keeper, ks=[1], direction="desc")
    test_m = compute_metrics(t_for_metric, bundle.test_keeper, ks=[1], direction="desc")

    val_step_acc, val_agent_acc = list(val_m.values())
    test_step_acc, test_agent_acc = list(test_m.values())
    print("\n  reproduced   val:  step={:.4f}  agent={:.4f}".format(
        val_step_acc, val_agent_acc))
    print("  table        val:  step={:.4f}  agent={:.4f}".format(
        row["step_acc_val"],  row["agent_acc_val"]))
    print("  reproduced   test: step={:.4f}  agent={:.4f}".format(
        test_step_acc, test_agent_acc))
    print("  table        test: step={:.4f}  agent={:.4f}".format(
        row["step_acc_test"], row["agent_acc_test"]))


if __name__ == "__main__":
    main()