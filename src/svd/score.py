"""SVD base-score runner for one (model, subset, pooling, seed).

Loads per-step representations, splits them (train/val/test), fits SVD on train,
scores every step by ranged projection across the config grid, and writes the
val+test metric table to
    {outputs_root}/{model}/{subset}/svd_pooling-{pooling}_seed-{seed}.tsv

This is the runner the SVD sweep shells out to (was
``experiments/svd/run_all_positions.py``); it now lives under ``src/`` so the
stage matches activations→``src.activations.extract`` and
attention→``src.attention.streaming``.

    python -m src.svd.score \
        --reps-root outputs-correct-error/activations \
        --data-root data/correct-error \
        --outputs-root outputs-correct-error/weighted-projections/325 \
        --model qwen3.5-9b --subset gaia --pooling mean --seed 1 \
        --device cuda --train-split 0.3 --val-split 0.2 --test-split 0.5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from src.utils.utils import load_representations, split_data
from src.svd.computation import precompute_svd, SCORING_FNS
from src.svd.reproduce import _validate_and_derive_split_ratios, N_COMPONENTS


def _key_hidden(s: str):
    """Sort key for layer shorthands: 'embed' first, then act/<N> numerically."""
    if s == "embed":
        return (-1, 0, "")
    num = int(re.search(r"(\d+)", s).group(1))
    return (0, num, s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps-root",    type=Path, default=Path("outputs/activations"))
    p.add_argument("--data-root",    type=Path, default=Path("data/ww"))
    p.add_argument("--outputs-root", type=Path, default=Path("outputs/metrics"))

    p.add_argument("--model",     type=str, required=True)
    p.add_argument("--subset",    type=str, required=True)
    p.add_argument("--pooling",   type=str, required=True)   # mean | last
    p.add_argument("--positions", nargs="+", type=str, default=["all"])

    p.add_argument("--seed",   type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--train-split", type=float, default=0.4)
    p.add_argument("--val-split",   type=float, default=0.2)
    p.add_argument("--test-split",  type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed   = args.seed

    rep_dir    = args.reps_root    / args.model / args.subset
    output_dir = args.outputs_root / args.model / args.subset
    data_dir   = args.data_root    / args.subset

    output_dir.mkdir(parents=True, exist_ok=True)
    svd_outpath = output_dir / f"svd_pooling-{args.pooling}_seed-{seed}.tsv"
    if svd_outpath.exists():
        print("[skipped] SVD file exists.")
        return

    print(f"Model: {args.model}  Subset: {args.subset}  "
          f"Pooling: {args.pooling}  Seed: {seed}")
    print(f"Representation dir: {rep_dir}")
    print(f"Data dir:           {data_dir}")

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    files = [file.name for file in files]  # silent failure if not included
    assert files, f"No .safetensors files in {rep_dir}"

    # Same two sequential split_data ratios the reproducer uses (shared helper).
    r_trval_test, r_train_val = _validate_and_derive_split_ratios(
        args.train_split, args.val_split, args.test_split,
    )
    print(f"Splits: train={args.train_split} val={args.val_split} test={args.test_split}")

    train_files, test_files = split_data(files,       r_trval_test, seed)
    train_files, val_files  = split_data(train_files, r_train_val,  seed)
    print(f"Trajectories: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    rep_kwargs = dict(rep_dir=rep_dir, data_dir=data_dir,
                      pooling=args.pooling, weight_names="all", device=device)
    train_reps = load_representations(**rep_kwargs, files=train_files)
    val_reps   = load_representations(**rep_kwargs, files=val_files)
    test_reps  = load_representations(**rep_kwargs, files=test_files)

    layer_idxs = sorted(train_reps.stores.keys(), key=_key_hidden)
    if args.positions != ["all"]:
        layer_idxs = [l for l in layer_idxs if l in args.positions]
    print(f"All positions: {layer_idxs}\n")

    precomputed = precompute_svd(
        train_reps, val_reps, test_reps,
        n_components=N_COMPONENTS, scoring_fns=SCORING_FNS, device=device,
    )
    svd_accuracy = precomputed["svd_accuracy"]
    svd_accuracy = svd_accuracy[svd_accuracy["direction"] == "asc"]
    svd_accuracy = svd_accuracy.sort_values("step_acc_test", ascending=False)

    svd_accuracy.to_csv(svd_outpath, sep="\t", index=False)
    print(f"Saved to {svd_outpath}")


if __name__ == "__main__":
    main()
