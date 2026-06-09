"""
python -m experiments.svd.run_all_positions \
    --reps-root    "outputs/activations" \
    --data-root    "data/ww" \
    --outputs-root "outputs/projections" \
    --model        qwen3-8b \
    --subset       algorithm-generated \
    --pooling      mean \
    --positions    all \
    --seed         1 \
    --device       cuda
"""

import argparse
import pandas as pd
import numpy as np
import itertools
import torch
import re
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from typing import Callable

from src.utils.utils import (
    load_representations,
    split_data,
    gather_configs_and_metrics,
)
from src.svd.computation import precompute_svd, SCORING_FNS

# -------------------------------------------------------------------
# Scoring functions
# -------------------------------------------------------------------
def key_hidden(s):
    if s == 'embed': return (-1, 0, '')
    match = re.search(r'(\d+)', s)
    num = int(match.group(1))
    return (0, num, s)


def get_position_best_config(
    scores, keeper, layer_idx
):
    metrics = gather_configs_and_metrics(scores, keeper=keeper, ks=[1])
    config = metrics.query(
        f"weight == '{layer_idx}' and direction == 'asc'"
    ).sort_values(["step_acc"], ascending=False).iloc[0].to_dict()
    return config

def parse_args():
    parser = argparse.ArgumentParser()

    # i/o configs
    parser.add_argument("--reps-root",    type=Path, default=Path("outputs/activations"))
    parser.add_argument("--data-root",    type=Path, default=Path("data/ww"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/metrics"))

    # model x data configs
    parser.add_argument("--model",     type=str, required=True)  # e.g. llama-3.1-8b
    parser.add_argument("--subset",    type=str, required=True)  # e.g. hand-crafted
    parser.add_argument("--pooling",   type=str, required=True)  # mean | last
    parser.add_argument("--positions", nargs="+", type=str, default=["all"])

    # misc
    parser.add_argument("--seed",   type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args   = parse_args()
    DEVICE = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed   = args.seed

    rep_dir    = args.reps_root    / args.model / args.subset
    output_dir = args.outputs_root / args.model / args.subset
    data_dir   = args.data_root    / args.subset

    output_dir.mkdir(parents=True, exist_ok=True)
    svd_outpath = output_dir / f"svd_pooling-{args.pooling}_seed-{seed}.tsv"
    if svd_outpath.exists(): 
        print("[skipped] SVD file exists.")
        return

    print(f"Model:              {args.model}")
    print(f"Subset:             {args.subset}")
    print(f"Pooling:            {args.pooling}")
    print(f"Seed:               {seed}")
    print(f"Representation dir: {rep_dir}")
    print(f"Data dir:           {data_dir}")

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    files = [file.name for file in files] # silent failure if not included
    assert files, f"No .safetensors files in {rep_dir}"

    # Split 4 : 2 : 4 (train : val : test)
    train_files, test_files = split_data(files, 0.6, seed)
    train_files, val_files  = split_data(train_files, 2/3, seed)

    print(f"Train trajectories: {len(train_files)}")
    print(f"Val trajectories:   {len(val_files)}")
    print(f"Test trajectories:  {len(test_files)}")

    rep_kwargs = dict(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=args.pooling,
        weight_names="all",
        device=DEVICE,
    )
    # breakpoint()
    train_reps = load_representations(**rep_kwargs, files=train_files)
    val_reps   = load_representations(**rep_kwargs, files=val_files)
    test_reps  = load_representations(**rep_kwargs, files=test_files)

    LAYER_IDXS = sorted(train_reps.stores.keys(), key=key_hidden)[:]
    if args.positions != ["all"]:
        LAYER_IDXS = [l for l in LAYER_IDXS if l in args.positions]
    print(f"All positions: {LAYER_IDXS}\n")

    # -------------------------------------------------------------------
    # SVD direct projection
    # -------------------------------------------------------------------
    precomputed_svd = precompute_svd(
        train_reps, val_reps, test_reps,
        n_components=20, scoring_fns=SCORING_FNS, device=DEVICE,
    )
    train_scores = precomputed_svd["train_scores"]
    val_scores   = precomputed_svd["val_scores"]
    test_scores  = precomputed_svd["test_scores"]

    svd_accuracy = precomputed_svd["svd_accuracy"]
    svd_accuracy = svd_accuracy[svd_accuracy["direction"] == "asc"]
    svd_accuracy = svd_accuracy.sort_values("step_acc_test", ascending=False)

    svd_accuracy.to_csv(svd_outpath, sep="\t", index=False)
    print(f"Saved to {svd_outpath}")

if __name__ == "__main__":
    main()