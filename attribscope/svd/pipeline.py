"""
attribscope.svd.pipeline



# indata
CUDA_VISIBLE_DEVICES=7 python -m attribscope.svd2.pipeline indata \
    --reps-root /data/hoang/attrib/outputs    \
    --data-root data/ww                       \
    --outputs-root /data/hoang/attrib/results \
    --model qwen3-8b                          \
    --subset hand-crafted                     \
    --rep-type grads                          \
    --pooling grad                            \
    --loss ntp                                \
    --n-components-fit 10                     \
    --n-components-score all                  \
    --ks 1 3 5 10                             \
    --split-ratio 0.5                         \
    --split-seed 0                            \
    --device cuda

# cross
CUDA_VISIBLE_DEVICES=7 python -m attribscope.svd2.pipeline cross \
    --reps-root /data/hoang/attrib/outputs    \
    --data-root data/ww                       \
    --outputs-root /data/hoang/attrib/results \
    --model qwen3-8b                          \
    --fit-subset algorithm-generated          \
    --score-subset hand-crafted               \
    --rep-type grads                          \
    --pooling grad                            \
    --loss kl_uniform                         \
    --n-components-fit 10                     \
    --n-components-score all                  \
    --ks 1 3 5 10                             \
    --device cuda

# self-ceiling, the same as indata without split configs.
CUDA_VISIBLE_DEVICES=7 python -m attribscope.svd2.pipeline self-ceiling \
    --reps-root /data/hoang/attrib/outputs    \
    --data-root data/ww                       \
    --outputs-root /data/hoang/attrib/results \
    --model qwen3-8b                          \
    --subset hand-crafted                     \
    --rep-type grads                          \
    --pooling grad                            \
    --loss ntp                                \
    --n-components-fit 10                     \
    --n-components-score all                  \
    --ks 1 3 5 10                             \
    --device cuda
"""
from __future__ import annotations

from itertools import product as iproduct
from pathlib import Path

import json
import pandas as pd
import torch
from tqdm import tqdm
from safetensors import safe_open
import argparse

from attribscope.svd.utils import (
    _resolve_dir,
    split_data,
    load_representations,
    RepresentationStores,
    run_metrics
)
from attribscope.svd.computation import fit_all, score_all

def run_pipeline(
    val_reps:           RepresentationStores,
    test_reps:          RepresentationStores,
    mode:               str,        # "A" → fit on val, "B" → fit on test
    n_components_fit:   int,
    n_components_score: list[int],
    ks:                 list[int],
    device:             torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit SVD on one split and score both splits.

    Parameters
    ----------
    val_reps / test_reps : pre-loaded representation stores for each split.
    mode                 : "A" fits on val, "B" fits on test. The split that
                           was *also* used for fitting is the transductive
                           output; the other is inductive.
    n_components_fit     : number of singular vectors to retain when fitting.
    n_components_score   : list of c values to evaluate at scoring time.
    ks                   : top-k cutoffs for the accuracy metrics.
    device               : torch device the store tensors live on.

    Returns
    -------
    (val_df, test_df) — one row per (weight × method × c × centered × direction × k).
    """
    if mode not in ("A", "B"):
        raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")

    fit_reps = val_reps if mode == "A" else test_reps
    svd      = fit_all(fit_reps.stores, n_components_fit)

    score_kwargs = dict(svd=svd, n_components_score=n_components_score, device=device)
    val_scores  = score_all(val_reps.stores,  **score_kwargs)
    test_scores = score_all(test_reps.stores, **score_kwargs)

    val_df  = run_metrics(val_scores,  keeper=val_reps.keeper,  ks=ks)
    test_df = run_metrics(test_scores, keeper=test_reps.keeper, ks=ks)

    merged_df = pd.merge(
        val_df, test_df, suffixes=('|val', '|test'),
        on=['weight', 'pooling', 'method', 'c', 'centered', 'direction', 'k'],
    )
    return merged_df

def run_indata(
    reps_root:    Path, # e,g., outputs/ or /data/username/attrib
    data_root:    Path, # e.g., data/ww
    outputs_root: Path, # e.g., outputs/ or /data/username/attrib

    model:        str, # llama-3.1-8b | qwen3-8b
    subset:       str, # algorithm-generated | hand-crafted
    rep_type:     str, # grads | hidden
    pooling:      str, # grads -> grad, hidden -> last | mean
    weight_names: str | list[str], # default `all`
    loss:         str, # ntp | kl_uniform | kl_temp
    temperature:  float | None, # None | None | 1.x

    n_components_fit:   int, # 10, by default,
    n_components_score: list[int], # [1, 2, 3, ..., 10]
    ks:                 list[int], # [1, 3, 5, 10]

    split_ratio:  float, # fit / score ratio, 0.3 or 0.5 for instance
    split_seed:   int,
    device:       torch.device = torch.device("cpu")
):

    if outputs_root:
        output_dir = _resolve_dir(
            root_dir=outputs_root, model=model, subset=subset,
            rep_type=rep_type, loss=loss, temperature=temperature,
            dir_type="metrics"
        )
        output_file = output_dir / \
            f"indata_{pooling}_{split_ratio}_{split_seed}.tsv"
        
        if output_file.exists():
            print(f" [skip] {output_file}")
            return

    rep_dir = _resolve_dir(
        root_dir=reps_root, model=model, subset=subset,
        rep_type=rep_type, loss=loss, temperature=temperature,
        dir_type="representations"
    )

    data_dir = data_root / subset

    print("Representation dir:", rep_dir)
    print("Trajectory dir:    ", data_dir)

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, (f"No .safetensors files in {rep_dir}")
    fit_files, score_files = split_data(files, split_ratio, split_seed)

    val_reps = load_representations(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=pooling,
        weight_names=weight_names,
        device=device,
        files=fit_files,
    )
    test_reps = load_representations(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=pooling,
        weight_names=weight_names,
        device=device,
        files=score_files,
    )

    metrics_df = run_pipeline(
        val_reps=val_reps, test_reps=test_reps,
        mode="A", # fit on val, score both
        n_components_fit=n_components_fit,
        n_components_score=n_components_score,
        ks=ks, device=device,
    )

    if outputs_root:
        metrics_df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved results at {output_file}")

    return metrics_df


def run_cross(
    reps_root:    Path, # e,g., outputs/ or /data/username/attrib
    data_root:    Path, # e.g., data/ww
    outputs_root: Path, # e.g., outputs/ or /data/username/attrib

    model:        str, # llama-3.1-8b | qwen3-8b
    fit_subset:   str, # algorithm-generated | hand-crafted | tau-retail
    score_subset: str, # algorithm-generated | hand-crafted
    rep_type:     str, # grads | hidden
    pooling:      str, # grads -> grad, hidden -> last | mean
    weight_names: str | list[str], # default `all`
    loss:         str, # ntp | kl_uniform | kl_temp
    temperature:  float | None, # None | None | 1.x

    n_components_fit:   int, # 10, by default,
    n_components_score: list[int], # [1, 2, 3, ..., 10]
    ks:                 list[int], # [1, 3, 5, 10]

    device:       torch.device = torch.device("cpu")
):

    if outputs_root:
        output_dir = _resolve_dir(
            root_dir=outputs_root, model=model, subset=score_subset,
            rep_type=rep_type, loss=loss, temperature=temperature,
            dir_type="metrics"
        )

        output_file = output_dir / f"cross_{pooling}_{fit_subset}.tsv"
        
        if output_file.exists():
            print(f" [skip] {output_file}")
            return
        
    # -- Loading subset for fitting SVD ----------------------
    rep_fit_dir = _resolve_dir(
        root_dir=reps_root, model=model, subset=fit_subset,
        rep_type=rep_type, loss=loss, temperature=temperature,
        dir_type="representations"
    )

    data_fit_dir = data_root / fit_subset

    print("Representation dir for fitting SVD:", rep_fit_dir)
    print("Trajectory dir for fitting SVD:    ", data_fit_dir)

    val_reps = load_representations(
        rep_dir=rep_fit_dir,
        data_dir=data_fit_dir,
        pooling=pooling,
        weight_names=weight_names,
        device=device,
    )

    # -- Loading subset for scoring with SVD ------------------

    rep_score_dir = _resolve_dir(
        root_dir=reps_root, model=model, subset=score_subset,
        rep_type=rep_type, loss=loss, temperature=temperature,
        dir_type="representations"
    )

    data_score_dir = data_root / score_subset

    print("Representation dir for scoring with SVD:", rep_score_dir)
    print("Trajectory dir for scoring with SVD:    ", data_score_dir)

    test_reps = load_representations(
        rep_dir=rep_score_dir,
        data_dir=data_score_dir,
        pooling=pooling,
        weight_names=weight_names,
        device=device,
    )

    # -- Running SVD fitting and scoring -----------------------

    metrics_df = run_pipeline(
        val_reps=val_reps, test_reps=test_reps,
        mode="A", # fit on val, score both
        n_components_fit=n_components_fit,
        n_components_score=n_components_score,
        ks=ks, device=device,
    )

    if outputs_root:
        metrics_df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved results at {output_file}")

    return metrics_df


def run_self_ceiling(
    reps_root:    Path, # e,g., outputs/ or /data/username/attrib/outputs
    data_root:    Path, # e.g., data/ww
    outputs_root: Path, # e.g., outputs/ or /data/username/attrib/metrics

    model:        str, # llama-3.1-8b | qwen3-8b
    subset:       str, # algorithm-generated | hand-crafted
    rep_type:     str, # grads | hidden
    pooling:      str, # grads -> grad, hidden -> last | mean
    weight_names: str | list[str], # default `all`
    loss:         str, # ntp | kl_uniform | kl_temp
    temperature:  float | None, # None | None | 1.x

    n_components_fit:   int, # 10, by default,
    n_components_score: list[int], # [1, 2, 3, ..., 10]
    ks:                 list[int], # [1, 3, 5, 10]

    device:       torch.device = torch.device("cpu")
):
    if outputs_root:
        output_dir = _resolve_dir(
            root_dir=outputs_root, model=model, subset=subset,
            rep_type=rep_type, loss=loss, temperature=temperature,
            dir_type="metrics"
        )
        
        output_file = output_dir / f"indata_{pooling}_ceiling.tsv"
        
        if output_file.exists():
            print(f" [skip] {output_file}")
            return
        
    rep_dir = _resolve_dir(
        root_dir=reps_root, model=model, subset=subset,
        rep_type=rep_type, loss=loss, temperature=temperature,
        dir_type="representations"
    )
    data_dir = data_root / subset

    print("Representation dir:", rep_dir)
    print("Trajectory dir:    ", data_dir)

    reps = load_representations(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=pooling,
        weight_names=weight_names,
        device=device,
    )

    metrics_df = run_pipeline(
        val_reps=reps, test_reps=reps,
        mode="A", # fit on val, score both
        n_components_fit=n_components_fit,
        n_components_score=n_components_score,
        ks=ks, device=device,
    )

    if outputs_root:
        metrics_df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved results at {output_file}")

    return metrics_df


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def common(q):
        q.add_argument("--reps-root",    type=Path, required=True)
        q.add_argument("--data-root",    type=Path, default=Path("data/ww"))
        q.add_argument("--outputs-root", type=Path, default=None)
        q.add_argument("--model",        required=True)
        q.add_argument("--rep-type",     required=True, choices=["grads", "hidden"])
        q.add_argument("--pooling",      required=True)
        q.add_argument("--weight-names", nargs="+", default=["all"])
        q.add_argument("--loss",         default="ntp", choices=["ntp", "kl_uniform", "kl_temp"])
        q.add_argument("--temperature",  type=float, default=None)
        q.add_argument("--n-components-fit",   type=int,   default=10)
        q.add_argument("--n-components-score", nargs="+",  default=["all"])
        q.add_argument("--ks",           nargs="+", type=int, default=[1, 3, 5, 10])
        q.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")

    p_indata = sub.add_parser("indata")
    common(p_indata)
    p_indata.add_argument("--subset",      required=True)
    p_indata.add_argument("--split-ratio", type=float, default=0.5)
    p_indata.add_argument("--split-seed",  type=int,   default=42)

    p_cross = sub.add_parser("cross")
    common(p_cross)
    p_cross.add_argument("--fit-subset",   required=True)
    p_cross.add_argument("--score-subset", required=True)

    p_ceil = sub.add_parser("self-ceiling")
    common(p_ceil)
    p_ceil.add_argument("--subset", required=True)

    return p.parse_args()


def main():
    args = parse_args()

    if args.outputs_root is None:
        args.outputs_root = args.reps_root

    weight_names = "all" if args.weight_names == ["all"] else args.weight_names
    n_components_score = (
        list(range(1, args.n_components_fit + 1))
        if args.n_components_score == ["all"]
        else [int(c) for c in args.n_components_score]
    )

    common = dict(
        reps_root=args.reps_root, 
        data_root=args.data_root, 
        outputs_root=args.outputs_root,
        model=args.model, 
        rep_type=args.rep_type, 
        pooling=args.pooling,
        weight_names=weight_names, 
        loss=args.loss, 
        temperature=args.temperature,
        n_components_fit=args.n_components_fit, 
        n_components_score=n_components_score,
        ks=args.ks, 
        device=torch.device(args.device),
    )

    if args.mode == "indata":
        run_indata(
            **common, 
            subset=args.subset,
            split_ratio=args.split_ratio, 
            split_seed=args.split_seed
        )
    elif args.mode == "cross":
        run_cross(
            **common, 
            fit_subset=args.fit_subset, 
            score_subset=args.score_subset
        )
    elif args.mode == "self-ceiling":
        run_self_ceiling(
            **common, 
            subset=args.subset
        )


if __name__ == "__main__":
    main()