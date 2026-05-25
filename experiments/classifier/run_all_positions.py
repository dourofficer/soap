import argparse
import pandas as pd
import numpy as np
import itertools
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from attribscope.classifier.classifier import (
    MLPClassifier,
    train, infer,
    seed_everything,
    key_hidden
)
from attribscope.svd2.utils import (
    RepresentationStores,
    load_representations,
    _resolve_dir,
    split_data,
    compute_metrics,
    get_mistake_meta
)
from attribscope.svd2.computation import fit_all, score_all
from attribscope.svd2.utils import run_metrics

# -------------------------------------------------------------------
# Hardcoded training configs
# -------------------------------------------------------------------
EPOCHS        = 500
LEARNING_RATE = 0.02
CLF_DIM       = 1024
BATCH_SIZE    = 512
WEIGHT_DECAY  = 3e-4
MOMENTUM      = 0.9
POS_WEIGHT    = None
LOGGING_STEPS = 100
VAL_METRIC    = "f1"


def precompute_svd(
    train_reps: RepresentationStores,
    val_reps:   RepresentationStores,
    test_reps:  RepresentationStores,
    n_components: int = 10,
    device: torch.device = torch.device("cuda")
):
    svd_components = fit_all(train_reps.stores, n_components=n_components)

    n_components_score = list(range(1, n_components + 1))
    score_kwargs = dict(
        svd=svd_components,
        n_components_score=n_components_score,
        device=device
    )
    train_scores = score_all(train_reps.stores, **score_kwargs)
    val_scores   = score_all(val_reps.stores,   **score_kwargs)
    test_scores  = score_all(test_reps.stores,  **score_kwargs)

    val_df  = run_metrics(val_scores,  keeper=val_reps.keeper,  ks=[1])
    test_df = run_metrics(test_scores, keeper=test_reps.keeper, ks=[1])
    merged_df = pd.merge(
        val_df, test_df, suffixes=('_val', '_test'),
        on=['weight', 'pooling', 'method', 'c', 'centered', 'direction', 'k'],
    )

    return dict(
        svd_components = svd_components,
        svd_accuracy   = merged_df,
        train_scores   = train_scores,
        val_scores     = val_scores,
        test_scores    = test_scores
    )


def get_position_best_config(
    scores, keeper, layer_idx
):
    metrics = run_metrics(scores, keeper=keeper, ks=[1])
    config = metrics.query(
        f"weight == '{layer_idx}' and direction == 'asc'"
    ).sort_values(["step_acc"], ascending=False).iloc[0].to_dict()
    return config

def get_pseudo_labels(
    train_scores,
    val_scores,
    val_reps,
    layer_idx,
    threshold,
    device
):
    config = get_position_best_config(val_scores, keeper=val_reps.keeper, layer_idx=layer_idx)
    QUERY = (
        f"weight == '{layer_idx}' "
        f"and pooling == '{config['pooling']}' "
        f"and method  == '{config['method']}' "
        f"and c == {config['c']} "
        f"and centered == {config['centered']}"
    )
    print("---" * 20)
    print(f"Using best validation config:\n{QUERY}")
    print(f"The best validation results with direct projection on SVD components")
    print(f"Step@1: {config['step_acc']:.4f} Agent@1: {config['agent_acc']:.4f}\n")

    pseudo_scores  = pd.DataFrame(train_scores).query(QUERY).iloc[0].scores
    wild_threshold = np.sort(pseudo_scores)[int(len(pseudo_scores) * threshold)]

    y_train_pseudo = torch.Tensor(
        (pseudo_scores < wild_threshold)
    ).to(device=device)
    return y_train_pseudo


def prepare_data(
    train_reps:   RepresentationStores,
    val_reps:     RepresentationStores,
    test_reps:    RepresentationStores,
    train_scores: list[dict],
    val_scores:   list[dict],
    layer_idx:    str,
    threshold:    float,
    mode:         str,  # oracle | pseudo
    batch_size:   int = BATCH_SIZE,
    device:       torch.device = torch.device("cuda"),
):
    X_train = train_reps.stores[layer_idx].R.float().to(device)
    y_train = torch.Tensor(
        [idx.is_mistake for idx in train_reps.keeper.index]
    ).to(device=X_train.device)

    # Pseudo labels estimation
    best_config = get_position_best_config(val_scores, keeper=val_reps.keeper, layer_idx=layer_idx)
    y_train_pseudo = get_pseudo_labels(
        train_scores, val_scores, val_reps,
        layer_idx, threshold,
        device=X_train.device
    )

    X_val = val_reps.stores[layer_idx].R.float().to(device)
    y_val = torch.Tensor(
        [idx.is_mistake for idx in val_reps.keeper.index]
    ).to(device=X_val.device)

    X_test = test_reps.stores[layer_idx].R.float().to(device)
    y_test = torch.Tensor(
        [idx.is_mistake for idx in test_reps.keeper.index]
    ).to(device=X_test.device)

    if mode == "oracle":
        print("Using labeled training data as ceiling reference.")
        y_train_effective = y_train
    elif mode == "pseudo": y_train_effective = y_train_pseudo
    else: raise ValueError(f"Unsupported mode: {mode}")

    train_loader = DataLoader(TensorDataset(X_train, y_train_effective), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return dict(
        train_loader = train_loader,
        val_loader   = val_loader,
        train        = (X_train, y_train),
        validation   = (X_val, y_val),
        test         = (X_test, y_test),
        best_config  = best_config,
    )


def get_metrics(
    clf:       MLPClassifier,
    val_reps:  RepresentationStores,
    test_reps: RepresentationStores,
    layer_idx: str,
    threshold: float,
    device:    torch.device = torch.device("cuda")
):
    val_mistake_indices,  val_mistake_roles  = get_mistake_meta(val_reps.keeper)
    test_mistake_indices, test_mistake_roles = get_mistake_meta(test_reps.keeper)

    X_val = val_reps.stores[layer_idx].R.float().to(device)
    val_scores = infer(clf, X_val, return_logits=False, device=device)
    val_metrics = compute_metrics(
        scores=val_scores, keeper=val_reps.keeper,
        mistake_indices=val_mistake_indices, mistake_roles=val_mistake_roles,
        ks=[1], direction="desc",
    )
    val_step_acc, val_agent_acc = list(val_metrics.values())

    X_test = test_reps.stores[layer_idx].R.float().to(device)
    test_scores = infer(clf, X_test, return_logits=False, device=device)
    test_metrics = compute_metrics(
        scores=test_scores, keeper=test_reps.keeper,
        mistake_indices=test_mistake_indices, mistake_roles=test_mistake_roles,
        ks=[1], direction="desc",
    )
    test_step_acc, test_agent_acc = list(test_metrics.values())

    print(
        f"  Layer {layer_idx:>10} | "
        f"Validation Step@1: {val_step_acc:.4f}  Agent@1: {val_agent_acc:.4f} | "
        f"Test  Step@1: {test_step_acc:.4f}  Agent@1: {test_agent_acc:.4f}"
    )

    return dict(
        labels         = "oracle" if threshold == 0.0 else "pseudo",
        position       = layer_idx,
        threshold      = threshold,
        val_step_acc   = val_step_acc,
        val_agent_acc  = val_agent_acc,
        test_step_acc  = test_step_acc,
        test_agent_acc = test_agent_acc,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # i/o configs
    parser.add_argument("--reps-root",    type=Path, default=Path("/data/hoang/attrib/outputs"))
    parser.add_argument("--data-root",    type=Path, default=Path("data/ww"))
    parser.add_argument("--outputs-root", type=Path, default=Path("/data/hoang/attrib/results_pseudo"))

    # model x data configs
    parser.add_argument("--model",     type=str, required=True)  # e.g. llama-3.1-8b
    parser.add_argument("--subset",    type=str, required=True)  # e.g. hand-crafted
    parser.add_argument("--pooling",   type=str, required=True)  # mean | last
    parser.add_argument("--positions", nargs="+", type=str, default=["all"])

    # training configs
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.0, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])

    # misc
    parser.add_argument("--seed",   type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    DEVICE = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed   = args.seed

    rep_dir = _resolve_dir(
        root_dir=args.reps_root,
        model=args.model,
        subset=args.subset,
        rep_type="hidden",
        loss=None,
        temperature=None,
        dir_type="representations"
    )
    output_dir = _resolve_dir(
        root_dir=args.outputs_root,
        model=args.model,
        subset=args.subset,
        rep_type="hidden",
        loss=None,
        temperature=None,
        dir_type="metrics"
    )
    data_dir = args.data_root / args.subset

    print(f"Model:              {args.model}")
    print(f"Subset:             {args.subset}")
    print(f"Pooling:            {args.pooling}")
    print(f"Seed:               {seed}")
    print(f"Representation dir: {rep_dir}")
    print(f"Data dir:           {data_dir}")

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, f"No .safetensors files in {rep_dir}"

    train_files, test_files = split_data(files, 0.5, seed)
    train_files, val_files  = split_data(train_files, 0.8, seed)

    print(f"Train trajectories: {len(train_files)}")
    print(f"Val trajectories:   {len(val_files)}")
    print(f"Test trajectories:  {len(test_files)}")

    rep_kwargs = dict(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=args.pooling,
        weight_names="all",
        device=args.device,
    )
    train_reps = load_representations(**rep_kwargs, files=train_files)
    val_reps   = load_representations(**rep_kwargs, files=val_files)
    test_reps  = load_representations(**rep_kwargs, files=test_files)

    LAYER_IDXS = sorted(train_reps.stores.keys(), key=key_hidden)
    if args.positions != ["all"]:
        LAYER_IDXS = [l for l in LAYER_IDXS if l in args.positions]
    print(f"All positions: {LAYER_IDXS}\n")

    # -------------------------------------------------------------------
    # SVD direct projection
    # -------------------------------------------------------------------
    precomputed_svd = precompute_svd(train_reps, val_reps, test_reps,
                                     n_components=20, device=DEVICE)
    train_scores = precomputed_svd["train_scores"]
    val_scores   = precomputed_svd["val_scores"]
    svd_accuracy = precomputed_svd["svd_accuracy"]

    svd_accuracy = svd_accuracy[svd_accuracy["direction"] == "asc"]
    svd_accuracy = svd_accuracy.sort_values("step_acc_val", ascending=False)

    svd_outpath = output_dir / f"svd_pooling-{args.pooling}_seed-{seed}.tsv"
    if svd_outpath.exists(): print("[skipped] SVD file exists.")
    else: svd_accuracy.to_csv(svd_outpath, sep="\t", index=False)


    # -------------------------------------------------------------------
    # Classifier sweep over positions x thresholds
    # -------------------------------------------------------------------
    metric_rows  = []
    inner_combos = list(itertools.product(LAYER_IDXS, args.thresholds))

    for icb, (layer_idx, threshold) in tqdm(enumerate(inner_combos)):
        print("---" * 20)
        print(f"COMBO: [{icb + 1}/{len(inner_combos)}] | POSITION: {layer_idx} | THRESHOLD: {threshold}")
        mode = "oracle" if threshold == 0.0 else "pseudo"

        prepared_data = prepare_data(
            train_reps, val_reps, test_reps, train_scores, val_scores,
            layer_idx=layer_idx, threshold=threshold, mode=mode, device=DEVICE
        )
        train_loader     = prepared_data["train_loader"]
        val_loader       = prepared_data["val_loader"]
        X_train, y_train = prepared_data["train"]
        best_val_config  = prepared_data["best_config"]

        seed_everything(seed)
        clf = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=CLF_DIM)
        clf, _ = train(clf,
            train_loader  = train_loader,
            val_loader    = val_loader,
            epochs        = EPOCHS,
            learning_rate = LEARNING_RATE,
            weight_decay  = WEIGHT_DECAY,
            momentum      = MOMENTUM,
            pos_weight    = None,
            logging_steps = LOGGING_STEPS,
            val_metric    = VAL_METRIC,
            device        = DEVICE,
        )
        metrics = get_metrics(clf, val_reps, test_reps, layer_idx, threshold, DEVICE)
        metric_rows.append({**best_val_config, **metrics})

    metric_df = pd.DataFrame(metric_rows).sort_values("test_step_acc", ascending=False)
    output_path = output_dir / f"classifier_pooling-{args.pooling}_seed-{seed}.tsv"
    metric_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved results to {output_path}")