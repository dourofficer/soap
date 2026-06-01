"""
sweep_reweighting.py — dependency-aware reweighting sweep over best classifier configs.

For each best (pooling, seed) row in the oracle/pseudo classifier-results TSVs,
retrain the classifier with the recorded layer/threshold, then sweep reweighting
across (gamma, k) and save all metrics to a single TSV. Alongside the classifier
scores, the direct SVD-projection scores for the same best config are reweighted
too, as an unsupervised baseline (recorded with mode == "svd").

Uses a single train/validation split (6:4), aligned with the ceiling experiments.

Usage:
CUDA_VISIBLE_DEVICES=5 python -m attribscope.reweight.run \
    --model llama-3.1-8b \
    --subset hand-crafted \
    --device cuda

CUDA_VISIBLE_DEVICES=5 python -m attribscope.reweight.run \
    --model llama-3.1-8b \
    --subset algorithm-generated \
    --device cuda

CUDA_VISIBLE_DEVICES=5 python -m attribscope.reweight.run \
    --model qwen3-8b \
    --subset hand-crafted \
    --device cuda

CUDA_VISIBLE_DEVICES=5 python -m attribscope.reweight.run \
    --model qwen3-8b \
    --subset algorithm-generated \
    --device cuda
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from attribscope.svd.utils import (
    load_representations, _resolve_dir, split_data,
)
from experiments.classifier.run_all_positions_noval import prepare_data as prepare_data_normal
from experiments.classifier.run_all_positions_noval import precompute_svd as precompute_svd_normal
from experiments.classifier.run_all_positions_noval_trunc import prepare_data as prepare_data_trunc
from experiments.classifier.run_all_positions_noval_trunc import precompute_svd as precompute_svd_truncate
from attribscope.classifier.classifier import (
    seed_everything, MLPClassifier, infer, train
)
from attribscope.reweight.reweight import (
    load_weighting, process_weighting, resolve_layer,
    reweight_scores, compute_metrics,
)

# ── Sweep hyperparameters (fixed) ────────────────────────────────────────────
GAMMAS       = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 … 1.0
KS           = [1, 2, 3, 5, 100]                            # 100 ≈ full context
TEMP         = 64 # 1.0 - change nothing | math.sqrt(4096)
SIM          = "raw_dot"
N_COMPONENTS = 20
EPOCHS       = 500
LR           = 0.02
WEIGHT_DECAY = 3e-4
MOMENTUM     = 0.9
HIDDEN_DIM   = 1024
LOGGING_STEPS = 100
VAL_METRIC   = "f1"
SPLIT_RATIO  = 0.6                                          # train : val = 6 : 4
SVD_ORIENT   = "inverse"                                     # "negate" | "inverse" | "sigmoid"

# ── Paths ────────────────────────────────────────────────────────────────────
REPS_ROOT    = Path("outputs/representation-full")
DATA_ROOT    = Path("data/ww")
RESULTS_ROOT = Path("outputs/classifier-full-noval")
REP_TYPE     = "hidden"
WEIGHT_NAMES = "all"


def orient_svd_scores(scores: torch.Tensor, strategy: str = SVD_ORIENT) -> torch.Tensor:
    """Flip SVD projection scores (lower = mistake) to the pipeline
    convention (higher = mistake) before reweighting."""
    if strategy == "negate":
        return -scores
    elif strategy == "inverse":
        return 1.0 / scores            # assumes scores > 0
    elif strategy == "sigmoid":
        return torch.sigmoid(-scores)  # decreasing in scores → maps to (0, 1)
    else:
        raise ValueError(f"Unknown orient strategy: {strategy}")


def select_svd_score_vector(score_records: list[dict], cfg: dict, comp_cols: list[str]):
    """Pull the single SVD test-score vector matching the best cfg row.

    Matches on (weight, pooling, centered, *comp_cols). `method` is intentionally
    not part of the key; in the trunc path the `norm` rows carry None for the
    component bounds, so a cfg with concrete component values won't match them.
    """
    df   = pd.DataFrame(score_records)
    mask = (
        (df["weight"]   == cfg["weight"]) &
        (df["pooling"]  == cfg["pooling"]) &
        (df["centered"] == bool(cfg["centered"]))
    )
    for col in comp_cols:
        mask &= (df[col] == cfg[col])
    hits = df[mask]
    assert len(hits) > 0, (
        f"No SVD score record for weight={cfg['weight']} pooling={cfg['pooling']} "
        f"{ {c: cfg[c] for c in comp_cols} } centered={cfg['centered']}"
    )
    return hits.iloc[0]["scores"]


def load_best_configs(
    model: str, subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read classifier-results TSVs and pick best (pooling, seed) row per mode.

    Returns the oracle/pseudo report frames plus the component-column names
    actually present in the results (`["c"]` or `["c_begin", "c_end"]`).
    """
    base   = RESULTS_ROOT / REP_TYPE / model / "metrics" / subset
    frames = []
    # breakpoint()
    for path in sorted(base.glob("classifier_*.tsv")):
        parts   = path.stem.split("_")
        pooling = parts[1][len("pooling-"):]
        seed    = int(parts[2][len("seed-"):])
        frames.append(
            pd.read_csv(path, sep="\t").assign(pooling=pooling, seed=seed)
        )
        # breakpoint()
    clf_df = pd.concat(frames, ignore_index=True)

    # Detect which component columns this results file uses.
    if {"c_begin", "c_end"}.issubset(clf_df.columns):
        comp_cols = ["c_begin", "c_end"]
    elif "c" in clf_df.columns:
        comp_cols = ["c"]
    else:
        raise KeyError(
            f"Expected 'c' or 'c_begin'/'c_end' in {base}; got {list(clf_df.columns)}"
        )

    cols = ["threshold", "weight", "pooling", *comp_cols, "centered", "seed",
            "test_step_acc"]

    reports = {}
    for label in ["oracle", "pseudo"]:
        sub = clf_df.query("labels == @label")[cols]
        idx = sub.groupby(["pooling", "seed"])["test_step_acc"].idxmax()
        reports[label] = sub.loc[idx].reset_index(drop=True)
    return reports["oracle"], reports["pseudo"], comp_cols


def run_sweep(model: str, subset: str, device: torch.device) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    oracle_report, pseudo_report, comp_cols = load_best_configs(model, subset)
    if comp_cols == ['c']: 
        prepare_data = prepare_data_normal
        precompute_svd = precompute_svd_normal
        svd_kwargs = dict(
            n_components=N_COMPONENTS,
            device=device
        )
    elif comp_cols == ['c_begin', 'c_end']: 
        from attribscope.svd.computation import ranged_projection_svd
        SCORING_FNS = {"trunc_proj": ranged_projection_svd}
        prepare_data = prepare_data_trunc
        precompute_svd = precompute_svd_truncate
        svd_kwargs = dict(
            n_components=N_COMPONENTS,
            scoring_fns=SCORING_FNS,
            device=device
        )
    else:
        raise ValueError(f"What are these columns: {comp_cols}")
    
    rep_dir = _resolve_dir(
        root_dir=REPS_ROOT, model=model, subset=subset, rep_type=REP_TYPE,
        loss=None, temperature=None, dir_type="representations",
    )
    data_dir = DATA_ROOT / subset

    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    files = [f.name for f in files]
    assert files, f"No .safetensors files in {rep_dir}"

    out_dir  = Path("outputs/reweighted_results") / model / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reweight_sweep_noval.tsv"

    # ── Shared weighting, loaded once ────────────────────────────────────────
    all_weights = load_weighting(model, subset, device=device)

    # ── Cache reps + SVD per (pooling, seed) ─────────────────────────────────
    rep_cache: dict[tuple[str, int], dict] = {}

    def get_reps_and_svd(pooling: str, seed: int) -> dict:
        key = (pooling, seed)
        if key in rep_cache:
            return rep_cache[key]

        tr_files, te_files = split_data(files, SPLIT_RATIO, seed)

        rk = dict(rep_dir=rep_dir, data_dir=data_dir,
                  pooling=pooling, weight_names=WEIGHT_NAMES, device=device)
        train_reps = load_representations(**rk, files=tr_files)
        test_reps  = load_representations(**rk, files=te_files)

        svd_pre = precompute_svd(
            train_reps, test_reps,
            **svd_kwargs
        )
        rep_cache[key] = {
            "train_reps": train_reps,
            "test_reps":  test_reps,
            "svd":        svd_pre,
        }
        return rep_cache[key]

    def train_and_score(cfg: dict, mode: str):
        pooling   = cfg["pooling"]
        seed      = int(cfg["seed"])
        layer     = cfg["weight"]
        threshold = float(cfg["threshold"]) if mode == "pseudo" else 0.0

        cached   = get_reps_and_svd(pooling, seed)
        prepared = prepare_data(
            cached["train_reps"], cached["test_reps"],
            cached["svd"]["train_scores"], cached["svd"]["test_scores"],
            layer_idx=layer, threshold=threshold, mode=mode, device=device,
        )

        seed_everything(seed)
        clf = MLPClassifier(input_dim=prepared["train"][0].shape[1],
                            hidden_dim=HIDDEN_DIM)
        clf, _ = train(clf,
            train_loader  = prepared["train_loader"],
            val_loader    = prepared["test_loader"],
            epochs        = EPOCHS,
            learning_rate = LR,
            weight_decay  = WEIGHT_DECAY,
            momentum      = MOMENTUM,
            pos_weight    = None,
            logging_steps = LOGGING_STEPS,
            val_metric    = VAL_METRIC,
            device        = device,
        )

        X_test = cached["test_reps"].stores[layer].R.float().to(device)
        test_scores = infer(clf, X_test, return_logits=False, device=device)
        return test_scores, cached["test_reps"]

    # ── Sweep ────────────────────────────────────────────────────────────────
    records = []
    reports = [("oracle", oracle_report), ("pseudo", pseudo_report)]
    n_total = sum(len(df) for _, df in reports)

    with tqdm(total=n_total, desc="configs") as pbar:
        for mode, report_df in reports:
            for _, cfg_row in report_df.iterrows():
                cfg = cfg_row.to_dict()
                thr = float(cfg["threshold"]) if mode == "pseudo" else 0.0

                test_scores, test_reps = train_and_score(cfg, mode)
                keeper = test_reps.keeper

                sel_weights = process_weighting(
                    all_weights,
                    layer=resolve_layer(cfg["weight"]),
                    temp=TEMP, sim=SIM,
                )

                # ── Classifier scores ────────────────────────────────────────
                base = compute_metrics(scores=test_scores, keeper=keeper,
                                       ks=[1], direction="desc")
                b_step, b_agent = base["step@1_desc"], base["agent@1_desc"]

                for gamma, k in product(GAMMAS, KS):
                    reweighted = reweight_scores(
                        test_scores, keeper, sel_weights, gamma=gamma, k=k,
                    )
                    m = compute_metrics(scores=reweighted, keeper=keeper,
                                        ks=[1], direction="desc")
                    assert (reweighted != test_scores).sum().item() > 0, \
                        "The weighted scores were not changed at all."
                    records.append({
                        "mode":               mode,
                        "threshold":          thr,
                        "weight":             cfg["weight"],
                        "pooling":            cfg["pooling"],
                        **{col: cfg[col] for col in comp_cols},
                        "centered":           bool(cfg["centered"]),
                        "seed":               int(cfg["seed"]),
                        "gamma":              gamma,
                        "k":                  k,
                        "baseline_step_acc":  b_step,
                        "baseline_agent_acc": b_agent,
                        "step_acc":           m["step@1_desc"],
                        "agent_acc":          m["agent@1_desc"],
                    })

                # ── SVD baseline: reweight direct-projection scores ───────────
                cached    = get_reps_and_svd(cfg["pooling"], int(cfg["seed"]))
                svd_vec   = select_svd_score_vector(
                    cached["svd"]["test_scores"], cfg, comp_cols,
                )
                svd_scores = orient_svd_scores(
                    torch.as_tensor(svd_vec, dtype=torch.float32, device=device)
                )
                svd_base = compute_metrics(scores=svd_scores, keeper=keeper,
                                           ks=[1], direction="desc")
                sb_step, sb_agent = svd_base["step@1_desc"], svd_base["agent@1_desc"]

                for gamma, k in product(GAMMAS, KS):
                    reweighted = reweight_scores(
                        svd_scores, keeper, sel_weights, gamma=gamma, k=k,
                    )
                    m = compute_metrics(scores=reweighted, keeper=keeper,
                                        ks=[1], direction="desc")
                    assert (reweighted != svd_scores).sum().item() > 0, \
                        "The weighted SVD scores were not changed at all."
                    records.append({
                        "mode":               "svd",
                        "threshold":          thr,
                        "weight":             cfg["weight"],
                        "pooling":            cfg["pooling"],
                        **{col: cfg[col] for col in comp_cols},
                        "centered":           bool(cfg["centered"]),
                        "seed":               int(cfg["seed"]),
                        "gamma":              gamma,
                        "k":                  k,
                        "baseline_step_acc":  sb_step,
                        "baseline_agent_acc": sb_agent,
                        "step_acc":           m["step@1_desc"],
                        "agent_acc":          m["agent@1_desc"],
                    })

                pbar.update(1)

    cols = [
        "mode", "threshold", "weight", "pooling", *comp_cols, "centered", "seed",
        "gamma", "k",
        "baseline_step_acc", "baseline_agent_acc", "step_acc", "agent_acc",
    ]
    pd.DataFrame(records, columns=cols).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(records)} rows to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Reweighting sweep over best classifier configs.")
    p.add_argument("--model",  required=True, choices=["llama-3.1-8b", "qwen3-8b"])
    p.add_argument("--subset", required=True, choices=["algorithm-generated", "hand-crafted"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_sweep(args.model, args.subset, torch.device(args.device))


if __name__ == "__main__":
    main()