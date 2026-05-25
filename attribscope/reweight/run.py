"""
sweep_reweighting.py — dependency-aware reweighting sweep over best classifier configs.

For each best (pooling, seed) row in the oracle/pseudo classifier-results TSVs,
retrain the classifier with the recorded layer/threshold, then sweep reweighting
across (gamma, k) and save all metrics to a single TSV.

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

from attribscope.svd2.utils import (
    load_representations, _resolve_dir, split_data,
)
from attribscope.classifier.run_all_positions import (
    prepare_data, precompute_svd, run_one,
)
from attribscope.classifier.classifier import (
    seed_everything, MLPClassifier, infer,
)
from attribscope.reweight.reweight import (
    load_weighting, process_weighting, resolve_layer,
    reweight_scores, compute_metrics,
)

# ── Sweep hyperparameters (fixed) ────────────────────────────────────────────
GAMMAS       = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 … 1.0
KS           = [1, 2, 3, 5, 100]                            # 100 ≈ full context
TEMP         = 1.0
SIM          = "raw_dot"
N_COMPONENTS = 20
EPOCHS       = 500
LR           = 0.02
HIDDEN_DIM   = 1024

# ── Paths ────────────────────────────────────────────────────────────────────
REPS_ROOT    = Path("/data/hoang/attrib/outputs")
DATA_ROOT    = Path("data/ww")
RESULTS_ROOT = Path("/data/hoang/attrib/results_svd")
REP_TYPE     = "hidden"
WEIGHT_NAMES = "all"


def load_best_configs(model: str, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read classifier-results TSVs and pick best (pooling, seed) row per mode."""
    base   = RESULTS_ROOT / REP_TYPE / model / "metrics" / subset
    frames = []
    for path in sorted(base.glob("classifer_*.tsv")):
        parts   = path.stem.split("_")
        pooling = parts[1][len("pooling-"):]
        seed    = int(parts[2][len("seed-"):])
        frames.append(
            pd.read_csv(path, sep="\t").assign(pooling=pooling, seed=seed)
        )
    clf_df = pd.concat(frames, ignore_index=True)

    cols = ["threshold", "weight", "pooling", "c", "centered", "seed",
            "test_step_acc"]

    reports = {}
    for label in ["oracle", "pseudo"]:
        sub = clf_df.query("labels == @label")[cols]
        idx = sub.groupby(["pooling", "seed"])["test_step_acc"].idxmax()
        reports[label] = sub.loc[idx].reset_index(drop=True)
    return reports["oracle"], reports["pseudo"]


def run_sweep(model: str, subset: str, device: torch.device) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    oracle_report, pseudo_report = load_best_configs(model, subset)

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
    out_path = out_dir / "reweight_sweep.tsv"

    # ── Shared weighting, loaded once ────────────────────────────────────────
    all_weights = load_weighting(model, subset, device=device)

    # ── Cache reps + SVD per (pooling, seed) ─────────────────────────────────
    rep_cache: dict[tuple[str, int], dict] = {}

    def get_reps_and_svd(pooling: str, seed: int) -> dict:
        key = (pooling, seed)
        if key in rep_cache:
            return rep_cache[key]

        tr_files, te_files = split_data(files, 0.5, seed)
        tr_files, va_files = split_data(tr_files, 0.8, seed)

        rk = dict(rep_dir=rep_dir, data_dir=data_dir,
                  pooling=pooling, weight_names=WEIGHT_NAMES, device=device)
        train_reps = load_representations(**rk, files=tr_files)
        val_reps   = load_representations(**rk, files=va_files)
        test_reps  = load_representations(**rk, files=te_files)

        svd_pre = precompute_svd(
            train_reps, val_reps, test_reps,
            n_components=N_COMPONENTS, device=device,
        )
        rep_cache[key] = {
            "train_reps": train_reps, "val_reps": val_reps,
            "test_reps":  test_reps,  "svd":      svd_pre,
        }
        return rep_cache[key]

    def train_and_score(cfg: dict, mode: str):
        pooling   = cfg["pooling"]
        seed      = int(cfg["seed"])
        layer     = cfg["weight"]
        threshold = float(cfg["threshold"]) if mode == "pseudo" else 0.0

        cached   = get_reps_and_svd(pooling, seed)
        prepared = prepare_data(
            cached["train_reps"], cached["val_reps"], cached["test_reps"],
            cached["svd"]["train_scores"], cached["svd"]["val_scores"],
            layer_idx=layer, threshold=threshold, mode=mode, device=device,
        )

        seed_everything(seed)
        clf = MLPClassifier(input_dim=prepared["train"][0].shape[1],
                            hidden_dim=HIDDEN_DIM)
        clf, _ = run_one(
            clf, prepared["train_loader"], prepared["val_loader"],
            cached["val_reps"], cached["test_reps"],
            layer_idx=layer, threshold=threshold,
            epochs=EPOCHS, learning_rate=LR, device=device,
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

                test_scores, test_reps = train_and_score(cfg, mode)
                keeper = test_reps.keeper

                base = compute_metrics(scores=test_scores, keeper=keeper,
                                       ks=[1], direction="desc")
                b_step, b_agent = base["step@1_desc"], base["agent@1_desc"]

                sel_weights = process_weighting(
                    all_weights,
                    layer=resolve_layer(cfg["weight"]),
                    temp=TEMP, sim=SIM,
                )

                for gamma, k in product(GAMMAS, KS):
                    reweighted = reweight_scores(
                        test_scores, keeper, sel_weights, gamma=gamma, k=k,
                    )
                    m = compute_metrics(scores=reweighted, keeper=keeper,
                                        ks=[1], direction="desc")
                    records.append({
                        "mode":               mode,
                        "threshold":          float(cfg["threshold"]) if mode == "pseudo" else 0.0,
                        "weight":             cfg["weight"],
                        "pooling":            cfg["pooling"],
                        "c":                  int(cfg["c"]),
                        "centered":           bool(cfg["centered"]),
                        "seed":               int(cfg["seed"]),
                        "gamma":              gamma,
                        "k":                  k,
                        "baseline_step_acc":  b_step,
                        "baseline_agent_acc": b_agent,
                        "step_acc":           m["step@1_desc"],
                        "agent_acc":          m["agent@1_desc"],
                    })
                pbar.update(1)

    cols = [
        "mode", "threshold", "weight", "pooling", "c", "centered", "seed",
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