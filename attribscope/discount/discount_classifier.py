"""Classifier-strategy reweighting sweep.

For each (model, subset), filter the undiscounted table to
strategy in {classifier_pseudo, classifier_oracle}, reproduce each row's
val + test classifier scores, then sweep (layer_range, gamma, w) applying
the discount pass and recording step_acc / agent_acc on val + test.

One output TSV per (model, subset) at
    outputs/tables/discounted/sweep/{model}__{subset}__classifier.tsv

python -m attribscope.discount.discount_classifier \
    --model qwen3-8b \
    --subset algorithm-generated \
    --device cuda
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from attribscope.discount.weights import aggregate_attn
from attribscope.discount.discount import apply_discount
from attribscope.discount.reproduce import reproduce_classifier
from attribscope.svd.utils import compute_metrics

GAMMAS = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 .. 1.0
WS     = [1, 2, 3, 4, 5, "all"]

MODELS  = ["deepseek-8b", "llama-3.1-8b", "qwen3-8b", "qwen3-14b"]
SUBSETS = ["algorithm-generated", "hand-crafted"]

UNDISC_ROOT_DEFAULT = Path("outputs/tables/undiscounted")
ATTN_ROOT_DEFAULT   = Path("outputs/weighting_attn")
REPS_ROOT_DEFAULT   = Path("outputs/representation-full")
DATA_ROOT_DEFAULT   = Path("data/ww")
OUT_ROOT_DEFAULT    = Path("outputs/tables/discounted/sweep")

CLF_STRATEGIES = ("classifier_pseudo", "classifier_oracle")


def _range_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _row_record(row, layer_range, gamma, w, val_m, test_m) -> dict:
    v_step, v_agent = list(val_m.values())
    t_step, t_agent = list(test_m.values())
    return {
        "strategy":  row["strategy"],
        "weight":    row["weight"],
        "pooling":   row["pooling"],
        "method":    row["method"],
        "c_begin":   row["c_begin"],
        "c_end":     row["c_end"],
        "centered":  row["centered"],
        "threshold": row["threshold"],
        "seed":      int(row["seed"]),
        "layer_range": layer_range,
        "gamma":     gamma,
        "w":         w,
        "undisc_step_acc_val":   row["step_acc_val"],
        "undisc_agent_acc_val":  row["agent_acc_val"],
        "undisc_step_acc_test":  row["step_acc_test"],
        "undisc_agent_acc_test": row["agent_acc_test"],
        "disc_step_acc_val":     v_step,
        "disc_agent_acc_val":    v_agent,
        "disc_step_acc_test":    t_step,
        "disc_agent_acc_test":   t_agent,
    }


def run_one_pair(
    model: str, subset: str,
    undisc_root: Path, attn_root: Path,
    reps_root: Path, data_root: Path,
    out_root: Path,
    n_ranges: int,
    device: str,
) -> None:
    table_path = undisc_root / f"{model}__{subset}.tsv"
    if not table_path.exists():
        print(f"[skip] missing undiscounted table: {table_path}")
        return
    rows = pd.read_csv(table_path, sep="\t")
    rows = rows[rows["strategy"].isin(CLF_STRATEGIES)].reset_index(drop=True)
    if len(rows) == 0:
        print(f"[skip] no classifier rows in {table_path}")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{model}__{subset}__classifier.tsv"
    if out_path.exists():
        print(f"discounted result already exist: {out_path}")
        return

    weightings, bounds = aggregate_attn(
        attn_root, model, subset, n_ranges=n_ranges, device="cpu",
    )
    range_labels = [_range_label(lo, hi) for (lo, hi) in bounds]

    records = []
    n_per_row = len(weightings) * len(GAMMAS) * len(WS)
    pbar = tqdm(total=len(rows) * n_per_row, desc=f"{model}/{subset} clf")
    for _, row in rows.iterrows():
        bundle = reproduce_classifier(row, model, subset, reps_root, data_root, device)
        v_s = bundle.val_scores.cpu()
        t_s = bundle.test_scores.cpu()

        for r_idx, weighting in enumerate(weightings):
            for gamma, w in product(GAMMAS, WS):
                v_d = apply_discount(v_s, bundle.val_keeper,  weighting, gamma=gamma, w=w)
                t_d = apply_discount(t_s, bundle.test_keeper, weighting, gamma=gamma, w=w)
                val_m  = compute_metrics(v_d, bundle.val_keeper,  ks=[1], direction="desc")
                test_m = compute_metrics(t_d, bundle.test_keeper, ks=[1], direction="desc")
                records.append(_row_record(
                    row, range_labels[r_idx], gamma, w, val_m, test_m,
                ))
                pbar.update(1)
    pbar.close()

    pd.DataFrame(records).to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path}  ({len(records)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",  choices=MODELS,  default=None,
                    help="If unset, run all models.")
    ap.add_argument("--subset", choices=SUBSETS, default=None,
                    help="If unset, run both subsets.")
    ap.add_argument("--undisc-root", type=Path, default=UNDISC_ROOT_DEFAULT)
    ap.add_argument("--attn-root",   type=Path, default=ATTN_ROOT_DEFAULT)
    ap.add_argument("--reps-root",   type=Path, default=REPS_ROOT_DEFAULT)
    ap.add_argument("--data-root",   type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--out-root",    type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--n-ranges",    type=int,  default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    targets = [
        (m, s)
        for m in (MODELS  if args.model  is None else [args.model])
        for s in (SUBSETS if args.subset is None else [args.subset])
    ]
    for model, subset in targets:
        run_one_pair(
            model=model, subset=subset,
            undisc_root=args.undisc_root, attn_root=args.attn_root,
            reps_root=args.reps_root,    data_root=args.data_root,
            out_root=args.out_root,      n_ranges=args.n_ranges,
            device=args.device,
        )


if __name__ == "__main__":
    main()