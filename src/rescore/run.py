"""CRR rescoring runner for one (model, subset).

Filters the undiscounted table to strategy=='svd', reproduces each row's val+test
SVD-projection scores, then sweeps (layer_range, gamma, w, orient) applying the
single-pass discount and recording step/agent accuracy on val + test. Writes one
TSV at {out_root}/{model}/{subset}/svd.tsv.

This is the runner the rescore sweep shells out to (was the in-process body of
``experiments/rescore/sweep.py``). One process per (model, subset) keeps the
reproduce cache bounded to a single table's worth.

    python -m src.rescore.run \
        --model qwen3.5-9b --subset gaia \
        --undisc-root outputs-correct-error/undiscounted-splits/325 \
        --attn-root   outputs-correct-error/attention \
        --reps-root   outputs-correct-error/activations \
        --data-root   data/correct-error \
        --out-root    outputs-correct-error/discounted-splits/sweep/325 \
        --device cuda --n-ranges 4 \
        --gammas 0.1 0.2 0.3 --ws 1 2 3 all --orients negate inverse sigmoid \
        --train-split 0.3 --val-split 0.2 --test-split 0.5
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.rescore.weights import aggregate_attn
from src.rescore.discount import orient_svd_scores, apply_discount
from src.svd.reproduce import reproduce_svd, clear_cache
from src.utils.utils import compute_metrics


def _range_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _parse_w(x: str):
    """Sweep values for w are ints or the literal 'all'."""
    return x if x == "all" else int(x)


def _row_record(row, layer_range, gamma, w, orient, val_m, test_m) -> dict:
    v_step, v_agent = list(val_m.values())
    t_step, t_agent = list(test_m.values())
    return {
        "strategy":    row["strategy"],
        "position":    row["position"],
        "pooling":     row["pooling"],
        "method":      row["method"],
        "c_begin":     row["c_begin"],
        "c_end":       row["c_end"],
        "centered":    row["centered"],
        "weighted":    row["weighted"],
        "threshold":   row.get("threshold", ""),
        "seed":        int(row["seed"]),
        "layer_range": layer_range,
        "gamma":       gamma,
        "w":           w,
        "orient":      orient,
        "undisc_step_acc_val":   row["step_acc_val"],
        "undisc_agent_acc_val":  row["agent_acc_val"],
        "undisc_step_acc_test":  row["step_acc_test"],
        "undisc_agent_acc_test": row["agent_acc_test"],
        "disc_step_acc_val":     v_step,
        "disc_agent_acc_val":    v_agent,
        "disc_step_acc_test":    t_step,
        "disc_agent_acc_test":   t_agent,
    }


def run_one_pair(args) -> None:
    undisc_root = Path(args.undisc_root)
    attn_root   = Path(args.attn_root)
    reps_root   = Path(args.reps_root)
    data_root   = Path(args.data_root)
    out_root    = Path(args.out_root)

    table_path = undisc_root / args.model / args.subset / args.undisc_file
    print(f"Table Path: {table_path}")
    if not table_path.exists():
        print(f"[skip] missing undiscounted table: {table_path}")
        return
    rows = pd.read_csv(table_path, sep="\t")
    rows = rows[rows["strategy"] == "svd"].reset_index(drop=True)
    if len(rows) == 0:
        print(f"[skip] no svd rows in {table_path}")
        return

    out_path = out_root / args.model / args.subset / "svd.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] already exists: {out_path}")
        return

    weightings, bounds = aggregate_attn(
        attn_root, args.model, args.subset, n_ranges=args.n_ranges, device="cpu",
    )
    range_labels = [_range_label(lo, hi) for (lo, hi) in bounds]

    ws = [_parse_w(w) for w in args.ws]

    records = []
    n_per_row = len(args.orients) * len(weightings) * len(args.gammas) * len(ws)
    pbar = tqdm(total=len(rows) * n_per_row, desc=f"{args.model}/{args.subset} svd")
    for _, row in rows.iterrows():
        bundle = reproduce_svd(
            row, args.model, args.subset, reps_root, data_root, args.device,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
        )
        for orient in args.orients:
            v_o = orient_svd_scores(bundle.val_scores,  strategy=orient).cpu()
            t_o = orient_svd_scores(bundle.test_scores, strategy=orient).cpu()
            for r_idx, weighting in enumerate(weightings):
                for gamma, w in product(args.gammas, ws):
                    v_d = apply_discount(v_o, bundle.val_keeper,  weighting, gamma=gamma, w=w)
                    t_d = apply_discount(t_o, bundle.test_keeper, weighting, gamma=gamma, w=w)
                    val_m  = compute_metrics(v_d, bundle.val_keeper,  ks=[1], direction="desc")
                    test_m = compute_metrics(t_d, bundle.test_keeper, ks=[1], direction="desc")
                    records.append(_row_record(
                        row, range_labels[r_idx], gamma, w, orient, val_m, test_m,
                    ))
                    pbar.update(1)
    pbar.close()

    pd.DataFrame(records).to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path}  ({len(records)} rows)")

    clear_cache()
    if args.device == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--undisc-root", required=True)
    p.add_argument("--attn-root",   required=True)
    p.add_argument("--reps-root",   required=True)
    p.add_argument("--data-root",   required=True)
    p.add_argument("--out-root",    required=True)
    p.add_argument("--undisc-file", default="weighted_false.tsv")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--n-ranges", type=int, default=4)
    p.add_argument("--gammas",  nargs="+", type=float, required=True)
    p.add_argument("--ws",      nargs="+", type=str,   required=True)
    p.add_argument("--orients", nargs="+", type=str,   required=True)
    p.add_argument("--train-split", type=float, required=True)
    p.add_argument("--val-split",   type=float, required=True)
    p.add_argument("--test-split",  type=float, required=True)
    return p.parse_args()


def main() -> None:
    run_one_pair(parse_args())


if __name__ == "__main__":
    main()
