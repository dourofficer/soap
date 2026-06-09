"""SVD-strategy reweighting sweep.

For each (model, subset), filter the undiscounted table to strategy=='svd',
reproduce each row's val + test SVD-projection scores, then sweep
    (layer_range, gamma, w, orient)
applying the discount pass and recording step_acc / agent_acc on val + test.

One output TSV per (model, subset) at
    {out_root}/{model}__{subset}__svd.tsv

python -m src.experiments.rescore.sweep \
    --config src/experiments/rescore/configs/default.yaml
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.rescore.weights import aggregate_attn
from src.rescore.discount import orient_svd_scores, apply_discount
from src.svd.reproduce import reproduce_svd
from src.utils.utils import compute_metrics


def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def _range_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _row_record(row, layer_range, gamma, w, orient, val_m, test_m) -> dict:
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
        "threshold": row.get("threshold", ""),
        "seed":      int(row["seed"]),
        "layer_range": layer_range,
        "gamma":     gamma,
        "w":         w,
        "orient":    orient,
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
    model: str, subset: str, cfg: dict,
) -> None:
    undisc_root = Path(cfg["undisc_root"])
    attn_root   = Path(cfg["attn_root"])
    reps_root   = Path(cfg["reps_root"])
    data_root   = Path(cfg["data_root"])
    out_root    = Path(cfg["out_root"])
    device      = cfg["device"]
    n_ranges    = cfg["n_ranges"]
    gammas      = cfg["gammas"]
    ws          = cfg["ws"]
    orients     = cfg["orients"]

    table_path = undisc_root / f"{model}/{subset}.tsv"
    if not table_path.exists():
        print(f"[skip] missing undiscounted table: {table_path}")
        return
    rows = pd.read_csv(table_path, sep="\t")
    rows = rows[rows["strategy"] == "svd"].reset_index(drop=True)
    if len(rows) == 0:
        print(f"[skip] no svd rows in {table_path}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{model}/{subset}.tsv"
    if out_path.exists():
        print(f"[skip] already exists: {out_path}")
        return

    weightings, bounds = aggregate_attn(
        attn_root, model, subset, n_ranges=n_ranges, device="cpu",
    )
    range_labels = [_range_label(lo, hi) for (lo, hi) in bounds]

    records = []
    n_per_row = len(orients) * len(weightings) * len(gammas) * len(ws)
    pbar = tqdm(total=len(rows) * n_per_row, desc=f"{model}/{subset} svd")
    for _, row in rows.iterrows():
        bundle = reproduce_svd(row, model, subset, reps_root, data_root, device)

        for orient in orients:
            v_o = orient_svd_scores(bundle.val_scores,  strategy=orient).cpu()
            t_o = orient_svd_scores(bundle.test_scores, strategy=orient).cpu()

            for r_idx, weighting in enumerate(weightings):
                for gamma, w in product(gammas, ws):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            run_one_pair(model, subset, cfg)


if __name__ == "__main__":
    main()