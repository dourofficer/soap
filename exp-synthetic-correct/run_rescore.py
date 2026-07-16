"""Stage 3 — one (model, target-dataset, subset): CRR sweep over the persisted
cross-dataset base scores.

Structural fork of src/rescore/run.py with `reproduce_svd` (same-dataset refit)
replaced by the .pt bundles stage 1 persisted. Discount math, attention
aggregation and metrics are imported unchanged. Identical configs selected by
both conventions are computed once and emitted once per sel_by.

Writes results/rescore/sweep/{model}/{ds}/{subset}/svd.tsv.

    python exp-synthetic-correct/run_rescore.py \
        --model deepseek-8b --dataset ww --subset hand-crafted [--config …]
"""
from __future__ import annotations

import argparse
from itertools import product

import pandas as pd
import torch
from tqdm.auto import tqdm

import common
from src.rescore.discount import apply_discount, orient_svd_scores
from src.rescore.weights import aggregate_attn
from src.utils.utils import compute_metrics, load_representations


def _range_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _parse_w(x):
    return x if x == "all" else int(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--subset", required=True)
    return p.parse_args()


def _load_bundles(cfg, model, ds, subset) -> list[dict]:
    pt_dir = common.scores_dir(cfg, model, ds, subset)
    paths = sorted(pt_dir.glob("selected_pooling-*_seed-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no persisted base scores under {pt_dir} — "
                                "run stage svd first")
    return [torch.load(p, weights_only=False) for p in paths]


def _keeper_for(roots, model, subset, pooling, files):
    """Rebuild a keeper for a split cheaply: load a single weight (embed) —
    keeper construction only needs the step set + JSON sidecar + header
    metadata, which are identical regardless of weight_names."""
    reps = load_representations(
        rep_dir=roots["reps_root"] / model / subset,
        data_dir=roots["data_root"] / subset,
        pooling=pooling, weight_names=["embed"], device="cpu", files=files)
    return reps.keeper


def run_one(cfg, model, ds, subset) -> None:
    out_path = common.sweep_dir(cfg, model, ds, subset) / "svd.tsv"
    if out_path.exists():
        print(f"[skipped] already exists: {out_path}")
        return

    bundles = _load_bundles(cfg, model, ds, subset)
    roots = cfg["datasets"][ds]
    rc = cfg["rescore"]
    gammas = rc["gammas"]
    ws = [_parse_w(w) for w in rc["ws"]]
    orients = rc["orients"]

    weightings, bounds = aggregate_attn(
        roots["attn_root"], model, subset, n_ranges=rc["n_ranges"], device="cpu")
    range_labels = [_range_label(lo, hi) for (lo, hi) in bounds]

    # Keepers per (pooling, seed) split — cached on the file-list identity.
    keeper_cache: dict[tuple, tuple] = {}

    records = []
    for bundle in bundles:
        meta = bundle["meta"]
        pooling, seed = meta["pooling"], meta["seed"]
        ck = (pooling, tuple(bundle["val_files"]), tuple(bundle["test_files"]))
        if ck not in keeper_cache:
            keeper_cache[ck] = (
                _keeper_for(roots, model, subset, pooling, bundle["val_files"]),
                _keeper_for(roots, model, subset, pooling, bundle["test_files"]),
            )
        val_keeper, test_keeper = keeper_cache[ck]

        # Dedupe: identical config picked by several sel_by conventions is
        # swept once and emitted once per convention.
        by_config: dict[tuple, dict] = {}
        for row in bundle["rows"]:
            key = tuple(row[k] for k in common.CONFIG_KEYS)
            by_config.setdefault(key, {"row": row, "sel_bys": []})
            by_config[key]["sel_bys"].append(row["sel_by"])

        n_combos = len(orients) * len(weightings) * len(gammas) * len(ws)
        pbar = tqdm(total=len(by_config) * n_combos,
                    desc=f"{model}/{ds}/{subset} {pooling}/seed-{seed}")
        for entry in by_config.values():
            row, sel_bys = entry["row"], entry["sel_bys"]
            for orient in orients:
                v_o = orient_svd_scores(row["val_scores"], strategy=orient).cpu()
                t_o = orient_svd_scores(row["test_scores"], strategy=orient).cpu()
                for r_idx, weighting in enumerate(weightings):
                    for gamma, w in product(gammas, ws):
                        v_d = apply_discount(v_o, val_keeper, weighting,
                                             gamma=gamma, w=w)
                        t_d = apply_discount(t_o, test_keeper, weighting,
                                             gamma=gamma, w=w)
                        val_m = compute_metrics(v_d, val_keeper,
                                                ks=[1], direction="desc")
                        test_m = compute_metrics(t_d, test_keeper,
                                                 ks=[1], direction="desc")
                        v_step, v_agent = list(val_m.values())
                        t_step, t_agent = list(test_m.values())
                        for sel_by in sel_bys:
                            records.append({
                                "strategy": "svd",
                                **{k: row[k] for k in common.CONFIG_KEYS},
                                "threshold": "",
                                "seed": seed,
                                "sel_by": sel_by,
                                "source": f"{meta['source']['dataset']}/"
                                          f"{meta['source']['subset']}",
                                "layer_range": range_labels[r_idx],
                                "gamma": gamma,
                                "w": w,
                                "orient": orient,
                                "undisc_step_acc_val": row["step_acc_val"],
                                "undisc_agent_acc_val": row["agent_acc_val"],
                                "undisc_step_acc_test": row["step_acc_test"],
                                "undisc_agent_acc_test": row["agent_acc_test"],
                                "disc_step_acc_val": v_step,
                                "disc_agent_acc_val": v_agent,
                                "disc_step_acc_test": t_step,
                                "disc_agent_acc_test": t_agent,
                            })
                        pbar.update(1)
        pbar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path}  ({len(records)} rows)")


def main() -> None:
    args = parse_args()
    cfg = common.load_cfg(args.config, args.overrides)
    run_one(cfg, args.model, args.dataset, args.subset)


if __name__ == "__main__":
    main()
