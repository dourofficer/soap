"""Stage 1 — one (model, pooling, seed): fit SVD on the source train split,
score every target's val/test with that single in-memory fit.

Per target (ds, subset) writes:
  results/svd/{model}/{ds}/{subset}/svd_pooling-{pooling}_seed-{seed}.tsv
      full config-grid metric table (val+test merged, direction=asc)
  results/base-scores/{model}/{ds}/{subset}/selected_pooling-{pooling}_seed-{seed}.pt
      selected configs (per sel_by convention) + raw per-step score tensors
      + exact split file lists — the frozen record the rescore stage consumes.

    python exp-synthetic-correct/run_svd.py \
        --model deepseek-8b --pooling mean --seed 1 [--config …] [--set k=v …]
"""
from __future__ import annotations

import argparse
import gc

import torch

import common
import xfit
from src.svd.reproduce import N_COMPONENTS
from src.utils.utils import load_representations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--model", required=True)
    p.add_argument("--pooling", required=True)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = common.load_cfg(args.config, args.overrides)
    model, pooling, seed = args.model, args.pooling, args.seed
    device = cfg["device"]
    positions = cfg["positions"]
    weight_names = "all" if positions == ["all"] else list(positions)

    targets = list(common.iter_targets(cfg))
    outputs = {
        (ds, sub): (common.svd_tsv(cfg, model, ds, sub, pooling, seed),
                    common.scores_pt(cfg, model, ds, sub, pooling, seed))
        for ds, sub in targets
    }
    if all(t.exists() and p.exists() for t, p in outputs.values()):
        print(f"[skipped] all outputs exist for {model}/{pooling}/seed-{seed}")
        return

    # ── Source: split + fit ──────────────────────────────────────────────────
    src_ds, src_subset = cfg["source"]["dataset"], cfg["source"]["subset"]
    src_roots = cfg["datasets"][src_ds]
    src_rep_dir = src_roots["reps_root"] / model / src_subset
    src_data_dir = src_roots["data_root"] / src_subset

    files = common.list_traj_files(src_rep_dir)
    train_files, src_val_files, src_test_files = common.split_source(
        files, cfg["splits"], seed)
    print(f"source {src_ds}/{src_subset}: train={len(train_files)} "
          f"val={len(src_val_files)} test={len(src_test_files)}")

    train_reps = load_representations(
        rep_dir=src_rep_dir, data_dir=src_data_dir, pooling=pooling,
        weight_names=weight_names, device="cpu", files=train_files)
    # Resolve 'all' once from the source so every target loads the exact same
    # weight-name set (score_target hard-asserts coverage).
    weight_names = sorted({s.name for s in train_reps.stores.values()})

    svd_components = xfit.fit_source_svd(train_reps, N_COMPONENTS, device=device)
    del train_reps
    gc.collect()

    # ── Targets: split + score + tabulate + select + persist ────────────────
    for ds, subset in targets:
        tsv_path, pt_path = outputs[(ds, subset)]
        if tsv_path.exists() and pt_path.exists():
            print(f"[skipped] {ds}/{subset} outputs exist")
            continue

        roots = cfg["datasets"][ds]
        rep_dir = roots["reps_root"] / model / subset
        data_dir = roots["data_root"] / subset
        t_files = common.list_traj_files(rep_dir)

        if ds == src_ds and subset == src_subset:
            val_files, test_files = src_val_files, src_test_files
        else:
            val_files, test_files = common.split_target(
                t_files, cfg["target_val_ratio"], seed)
        print(f"target {ds}/{subset}: val={len(val_files)} test={len(test_files)}")

        rep_kwargs = dict(rep_dir=rep_dir, data_dir=data_dir, pooling=pooling,
                          weight_names=weight_names, device="cpu")
        val_reps = load_representations(**rep_kwargs, files=val_files)
        test_reps = load_representations(**rep_kwargs, files=test_files)

        val_records = xfit.score_target(val_reps, svd_components, device=device)
        test_records = xfit.score_target(test_reps, svd_components, device=device)

        table = xfit.tabulate_val_test(val_records, val_reps.keeper,
                                       test_records, test_reps.keeper)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(tsv_path, sep="\t", index=False)
        print(f"wrote {tsv_path}  ({len(table)} rows)")

        selected = xfit.select_configs(table,
                                       weighted_flags=cfg["select"]["weighted"],
                                       by=cfg["select"]["by"],
                                       top_k=cfg["select"]["top_k"])
        xfit.reselect_check(table, selected, top_k=cfg["select"]["top_k"])
        rows = xfit.extract_selected_scores(selected, val_records, test_records)

        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "meta": {
                "model": model, "pooling": pooling, "seed": seed,
                "source": {"dataset": src_ds, "subset": src_subset,
                           "splits": dict(cfg["splits"])},
                "target": {"dataset": ds, "subset": subset,
                           "val_ratio": cfg["target_val_ratio"]},
                "n_components": N_COMPONENTS,
                "positions": weight_names,
                "score_convention": "raw SVD projection, lower = error",
            },
            "val_files": val_files,
            "test_files": test_files,
            "rows": rows,
        }, pt_path)
        print(f"wrote {pt_path}  ({len(rows)} selected rows)")

        del val_reps, test_reps, val_records, test_records
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
