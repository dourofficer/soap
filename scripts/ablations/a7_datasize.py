"""A7 / fig:datasize — data efficiency of the reference fit.

Val and test splits stay fixed (frozen triples). Per seed, the train split is
subsampled to {1/3, 2/3, 1} of its trajectories (10/20/30% of the corpus) by a
seeded shuffle-and-take-prefix, so the fractions are nested. For EACH fraction the
full config is re-selected from scratch — dense base grid (position x band), then
the backprop rescore grid (layer_range x gamma x w) on the winning base config,
test-selected over the triple by the standard rule. Each fraction is therefore "the
best SOAP can do with that much reference data", matching the optimistic protocol
everywhere else. The fraction-1 cell must reproduce Table 1's base and SOAP rows.

Shardable: --configs / --models narrow the cells, so cells can run in parallel and
the partial TSVs are merged afterwards (--out).

    python scripts/ablations/a7_datasize.py --configs configs-main/ww.yaml \
        --models qwen3.5-9b --out results-ablations/a7_parts/ww-qwen.tsv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIGS_NOGT, POOLING, RESULTS_DIR, anchor_rows,  # noqa: E402
                    assert_close, cell_paths, iter_cells, load_selection,
                    select_config)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import (ENSEMBLE_POSITION, band_bounds, base_positions,  # noqa: E402
                        ens_score_steps, fit_svd, member_positions, score_steps)
from main.stores import load_representations, split_files             # noqa: E402

FRACTIONS = [("1/3", 1 / 3), ("2/3", 2 / 3), ("1", 1.0)]
OUT = RESULTS_DIR / "a7_datasize.tsv"


def subsample(train_files: list[str], frac: float, seed: int) -> list[str]:
    """Seeded shuffle, take the first round(n*frac) — nested across fractions."""
    files = list(train_files)
    random.Random(seed).shuffle(files)
    k = max(1, int(round(len(files) * frac)))
    return files[:k]


def load_split(rep_dir, data_dir, files, device, names="all"):
    return load_representations(rep_dir, data_dir, poolings=[POOLING],
                                weight_names=names, files=files, device=device)


def base_grid_rows(cfg, model, subset, seeds, rep_dir, data_dir, files, frac,
                   device) -> tuple[list[dict], dict[int, int]]:
    """The dense (position x band) grid on the subsampled train split, per seed."""
    n_comp = cfg["n_components"]
    bands = band_bounds(n_comp)
    rows, n_train = [], {}
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        sub = subsample(parts["train"], frac, seed)
        n_train[seed] = len(sub)
        train = load_split(rep_dir, data_dir, sub, device)
        val = load_split(rep_dir, data_dir, parts["val"], device)
        test = load_split(rep_dir, data_dir, parts["test"], device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        available = train.positions()
        positions = base_positions(available, cfg.get("positions", "all"),
                                   cfg.get("ensemble", True))
        members = member_positions(available)
        fits: dict[str, torch.Tensor] = {}
        for position in tqdm(positions, desc=f"base {model}/{subset} f={frac:.2f} s{seed}",
                             leave=False):
            if position == ENSEMBLE_POSITION:
                for p in members:
                    fits.setdefault(p, fit_svd(train.stores[(POOLING, p)].R, n_comp))
                tr = {p: train.stores[(POOLING, p)].R for p in members}
                vR = {p: val.stores[(POOLING, p)].R for p in members}
                tR = {p: test.stores[(POOLING, p)].R for p in members}
                vs = [ens_score_steps(cb, ce, members, fits, tr, vR) for cb, ce in bands]
                ts = [ens_score_steps(cb, ce, members, fits, tr, tR) for cb, ce in bands]
            else:
                V = fits.setdefault(position,
                                    fit_svd(train.stores[(POOLING, position)].R, n_comp))
                vs = [score_steps(val.stores[(POOLING, position)].R, V, cb, ce)
                      for cb, ce in bands]
                ts = [score_steps(test.stores[(POOLING, position)].R, V, cb, ce)
                      for cb, ce in bands]
            vm = compute_metrics_batch(torch.stack(vs), None, [1], ctx=val_ctx)
            tm = compute_metrics_batch(torch.stack(ts), None, [1], ctx=test_ctx)
            for i, (cb, ce) in enumerate(bands):
                rows.append({"seed": seed, "position": position,
                             "c_begin": cb, "c_end": ce,
                             "step_acc_test@1": float(tm["step@1"][i]),
                             "agent_acc_test@1": float(tm["agent@1"][i]),
                             "step_acc_val@1": float(vm["step@1"][i]),
                             "agent_acc_val@1": float(vm["agent@1"][i])})
        del train, val, test
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows, n_train


def rescore_grid_rows(cfg, model, subset, seeds, base_cfg, rep_dir, data_dir, files,
                      frac, weightings, labels, device) -> list[dict]:
    """The backprop rescore grid on the selected base config, per seed."""
    n_comp = cfg["n_components"]
    gammas = list(cfg["gammas"])
    ws = list(cfg["ws"])
    position = base_cfg["position"]
    cb, ce = int(base_cfg["c_begin"]), int(base_cfg["c_end"])
    rows = []
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        sub = subsample(parts["train"], frac, seed)
        if position == ENSEMBLE_POSITION:
            names = member_positions(rep_names_of(rep_dir, files))
        else:
            names = [position]
        train = load_split(rep_dir, data_dir, sub, device, names)
        val = load_split(rep_dir, data_dir, parts["val"], device, names)
        test = load_split(rep_dir, data_dir, parts["test"], device, names)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        if position == ENSEMBLE_POSITION:
            members = names
            fits = {p: fit_svd(train.stores[(POOLING, p)].R, n_comp) for p in members}
            tr = {p: train.stores[(POOLING, p)].R for p in members}
            s_val = ens_score_steps(cb, ce, members, fits, tr,
                                    {p: val.stores[(POOLING, p)].R for p in members})
            s_test = ens_score_steps(cb, ce, members, fits, tr,
                                     {p: test.stores[(POOLING, p)].R for p in members})
        else:
            V = fit_svd(train.stores[(POOLING, position)].R, n_comp)
            s_val = score_steps(val.stores[(POOLING, position)].R, V, cb, ce)
            s_test = score_steps(test.stores[(POOLING, position)].R, V, cb, ce)

        for r_idx, label in enumerate(labels):
            for w in ws:
                vmats = {"backprop": build_W(val.keeper, weightings[r_idx], w, device)}
                tmats = {"backprop": build_W(test.keeper, weightings[r_idx], w, device)}
                Sv = apply_strategy(s_val, val.keeper, vmats, "backprop", gammas).T.contiguous()
                St = apply_strategy(s_test, test.keeper, tmats, "backprop", gammas).T.contiguous()
                vm = compute_metrics_batch(Sv, None, [1], ctx=val_ctx)
                tm = compute_metrics_batch(St, None, [1], ctx=test_ctx)
                for gi, gamma in enumerate(gammas):
                    rows.append({"seed": seed, "layer_range": label, "gamma": gamma,
                                 "w": str(w),
                                 "step_acc_test@1": float(tm["step@1"][gi]),
                                 "agent_acc_test@1": float(tm["agent@1"][gi]),
                                 "step_acc_val@1": float(vm["step@1"][gi]),
                                 "agent_acc_val@1": float(vm["agent@1"][gi])})
        del train, val, test
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def rep_names_of(rep_dir, files):
    from main.stores import rep_names
    return rep_names(Path(rep_dir) / files[0])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=CONFIGS_NOGT)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    out_rows = []
    for cfg, model, subset in iter_cells(args.configs):
        if args.models and model not in args.models:
            continue
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                            n_ranges=cfg["n_ranges"], device=args.device)
        labels = [f"{lo}-{hi}" for lo, hi in bounds]

        for fname, frac in FRACTIONS:
            base_rows, n_train = base_grid_rows(cfg, model, subset, seeds, rep_dir,
                                                data_dir, files, frac, args.device)
            base_df = pd.DataFrame(base_rows)
            base = select_config(base_df, ["position", "c_begin", "c_end"], seeds,
                                 "step_acc_test@1", "agent_acc_test@1")
            assert base is not None
            resc_rows = rescore_grid_rows(cfg, model, subset, seeds, base["config"],
                                          rep_dir, data_dir, files, frac,
                                          weightings, labels, args.device)
            soap = select_config(pd.DataFrame(resc_rows),
                                 ["layer_range", "gamma", "w"], seeds,
                                 "step_acc_test@1", "agent_acc_test@1")
            assert soap is not None
            print(f"[{cfg['dataset']}] {model}/{subset} f={fname}: "
                  f"n_train={sorted(n_train.values())} base={base['config']} "
                  f"{base['step']:.4f} soap={soap['config']} {soap['step']:.4f}")

            if fname == "1":
                assert_close(base["step"], float(svd_row["step_acc_test"]),
                             f"{model}/{subset} fraction-1 base vs Table 1")
                assert_close(soap["step"], float(bp_row["step_acc_test"]),
                             f"{model}/{subset} fraction-1 soap vs Table 1")

            common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                      "seeds": ",".join(map(str, seeds)), "fraction": fname,
                      "n_train": ",".join(str(n_train[s]) for s in seeds),
                      **{k: base["config"][k] for k in ("position", "c_begin", "c_end")}}
            out_rows.append({**common, "row": "base", "layer_range": "", "gamma": 0.0,
                             "w": "", "step_acc_test": base["step"],
                             "agent_acc_test": base["agent"],
                             "step_acc_val": base["step_val"],
                             "agent_acc_val": base["agent_val"]})
            out_rows.append({**common, "row": "soap", **soap["config"],
                             "step_acc_test": soap["step"], "agent_acc_test": soap["agent"],
                             "step_acc_val": soap["step_val"],
                             "agent_acc_val": soap["agent_val"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
