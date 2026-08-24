"""E1 / tab:transfer — cross-distribution transfer with per-pair re-selection.

A 4x4 source->target grid over {WW-AG, WW-HC, TE-Cap, TE-Mag}, one grid per backbone.
For each pair the reference R is fit on the SOURCE's train split and evaluated on the
TARGET, whose trajectories are re-partitioned for the cross setting:

    val   = the target's train + val files of the main experiment (its train split
            is unused for fitting here, so its whole non-test half validates)
    test  = the target's test split of the main experiment, unchanged

Dependency weights always come from the target trajectories' own attention. Seeds
pair positionally: source seed i's train split with target seed i's val/test splits.

The full configuration is RE-SELECTED per pair — dense base grid (position x band),
then the backprop rescore grid on the winning base config — under TWO conventions:

    test  select on mean TARGET-test step accuracy (optimistic, as in Table 1)
    val   select on mean TARGET-val step accuracy, report test

Both conventions are recorded for every pair, diagonal included. Under the test
convention the diagonal repeats the main experiment's selection problem exactly, so
those cells must reproduce the selection table — asserted on every run. (The
reported tables still show the main-experiment numbers on every diagonal, whichever
convention the off-diagonal cells use.)

Shardable: --models / --sources narrow the pairs; merge the partial TSVs afterwards.

    python scripts/ablations/e1_transfer.py --models qwen3.5-9b \
        --sources WW-AG WW-HC --out results-ablations/e1_parts/qwen-ww.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    cell_paths, iter_cells, load_selection, select_config)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import (ENSEMBLE_POSITION, band_bounds, base_positions,  # noqa: E402
                        ens_score_steps, fit_svd, member_positions, score_steps)
from main.stores import load_representations, split_files             # noqa: E402
from main.sweep import BASE_SWEPT, RESCORE_SWEPT                      # noqa: E402

OUT = RESULTS_DIR / "e1_transfer.tsv"
SHORT = {("ww", "algorithm-generated"): "WW-AG", ("ww", "hand-crafted"): "WW-HC",
         ("traceelephant", "captain"): "TE-Cap", ("traceelephant", "magentic"): "TE-Mag"}
METRIC_COLS = ["step_acc_val@1", "agent_acc_val@1", "step_acc_test@1", "agent_acc_test@1"]


def collect_cells(model: str) -> list[dict]:
    cells = []
    for cfg, m, subset in iter_cells():
        if m != model:
            continue
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        cells.append({"cfg": cfg, "subset": subset,
                      "name": SHORT[(cfg["dataset"], subset)],
                      "svd": svd_row, "bp": bp_row, "rep_dir": rep_dir,
                      "data_dir": data_dir, "files": files,
                      "seeds": C.seeds_for(cfg, subset)})
    return cells


def load_split(cell, files, device, names="all"):
    return load_representations(cell["rep_dir"], cell["data_dir"], poolings=[POOLING],
                                weight_names=names, files=files, device=device)


def pair_splits(src, tgt, i):
    """(source train files, target val files, target test files) for pair index i."""
    sparts = split_files(src["files"], src["cfg"]["splits"], src["seeds"][i])
    tparts = split_files(tgt["files"], tgt["cfg"]["splits"], tgt["seeds"][i])
    return sparts["train"], tparts["train"] + tparts["val"], tparts["test"]


def _metrics_row(vm, tm, j) -> dict:
    return {"step_acc_val@1": float(vm["step@1"][j]), "agent_acc_val@1": float(vm["agent@1"][j]),
            "step_acc_test@1": float(tm["step@1"][j]), "agent_acc_test@1": float(tm["agent@1"][j])}


def base_grid(src, tgt, model, device) -> pd.DataFrame:
    """Dense (position x band) grid: R fit on source train, scored on target val/test.
    The `seed` column is the PAIR index (0..2), pairing the two triples positionally."""
    cfg = src["cfg"]
    n_comp = cfg["n_components"]
    bands = band_bounds(n_comp)
    rows = []
    for i in range(len(src["seeds"])):
        f_train, f_val, f_test = pair_splits(src, tgt, i)
        train = load_split(src, f_train, device)
        val = load_split(tgt, f_val, device)
        test = load_split(tgt, f_test, device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        available = train.positions()
        assert available == val.positions(), "source/target position sets differ"
        positions = base_positions(available, cfg.get("positions", "all"),
                                   cfg.get("ensemble", True))
        members = member_positions(available)
        fits: dict[str, torch.Tensor] = {}
        for position in tqdm(positions, desc=f"{src['name']}->{tgt['name']} i{i}",
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
            for j, (cb, ce) in enumerate(bands):
                rows.append({"seed": i, "position": position, "c_begin": cb, "c_end": ce,
                             **_metrics_row(vm, tm, j)})
        del train, val, test
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def rescore_grid(src, tgt, base_cfg, weightings, labels, device) -> pd.DataFrame:
    """Backprop rescore grid (layer_range x gamma x w) on one base config."""
    cfg = src["cfg"]
    n_comp = cfg["n_components"]
    gammas = list(cfg["gammas"])
    ws = list(cfg["ws"])
    position = base_cfg["position"]
    cb, ce = int(base_cfg["c_begin"]), int(base_cfg["c_end"])
    rows = []
    for i in range(len(src["seeds"])):
        f_train, f_val, f_test = pair_splits(src, tgt, i)
        if position == ENSEMBLE_POSITION:
            from main.stores import rep_names
            names = member_positions(rep_names(src["rep_dir"] / src["files"][0]))
        else:
            names = [position]
        train = load_split(src, f_train, device, names)
        val = load_split(tgt, f_val, device, names)
        test = load_split(tgt, f_test, device, names)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        if position == ENSEMBLE_POSITION:
            fits = {p: fit_svd(train.stores[(POOLING, p)].R, n_comp) for p in names}
            tr = {p: train.stores[(POOLING, p)].R for p in names}
            s_val = ens_score_steps(cb, ce, names, fits, tr,
                                    {p: val.stores[(POOLING, p)].R for p in names})
            s_test = ens_score_steps(cb, ce, names, fits, tr,
                                     {p: test.stores[(POOLING, p)].R for p in names})
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
                    rows.append({"seed": i, "layer_range": label, "gamma": gamma,
                                 "w": str(w), **_metrics_row(vm, tm, gi)})
        del train, val, test
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def config_metrics(df: pd.DataFrame, config: dict, n_seeds: int) -> dict:
    """Mean val/test metrics of one config over the pair indices."""
    g = df
    for ax, v in config.items():
        g = g[g[ax].astype(str) == str(v)]
    assert len(g) == n_seeds, f"config {config} has {len(g)} rows, expected {n_seeds}"
    return {c: float(g[c].mean()) for c in METRIC_COLS}


def out_row(model, src, tgt, convention, row, base_cfg, resc_cfg, m) -> dict:
    resc_cfg = resc_cfg or {}
    return {"model": model, "source": src["name"], "target": tgt["name"],
            "convention": convention, "row": row,
            "position": base_cfg["position"], "c_begin": int(base_cfg["c_begin"]),
            "c_end": int(base_cfg["c_end"]),
            "layer_range": resc_cfg.get("layer_range", ""),
            "gamma": float(resc_cfg.get("gamma", 0.0)),
            "w": resc_cfg.get("w", ""),
            "src_seeds": ",".join(map(str, src["seeds"])),
            "tgt_seeds": ",".join(map(str, tgt["seeds"])),
            "step_acc_test": m["step_acc_test@1"], "agent_acc_test": m["agent_acc_test@1"],
            "step_acc_val": m["step_acc_val@1"], "agent_acc_val": m["agent_acc_val@1"]}


def run_pair(src, tgt, model, attn, device) -> list[dict]:
    n_seeds = len(src["seeds"])
    weightings, labels = attn[tgt["name"]]
    base_df = base_grid(src, tgt, model, device)
    sel_cols = {"test": ("step_acc_test@1", "agent_acc_test@1"),
                "val": ("step_acc_val@1", "agent_acc_val@1")}
    base_sel = {conv: select_config(base_df, BASE_SWEPT, list(range(n_seeds)), sc, ac)
                for conv, (sc, ac) in sel_cols.items()}
    resc_cache: dict[tuple, pd.DataFrame] = {}
    rows = []
    for conv in ("test", "val"):
        bcfg = base_sel[conv]["config"]
        key = (bcfg["position"], int(bcfg["c_begin"]), int(bcfg["c_end"]))
        if key not in resc_cache:
            resc_cache[key] = rescore_grid(src, tgt, bcfg, weightings, labels, device)
        resc_df = resc_cache[key]
        sc, ac = sel_cols[conv]
        soap_sel = select_config(resc_df, RESCORE_SWEPT, list(range(n_seeds)), sc, ac)
        bm = config_metrics(base_df, bcfg, n_seeds)
        sm = config_metrics(resc_df, soap_sel["config"], n_seeds)
        rows.append(out_row(model, src, tgt, conv, "base", bcfg, None, bm))
        rows.append(out_row(model, src, tgt, conv, "soap", bcfg, soap_sel["config"], sm))
        print(f"  [{conv}] base={bcfg} {bm['step_acc_test@1']:.4f} "
              f"soap={soap_sel['config']} {sm['step_acc_test@1']:.4f}")

        # The test-selected diagonal repeats Table 1's selection problem exactly.
        if src["name"] == tgt["name"] and conv == "test":
            assert_close(bm["step_acc_test@1"], float(src["svd"]["step_acc_test"]),
                         f"{model} {src['name']} diagonal base vs Table 1")
            assert_close(sm["step_acc_test@1"], float(src["bp"]["step_acc_test"]),
                         f"{model} {src['name']} diagonal soap vs Table 1")
            print(f"  diagonal {src['name']} verified against the selection table")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--sources", nargs="+", default=None,
                   help="short names (WW-AG WW-HC TE-Cap TE-Mag); default all")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    models = sorted({m for _, m, _ in iter_cells()})
    if args.models:
        models = [m for m in models if m in args.models]
    rows = []
    for model in models:
        cells = collect_cells(model)
        attn = {}
        for c in cells:
            weightings, bounds = aggregate_attn(C.attn_root(c["cfg"]), model,
                                                c["subset"], n_ranges=c["cfg"]["n_ranges"],
                                                device=args.device)
            attn[c["name"]] = (weightings, [f"{lo}-{hi}" for lo, hi in bounds])
        for src in cells:
            if args.sources and src["name"] not in args.sources:
                continue
            for tgt in cells:
                print(f"[{model}] {src['name']} -> {tgt['name']}")
                rows.extend(run_pair(src, tgt, model, attn, args.device))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    order = ["WW-AG", "WW-HC", "TE-Cap", "TE-Mag"]
    for (model, conv), g in df[df.row == "soap"].groupby(["model", "convention"]):
        pivot = g.pivot_table(index="source", columns="target",
                              values="step_acc_test") * 100
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        print(f"\n=== {model} soap, {conv}-selected (step acc %; rows=source) ===")
        print(pivot[[s for s in order if s in pivot.columns]].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
