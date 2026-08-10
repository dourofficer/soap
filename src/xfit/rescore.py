"""Cross-fit rescoring: CRR (discount) + backprop on the SYNTHETIC-fit base scores.

Structurally identical to ``src.rescore.run.run_pair`` — same sweep axes, same vectorized
all-gammas matmuls, same ``SWEEP_COLS`` schema, same orient/normalize/strategy path — with
exactly ONE change: the base-score vectors are reproduced from the SYNTHETIC ``svd_entry``
(``score_config(target_R, synth_entry, ...)``) instead of ``fit_one(target_train)``. The
target attention (the causal weights) is the real target's, unchanged.

Run PER SCORER (base_table ``base_ext_<scorer>_test.tsv``, strategies discount+backprop),
then ``reduce_crr`` -> ``crr_ext_<scorer>_{test,val}.tsv`` + ``backprop_ext_<scorer>_*``.

    # from repo root (xfit.score must have run)
    python -m src.xfit.rescore --source magentic-qwen9b --proxy qwen3.5-9b
"""
from __future__ import annotations

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..common import paths
from ..metrics import KeeperContext, compute_metrics_batch
from ..stores import load_representations, split_files, split_data, list_rep_files
from ..score.svd import fit_one, score_config, N_COMPONENTS
from ..score.scorers import native_direction
from ..score.ensemble import ENSEMBLE_POSITION, member_positions, ens_score_vec
from ..rescore.weights import aggregate_attn
from ..rescore.strategies import orient, allowed_orients, normalize_scores
from ..rescore.run import SWEEP_COLS, _undisc_from_row
from .legacy import WCacheLegacy as WCache, LEGACY_STRATEGIES as STRATEGIES, reduce_crr, sweep_name
from . import prov
from .common import (load_config, source_tag, synth_reps_dir, synth_data_dir,
                     iter_sources, targets_for, target_cfg,
                     setting, paper_cfg, paper_jobs, paper_seeds)
from .score import build_entries, fit_entries_files, resolve_fit


def _base_rows(tcfg, proxy, subset, scorer, conv: str = "test") -> pd.DataFrame | None:
    p = paths.reduced_root(tcfg) / proxy / subset / f"base_ext_{scorer}_{conv}.tsv"
    if not p.exists():
        print(f"[skip] no base table {p}")
        return None
    return pd.read_csv(p, sep="\t")


def rescore_target(proxy, source, dataset, subset, scorer, entries, cfg,
                   device, force=False) -> None:
    tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source),
                      variant=f"ext_{scorer}")
    base = _base_rows(tcfg, proxy, subset, scorer)
    if base is None or base.empty:
        return
    out = paths.rescore_root(tcfg) / proxy / subset / sweep_name(tcfg)
    if out.exists() and not force:
        print(f"[skip] {out}")
        return

    ratio = cfg.get("target_val_ratio", 0.5)
    ks = cfg["ks"]
    gammas, ws = list(cfg["gammas"]), list(cfg["ws"])
    orients = list(cfg["orients"])
    score_norms = list(cfg.get("score_norms", ["none"]))
    strategies = list(cfg.get("strategies", ["discount", "backprop"]))
    n_ranges = cfg.get("n_ranges", 4)

    rep_dir = paths.reps_root(tcfg) / proxy / subset
    data_dir = paths.data_root(tcfg) / subset
    files = list_rep_files(rep_dir)
    weightings, bounds = aggregate_attn(paths.attn_root(tcfg), proxy, subset,
                                        n_ranges=n_ranges, device=device)
    range_labels = [f"{lo}-{hi}" for lo, hi in bounds]

    records = []
    for seed, seed_rows in base.groupby("seed"):
        val_files, test_files = split_data(files, ratio, int(seed))
        assert test_files == split_files(files, tcfg["splits"], int(seed))["test"], \
            f"5/5 test != 325 test for {dataset}/{subset} seed={seed}"
        load = lambda fl: load_representations(rep_dir, data_dir, poolings=cfg["poolings"],
                                               files=fl, device=device)
        val, test = load(val_files), load(test_files)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        val_WC = WCache(weightings, val.keeper, ws, device=device)
        test_WC = WCache(weightings, test.keeper, ws, device=device)

        for _, row in tqdm(list(seed_rows.iterrows()),
                           desc=f"{proxy}/{subset} {scorer} s{seed}", leave=False):
            pooling, position, method = row["pooling"], row["position"], row["method"]
            if position == ENSEMBLE_POSITION:
                continue                    # xfit never emits ensemble base rows
            cb, ce = int(row["c_begin"]), int(row["c_end"])
            cen, wt = bool(row["centered"]), bool(row["weighted"])
            entry = entries[(pooling, position)]
            s_val = score_config(val.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
            s_test = score_config(test.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
            direction = native_direction(method, row.get("direction"))
            undisc = _undisc_from_row(row)          # base metric, orient-independent

            for orient_name in allowed_orients(method, orients):
                ov, ot = orient(s_val, orient_name), orient(s_test, orient_name)
                for snorm in score_norms:
                    nv = normalize_scores(ov, val.keeper, snorm)
                    nt = normalize_scores(ot, test.keeper, snorm)
                    for r_idx, label in enumerate(range_labels):
                        for w in ws:
                            vmats, tmats = val_WC.mats(r_idx, w), test_WC.mats(r_idx, w)
                            for strat in strategies:
                                fn = STRATEGIES[strat]
                                Sv = fn(nv, val.keeper, vmats, gammas).T.contiguous()
                                St = fn(nt, test.keeper, tmats, gammas).T.contiguous()
                                vm = compute_metrics_batch(Sv, None, ks, "desc", ctx=val_ctx)
                                tm = compute_metrics_batch(St, None, ks, "desc", ctx=test_ctx)
                                for gi, gamma in enumerate(gammas):
                                    records.append({
                                        "seed": int(seed), "pooling": pooling, "position": position,
                                        "method": method, "c_begin": cb, "c_end": ce,
                                        "centered": cen, "weighted": wt, "direction": direction,
                                        "orient": orient_name, "score_norm": snorm, "strategy": strat,
                                        "layer_range": label, "gamma": gamma, "w": w,
                                        **undisc,
                                        "disc_step_acc_val": vm["step@1_desc"][gi],
                                        "disc_agent_acc_val": vm["agent@1_desc"][gi],
                                        "disc_step_acc_test": tm["step@1_desc"][gi],
                                        "disc_agent_acc_test": tm["agent@1_desc"][gi],
                                    })
        del val, test, val_WC, test_WC
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)[SWEEP_COLS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"  wrote {out}  ({len(df)} rows)")


def rescore_target_paper(proxy, source, dataset, subset, scorer, cfg,
                         device, force=False) -> list:
    """Paper-setting sweep: standard 325 val/test, per-seed fit corpus (real train or
    the recorded question-aligned subsample), and the ens-mid3 branch of the core
    ``run_pair`` instead of the legacy skip — the z-statistics come from the fit corpus
    through ``ens_score_vec``'s existing train-stats parameter."""
    pc = paper_cfg(cfg)
    tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg),
                      variant=f"ext_{scorer}")
    base = _base_rows(tcfg, proxy, subset, scorer, pc["base_table_convention"])
    if base is None or base.empty:
        return []
    out = paths.rescore_root(tcfg) / proxy / subset / sweep_name(tcfg)
    if out.exists() and not force:
        print(f"[skip] {out}")
        return []

    ks = cfg["ks"]
    gammas, ws = list(cfg["gammas"]), list(cfg["ws"])
    orients = list(cfg["orients"])
    score_norms = list(cfg.get("score_norms", ["none"]))
    strategies = list(cfg.get("strategies", ["discount", "backprop"]))
    n_ranges = cfg.get("n_ranges", 4)
    poolings = cfg["poolings"]

    rep_dir = paths.reps_root(tcfg) / proxy / subset
    data_dir = paths.data_root(tcfg) / subset
    files = list_rep_files(rep_dir)
    weightings, bounds = aggregate_attn(paths.attn_root(tcfg), proxy, subset,
                                        n_ranges=n_ranges, device=device)
    range_labels = [f"{lo}-{hi}" for lo, hi in bounds]

    records = []
    for seed, seed_rows in base.groupby("seed"):
        parts = split_files(files, tcfg["splits"], int(seed))
        fit_dir, fit_data, fit_fl = resolve_fit(proxy, source, dataset, subset,
                                                int(seed), parts, cfg, rep_dir, data_dir)
        fit_reps = load_representations(fit_dir, fit_data, poolings=poolings,
                                        files=fit_fl, device=device)
        load = lambda fl: load_representations(rep_dir, data_dir, poolings=poolings,
                                               files=fl, device=device)
        val, test = load(parts["val"]), load(parts["test"])
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        val_WC = WCache(weightings, val.keeper, ws, device=device)
        test_WC = WCache(weightings, test.keeper, ws, device=device)
        fit_cache: dict = {}

        def _fit(pooling, pos):
            k = (pooling, pos)
            if k not in fit_cache:
                fit_cache[k] = fit_one(fit_reps.stores[k].R, N_COMPONENTS)
            return fit_cache[k]

        for _, row in tqdm(list(seed_rows.iterrows()),
                           desc=f"{proxy}/{subset} {scorer} s{seed}", leave=False):
            pooling, position, method = row["pooling"], row["position"], row["method"]
            cb, ce = int(row["c_begin"]), int(row["c_end"])
            cen, wt = bool(row["centered"]), bool(row["weighted"])
            is_ens = position == ENSEMBLE_POSITION
            if is_ens:
                members = member_positions(fit_reps.positions())
                fits = {p: _fit(pooling, p) for p in members}
                fitR = {p: fit_reps.stores[(pooling, p)].R for p in members}
                s_val = ens_score_vec(method, cb, ce, cen, wt, members, fits, fitR,
                                      {p: val.stores[(pooling, p)].R for p in members})
                s_test = ens_score_vec(method, cb, ce, cen, wt, members, fits, fitR,
                                       {p: test.stores[(pooling, p)].R for p in members})
            else:
                entry = _fit(pooling, position)
                s_val = score_config(val.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
                s_test = score_config(test.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
            direction = "desc" if is_ens else native_direction(method, row.get("direction"))
            undisc = _undisc_from_row(row)          # base metric, orient-independent
            # ens-mid3 scores are already 'higher = error' -> no orientation.
            row_orients = ["none"] if is_ens else allowed_orients(method, orients)

            for orient_name in row_orients:
                ov, ot = orient(s_val, orient_name), orient(s_test, orient_name)
                for snorm in score_norms:
                    nv = normalize_scores(ov, val.keeper, snorm)
                    nt = normalize_scores(ot, test.keeper, snorm)
                    for r_idx, label in enumerate(range_labels):
                        for w in ws:
                            vmats, tmats = val_WC.mats(r_idx, w), test_WC.mats(r_idx, w)
                            for strat in strategies:
                                fn = STRATEGIES[strat]
                                Sv = fn(nv, val.keeper, vmats, gammas).T.contiguous()
                                St = fn(nt, test.keeper, tmats, gammas).T.contiguous()
                                vm = compute_metrics_batch(Sv, None, ks, "desc", ctx=val_ctx)
                                tm = compute_metrics_batch(St, None, ks, "desc", ctx=test_ctx)
                                for gi, gamma in enumerate(gammas):
                                    records.append({
                                        "seed": int(seed), "pooling": pooling, "position": position,
                                        "method": method, "c_begin": cb, "c_end": ce,
                                        "centered": cen, "weighted": wt, "direction": direction,
                                        "orient": orient_name, "score_norm": snorm, "strategy": strat,
                                        "layer_range": label, "gamma": gamma, "w": w,
                                        **undisc,
                                        "disc_step_acc_val": vm["step@1_desc"][gi],
                                        "disc_agent_acc_val": vm["agent@1_desc"][gi],
                                        "disc_step_acc_test": tm["step@1_desc"][gi],
                                        "disc_agent_acc_test": tm["agent@1_desc"][gi],
                                    })
        del val, test, val_WC, test_WC, fit_reps, fit_cache
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)[SWEEP_COLS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"  wrote {out}  ({len(df)} rows)")
    return [out]


def reduce_target(source, dataset, subset, scorer, cfg) -> None:
    reduce_crr(target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg),
                          variant=f"ext_{scorer}"))


def run_paper(cfg, only_proxy=None, only_source=None, only_dataset=None,
              device="cuda", force=False, do_rescore=True, do_reduce=True) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proxies = [only_proxy] if only_proxy else list(cfg["proxies"])
    scorers = list(cfg["methods"])
    jobs = [(s, d, sub) for s, d, sub in paper_jobs(cfg)
            if only_source in (None, s) and only_dataset in (None, d)]
    for source, dataset, subset in jobs:
        tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
        prov.ensure_tag_config(paths.scores_root(tcfg), cfg, force=force)
    outputs_by_ds: dict[str, list] = {}
    if do_rescore:
        for proxy in proxies:
            for source, dataset, subset in jobs:
                for scorer in scorers:
                    written = rescore_target_paper(proxy, source, dataset, subset,
                                                   scorer, cfg, device, force=force)
                    outputs_by_ds.setdefault(dataset, []).extend(written)
    if do_reduce:
        for source, dataset, subset in jobs:
            for scorer in scorers:
                reduce_target(source, dataset, subset, scorer, cfg)
    for dataset, outs in outputs_by_ds.items():
        subset = next(sub for _, d, sub in jobs if d == dataset)
        prov.record(cfg, target_cfg(dataset, subset, cfg), "rescore", outs)


def run(cfg, only_proxy=None, only_source=None, only_dataset=None,
        device="cuda", force=False, do_rescore=True, do_reduce=True) -> None:
    """Rescore (per proxy) then reduce (over ALL proxies). Separable like ``score.run`` so
    concurrent (proxy, source) sweeps can run with ``do_reduce=False`` and a single serial
    ``do_rescore=False`` pass reduces every proxy's sweep without racing on the writes."""
    if setting(cfg) == "paper":
        return run_paper(cfg, only_proxy=only_proxy, only_source=only_source,
                         only_dataset=only_dataset, device=device, force=force,
                         do_rescore=do_rescore, do_reduce=do_reduce)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proxies = [only_proxy] if only_proxy else list(cfg["proxies"])
    scorers = list(cfg["methods"])
    if do_rescore:
        for proxy in proxies:
            for harness, gen, source in iter_sources(cfg):
                if only_source not in (None, source):
                    continue
                targets = [t for t in targets_for(cfg, harness)
                           if only_dataset in (None, t["dataset"])]
                if not targets:
                    continue
                entries = build_entries(proxy, source, cfg, device)
                for tgt in targets:
                    for scorer in scorers:
                        rescore_target(proxy, source, tgt["dataset"], tgt["subset"],
                                       scorer, entries, cfg, device, force=force)
                del entries
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    if do_reduce:
        # reductions read ALL proxies' sweep TSVs -> run once per (source, target, scorer).
        done: set = set()
        for harness, gen, source in iter_sources(cfg):
            if only_source not in (None, source):
                continue
            for tgt in targets_for(cfg, harness):
                if only_dataset not in (None, tgt["dataset"]):
                    continue
                for scorer in scorers:
                    key = (source, tgt["dataset"], tgt["subset"], scorer)
                    if key in done:
                        continue
                    done.add(key)
                    reduce_target(source, tgt["dataset"], tgt["subset"], scorer, cfg)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proxy", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-rescore", action="store_true", help="skip sweeps; only reduce.")
    p.add_argument("--no-reduce", action="store_true", help="skip reductions (parallel-safe sweeps).")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    run(load_config(args.overrides), only_proxy=args.proxy, only_source=args.source,
        only_dataset=args.dataset, device=args.device, force=args.force,
        do_rescore=not args.no_rescore, do_reduce=not args.no_reduce)


if __name__ == "__main__":
    main()
