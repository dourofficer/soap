"""Cross-fit base scoring: SVD fit on SYNTHETIC reps, score the real 325 test split.

The cross-fit is trivial because ``src.score.svd`` already decouples fit from score:
``fit_one(R_synth)`` yields an ``svd_entry`` that ``score_from_entry`` applies to ANY
representation matrix. So the only new logic here is (a) fitting on the synthetic source
and (b) splitting the target 5/5 val/test — with the test half asserted byte-identical to
the 325 test, so the cells stay comparable to ``results_extended``.

Writes the standard score schema (identical ``OUT_COLS``) to
``outputs/<target_ds>/scores/xfit-<source>/<proxy>/<subset>/seed-<n>.tsv`` and then the
base reductions (default + per-scorer ``ext_<scorer>`` variants) via ``reduce_base``.

    # from repo root (extraction must have run)
    python -m src.xfit.score                          # all cells
    python -m src.xfit.score --source magentic-qwen9b --proxy qwen3.5-9b
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..common import paths
from ..metrics import KeeperContext
from ..stores import load_representations, split_files, split_data, list_rep_files
from ..score.svd import fit_one, score_from_entry, N_COMPONENTS
from ..score.ensemble import member_positions, ensemble_rows
from ..score.run import OUT_COLS
from .legacy import reduce_base
from . import align, prov
from .common import (load_config, source_tag, synth_reps_dir, synth_data_dir,
                     iter_sources, targets_for, target_cfg,
                     setting, paper_cfg, paper_jobs, paper_seeds, REAL_SOURCE)


# ── synthetic SVD entries (fit once per proxy x source) ──────────────────────
def build_entries(proxy: str, source: str, cfg: dict, device: str) -> dict:
    """``(pooling, position) -> fit_one(synthetic R)`` over ALL fit-pool source reps."""
    rep_dir, data_dir = synth_reps_dir(proxy, source), synth_data_dir(source)
    reps = load_representations(rep_dir, data_dir, poolings=cfg["poolings"], device=device)
    n_comp = cfg.get("n_components", N_COMPONENTS)
    entries = {}
    for pooling in cfg["poolings"]:
        for position in reps.positions():
            entries[(pooling, position)] = fit_one(reps.stores[(pooling, position)].R, n_comp)
    return entries


# ── score one target under a synthetic fit ───────────────────────────────────
def score_target(proxy, source, dataset, subset, entries, cfg, device, force=False) -> None:
    tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source))
    ratio = cfg.get("target_val_ratio", 0.5)
    methods, poolings = cfg["methods"], cfg["poolings"]
    ks, n_comp = cfg["ks"], cfg.get("n_components", N_COMPONENTS)

    rep_dir = paths.reps_root(tcfg) / proxy / subset
    data_dir = paths.data_root(tcfg) / subset
    files = list_rep_files(rep_dir)

    # key-coverage: the synthetic fit must span every target position for this proxy.
    tgt_positions = load_representations(
        rep_dir, data_dir, poolings=poolings[:1], files=files[:1], device="cpu").positions()
    missing = [(p, pos) for p in poolings for pos in tgt_positions if (p, pos) not in entries]
    if missing:
        raise SystemExit(f"synthetic fit {source}/{proxy} missing positions {missing[:3]}...")

    for seed in tcfg["seeds"]:
        out = paths.scores_root(tcfg) / proxy / subset / f"seed-{seed}.tsv"
        if out.exists() and not force:
            print(f"[skip] {out}")
            continue

        # 5/5 target split; assert the test half is the SAME trajectories as the 325 test.
        val_files, test_files = split_data(files, ratio, seed)
        ref_test = split_files(files, tcfg["splits"], seed)["test"]
        assert test_files == ref_test, (
            f"5/5 test != 325 test for {dataset}/{subset} seed={seed} "
            f"({len(test_files)} vs {len(ref_test)})")

        load = lambda fl: load_representations(rep_dir, data_dir, poolings=poolings,
                                               files=fl, device=device)
        val, test = load(val_files), load(test_files)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)

        rows = []
        for pooling in poolings:
            for position in tgt_positions:
                rows += score_from_entry(
                    pooling, position, entries[(pooling, position)],
                    val.stores[(pooling, position)].R, test.stores[(pooling, position)].R,
                    val_ctx, test_ctx, methods, [False], ks,
                    n_components=n_comp, device=None)
        for r in rows:
            r["seed"] = seed
        df = pd.DataFrame(rows)[OUT_COLS].sort_values(
            "step_acc_test", ascending=False, kind="mergesort")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep="\t", index=False)
        print(f"  wrote {out}  ({len(df)} rows)")
        del val, test
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ── paper setting: per-(target, seed) fits with the standard 325 partition ───
def fit_entries_files(rep_dir, data_dir, files, poolings, n_comp, device):
    """Fit every (pooling, position) on a file subset; also return the loaded stores
    (the ensemble's z-statistics come from the fit corpus, so the raw R is needed)."""
    reps = load_representations(rep_dir, data_dir, poolings=poolings, files=files,
                                device=device)
    entries = {(p, pos): fit_one(reps.stores[(p, pos)].R, n_comp)
               for p in poolings for pos in reps.positions()}
    return entries, reps


def _align_path(tcfg, subset, seed) -> Path:
    return paths.scores_root(tcfg) / "_align" / f"{subset}-seed{seed}.json"


def resolve_fit(proxy, source, dataset, subset, seed, parts, cfg,
                rep_dir, data_dir, record_align=False):
    """(fit_rep_dir, fit_data_dir, fit_files) for one paper cell.

    Real pseudo-source -> the target's own train split (exactly the main pipeline's
    fit). Synthetic -> the question-aligned per-seed subsample (or the whole fit pool
    under ``paper.fit: all``)."""
    pc = paper_cfg(cfg)
    if source == REAL_SOURCE:
        return rep_dir, data_dir, list(parts["train"])
    s_rep, s_data = synth_reps_dir(proxy, source), synth_data_dir(source)
    if pc["fit"] == "all":
        return s_rep, s_data, None
    stems = [Path(f).stem for f in parts["train"]]
    json_files, report = align.fit_files(dataset, subset, seed, source, stems, cfg["pools"])
    tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
    p = _align_path(tcfg, subset, seed)
    if record_align:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
    elif p.exists():
        recorded = json.loads(p.read_text())["files"]
        assert recorded == json_files, (
            f"alignment drift for {dataset}/{subset} seed={seed} source={source}: "
            f"recomputed fit set differs from the one the score stage recorded ({p})")
    return s_rep, s_data, [f.replace(".json", ".safetensors") for f in json_files]


def score_target_paper(proxy, source, dataset, subset, cfg, device, force=False) -> list:
    pc = paper_cfg(cfg)
    tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
    methods, poolings = cfg["methods"], cfg["poolings"]
    ks, n_comp = cfg["ks"], cfg.get("n_components", N_COMPONENTS)

    rep_dir = paths.reps_root(tcfg) / proxy / subset
    data_dir = paths.data_root(tcfg) / subset
    files = list_rep_files(rep_dir)

    written = []
    for seed in paper_seeds(cfg, dataset):
        out = paths.scores_root(tcfg) / proxy / subset / f"seed-{seed}.tsv"
        if out.exists() and not force:
            print(f"[skip] {out}")
            continue

        parts = split_files(files, tcfg["splits"], seed)   # the standard 325 partition
        fit_dir, fit_data, fit_fl = resolve_fit(proxy, source, dataset, subset, seed,
                                                parts, cfg, rep_dir, data_dir,
                                                record_align=True)
        entries, fit_reps = fit_entries_files(fit_dir, fit_data, fit_fl,
                                              poolings, n_comp, device)

        load = lambda fl: load_representations(rep_dir, data_dir, poolings=poolings,
                                               files=fl, device=device)
        val, test = load(parts["val"]), load(parts["test"])
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        positions = val.positions()
        missing = [(p, pos) for p in poolings for pos in positions
                   if (p, pos) not in entries]
        if missing:
            raise SystemExit(f"fit {source}/{proxy} missing positions {missing[:3]}...")

        # Same emission structure as src.score.run: positions, then (if enabled) the
        # ens-mid3 rows appended per pooling — ordering is load-bearing for the
        # stable-sort tie-breaks the Real-row assert depends on.
        rows = []
        for pooling in poolings:
            fits = {}
            for position in positions:
                entry = entries[(pooling, position)]
                fits[position] = entry
                rows += score_from_entry(
                    pooling, position, entry,
                    val.stores[(pooling, position)].R, test.stores[(pooling, position)].R,
                    val_ctx, test_ctx, methods, [False], ks,
                    n_components=n_comp, device=None)
            if pc["ensemble"]:
                members = member_positions(positions)
                rows += ensemble_rows(
                    pooling, members, fits,
                    {p: fit_reps.stores[(pooling, p)].R for p in members},
                    {p: val.stores[(pooling, p)].R for p in members},
                    {p: test.stores[(pooling, p)].R for p in members},
                    val_ctx, test_ctx, methods, [False], ks, n_components=n_comp)
        for r in rows:
            r["seed"] = seed
        df = pd.DataFrame(rows)[OUT_COLS].sort_values(
            "step_acc_test", ascending=False, kind="mergesort")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep="\t", index=False)
        written.append(out)
        print(f"  wrote {out}  ({len(df)} rows)")
        del val, test, entries, fit_reps
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return written


# ── base reductions for one target under a synthetic fit ─────────────────────
def reduce_target(source, dataset, subset, cfg) -> None:
    tag = source_tag(source, cfg)
    # default reduction -> base_{test,val} + base_by_method_{test,val} (feeds the SVD cell).
    reduce_base(target_cfg(dataset, subset, cfg, split_tag=tag,
                           headline_methods=list(cfg["methods"])))
    # per-scorer variants -> base_ext_<scorer>_{test,val} (feeds each rescore sweep).
    for scorer in cfg["methods"]:
        reduce_base(target_cfg(dataset, subset, cfg, split_tag=tag,
                               headline_methods=[scorer], variant=f"ext_{scorer}"))


def run_paper(cfg, only_proxy=None, only_source=None, only_dataset=None,
              device="cuda", force=False, do_score=True, do_reduce=True) -> None:
    """Paper-setting score phase: one SVD fit per (proxy, source|real, target, seed)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proxies = [only_proxy] if only_proxy else list(cfg["proxies"])
    jobs = [(s, d, sub) for s, d, sub in paper_jobs(cfg)
            if only_source in (None, s) and only_dataset in (None, d)]
    if not jobs:
        print("[paper] no jobs after narrowing")
        return
    outputs_by_ds: dict[str, list] = {}
    if do_score:
        # guard BEFORE scoring: a knob change over a half-populated tag must refuse,
        # not silently mix skipped old TSVs with a fresh stamp.
        for source, dataset, subset in jobs:
            tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
            prov.ensure_tag_config(paths.scores_root(tcfg), cfg, force=force)
        for proxy in proxies:
            for source, dataset, subset in tqdm(jobs, desc=f"paper/{proxy}", leave=False):
                print(f"== paper fit {proxy} on {source} -> {dataset}/{subset} ==")
                written = score_target_paper(proxy, source, dataset, subset, cfg,
                                             device, force=force)
                outputs_by_ds.setdefault(dataset, []).extend(written)
        for source, dataset, subset in jobs:
            tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
            prov.write_tag_config(paths.scores_root(tcfg), cfg,
                                  seeds=paper_seeds(cfg, dataset))
    if do_reduce:
        for source, dataset, subset in dict.fromkeys(jobs):
            tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
            prov.ensure_tag_config(paths.scores_root(tcfg), cfg, force=force)
            reduce_target(source, dataset, subset, cfg)
    for dataset, outs in outputs_by_ds.items():
        subset = next(sub for _, d, sub in jobs if d == dataset)
        prov.record(cfg, target_cfg(dataset, subset, cfg), "score", outs)


def run(cfg, only_proxy=None, only_source=None, only_dataset=None,
        device="cuda", force=False, do_score=True, do_reduce=True) -> None:
    """Score (per proxy) then reduce (over ALL proxies). The two phases are separable so
    the scoring of different (proxy, source) cells can run concurrently on separate GPUs
    with ``do_reduce=False``, and a single serial ``do_score=False`` pass then reduces
    over every proxy's TSVs without the reduction writes racing across jobs."""
    if setting(cfg) == "paper":
        return run_paper(cfg, only_proxy=only_proxy, only_source=only_source,
                         only_dataset=only_dataset, device=device, force=force,
                         do_score=do_score, do_reduce=do_reduce)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proxies = [only_proxy] if only_proxy else list(cfg["proxies"])
    if do_score:
        for proxy in proxies:
            for harness, gen, source in iter_sources(cfg):
                if only_source not in (None, source):
                    continue
                targets = [t for t in targets_for(cfg, harness)
                           if only_dataset in (None, t["dataset"])]
                if not targets:
                    continue
                print(f"== fit {proxy} on {source} ==")
                entries = build_entries(proxy, source, cfg, device)
                for tgt in tqdm(targets, desc=f"{proxy}/{source}", leave=False):
                    score_target(proxy, source, tgt["dataset"], tgt["subset"],
                                 entries, cfg, device, force=force)
                del entries
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    if do_reduce:
        # reductions read ALL proxies' score TSVs -> run once per (source, target).
        reduced: set = set()
        for harness, gen, source in iter_sources(cfg):
            if only_source not in (None, source):
                continue
            for tgt in targets_for(cfg, harness):
                if only_dataset not in (None, tgt["dataset"]):
                    continue
                key = (source, tgt["dataset"], tgt["subset"])
                if key in reduced:
                    continue
                reduced.add(key)
                reduce_target(source, tgt["dataset"], tgt["subset"], cfg)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proxy", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-score", action="store_true", help="skip scoring; only reduce.")
    p.add_argument("--no-reduce", action="store_true", help="skip reductions (parallel-safe scoring).")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    run(load_config(args.overrides), only_proxy=args.proxy, only_source=args.source,
        only_dataset=args.dataset, device=args.device, force=args.force,
        do_score=not args.no_score, do_reduce=not args.no_reduce)


if __name__ == "__main__":
    main()
