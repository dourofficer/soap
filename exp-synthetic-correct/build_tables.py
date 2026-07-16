"""Stages 2/4/5 — CPU reducers over the stage-1/3 outputs.

  --stage undisc   concat persisted selections → undiscounted/.../weighted_<flag>.tsv,
                   cross-checking every winner against the full-grid TSV
                   (re-derived with the production best_per_group reducer).
                   --validate additionally recomputes metrics from the stored
                   per-step tensors (rebuilds keepers; slower).
  --stage disc     reduce rescore sweep → rescore/reduced/.../svd.tsv
                   (CRR hyperparams selected on disc test) and svd_valsel.tsv
                   (selected on disc val), one row per (pooling, seed, sel_by).
  --stage summary  headline table: per (model, target, convention) mean±std of
                   test metrics over seeds. Convention pairs the base-config
                   selection (sel_by) with the matching CRR-hyperparam
                   selection (val↔svd_valsel.tsv, test↔svd.tsv); val is the
                   leak-free headline, test mirrors production.

    python exp-synthetic-correct/build_tables.py --stage undisc [--validate]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import common
from experiments.reports._common import best_per_group, sort_section
from src.utils.utils import compute_metrics, load_representations

UNDISC_COLS = [
    "strategy", "position", "pooling", "method", "c_begin", "c_end",
    "centered", "weighted", "threshold", "seed", "sel_by",
    "step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test",
]

DISC_COLS = [
    "seed", "pooling", "sel_by", "source",
    "position", "method", "c_begin", "c_end", "centered", "weighted",
    "undisc_step_acc_val", "undisc_agent_acc_val",
    "undisc_step_acc_test", "undisc_agent_acc_test",
    "svd_orient", "layer_range", "gamma", "w",
    "disc_step_acc_val", "disc_agent_acc_val",
    "disc_step_acc_test", "disc_agent_acc_test",
    "diff_step_acc_val", "diff_agent_acc_val",
    "diff_step_acc_test", "diff_agent_acc_test",
]


def _load_bundles(cfg, model, ds, subset):
    pt_dir = common.scores_dir(cfg, model, ds, subset)
    paths = sorted(pt_dir.glob("selected_pooling-*_seed-*.pt"))
    return [torch.load(p, weights_only=False) for p in paths]


# ── stage: undisc ─────────────────────────────────────────────────────────────

def _crosscheck_row(cfg, model, ds, subset, row: dict, pooling, seed) -> None:
    """Re-derive this winner from the full-grid TSV with the production
    reducer; any mismatch means selection or persistence drifted."""
    tsv = common.svd_tsv(cfg, model, ds, subset, pooling, seed)
    table = pd.read_csv(tsv, sep="\t")
    rows = table[table["weighted"] == bool(row["weighted"])]
    sel = row["sel_by"]
    ref = best_per_group(rows, [f"step_acc_{sel}", f"agent_acc_{sel}"],
                         ["pooling"]).iloc[0]
    for col in common.CONFIG_KEYS:
        assert ref[col] == row[col], (
            f"cross-check mismatch {model}/{ds}/{subset} {pooling}/seed-{seed} "
            f"sel_by={sel} col={col}: table={ref[col]!r} persisted={row[col]!r}")
    for col in ["step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test"]:
        assert abs(float(ref[col]) - float(row[col])) < 1e-9, (
            f"cross-check metric mismatch {model}/{ds}/{subset} "
            f"{pooling}/seed-{seed} sel_by={sel} col={col}")


def _validate_bundle(cfg, model, ds, subset, bundle) -> None:
    """Recompute metrics from the persisted per-step tensors and compare."""
    roots = cfg["datasets"][ds]
    meta = bundle["meta"]
    keepers = {}
    for split in ("val", "test"):
        reps = load_representations(
            rep_dir=roots["reps_root"] / model / subset,
            data_dir=roots["data_root"] / subset,
            pooling=meta["pooling"], weight_names=["embed"], device="cpu",
            files=bundle[f"{split}_files"])
        keepers[split] = reps.keeper
    for row in bundle["rows"]:
        for split in ("val", "test"):
            # Raw SVD scores are "lower = error" → direction asc, as in the
            # full-grid table.
            m = compute_metrics(row[f"{split}_scores"].numpy(),
                                keepers[split], ks=[1], direction="asc")
            step, agent = m["step@1_asc"], m["agent@1_asc"]
            assert abs(step - row[f"step_acc_{split}"]) < 1e-9, (
                f"validate: step_acc_{split} mismatch "
                f"{model}/{ds}/{subset} {meta['pooling']}/seed-{meta['seed']}")
            assert abs(agent - row[f"agent_acc_{split}"]) < 1e-9, (
                f"validate: agent_acc_{split} mismatch "
                f"{model}/{ds}/{subset} {meta['pooling']}/seed-{meta['seed']}")


def stage_undisc(cfg, validate: bool) -> None:
    for model in cfg["models"]:
        for ds, subset in common.iter_targets(cfg):
            bundles = _load_bundles(cfg, model, ds, subset)
            if not bundles:
                print(f"skip (no bundles): {model}/{ds}/{subset}")
                continue
            recs = []
            for bundle in bundles:
                meta = bundle["meta"]
                for row in bundle["rows"]:
                    _crosscheck_row(cfg, model, ds, subset, row,
                                    meta["pooling"], meta["seed"])
                    recs.append({
                        "strategy": "svd",
                        **{k: row[k] for k in common.CONFIG_KEYS},
                        "threshold": pd.NA,
                        "seed": meta["seed"],
                        "sel_by": row["sel_by"],
                        **{k: row[k] for k in ["step_acc_val", "agent_acc_val",
                                               "step_acc_test", "agent_acc_test"]},
                    })
                if validate:
                    _validate_bundle(cfg, model, ds, subset, bundle)
            df = pd.DataFrame(recs)[UNDISC_COLS]
            out_dir = common.undisc_dir(cfg, model, ds, subset)
            out_dir.mkdir(parents=True, exist_ok=True)
            for flag in cfg["select"]["weighted"]:
                part = sort_section(df[df["weighted"] == flag],
                                    order_cols=["sel_by", "pooling", "seed"])
                dst = out_dir / f"weighted_{'true' if flag else 'false'}.tsv"
                part.to_csv(dst, sep="\t", index=False, na_rep="")
                print(f"wrote {dst}  ({len(part)} rows)")


# ── stage: disc ───────────────────────────────────────────────────────────────

def stage_disc(cfg) -> None:
    for model in cfg["models"]:
        for ds, subset in common.iter_targets(cfg):
            src = common.sweep_dir(cfg, model, ds, subset) / "svd.tsv"
            if not src.exists():
                print(f"skip (missing): {src}")
                continue
            raw = pd.read_csv(src, sep="\t").rename(columns={"orient": "svd_orient"})
            out_dir = common.reduced_dir(cfg, model, ds, subset)
            out_dir.mkdir(parents=True, exist_ok=True)
            for sel, fname in (("test", "svd.tsv"), ("val", "svd_valsel.tsv")):
                best = best_per_group(
                    raw, [f"disc_step_acc_{sel}", f"disc_agent_acc_{sel}"],
                    ["pooling", "seed", "sel_by"])
                for split in ("val", "test"):
                    best[f"diff_step_acc_{split}"] = (
                        best[f"disc_step_acc_{split}"] - best[f"undisc_step_acc_{split}"])
                    best[f"diff_agent_acc_{split}"] = (
                        best[f"disc_agent_acc_{split}"] - best[f"undisc_agent_acc_{split}"])
                best = sort_section(best, order_cols=["sel_by", "pooling", "seed"])
                cols = [c for c in DISC_COLS if c in best.columns]
                dst = out_dir / fname
                best[cols].to_csv(dst, sep="\t", index=False, na_rep="")
                print(f"wrote {dst}  ({len(best)} rows)")


# ── stage: summary ────────────────────────────────────────────────────────────

def _pick_pooling(rows: pd.DataFrame, step_col: str, agent_col: str) -> pd.Series:
    """Among this seed's poolings, pick by the selection metric."""
    return rows.sort_values([step_col, agent_col], ascending=False,
                            kind="mergesort").iloc[0]


def stage_summary(cfg) -> None:
    per_seed, agg = [], []
    for model in cfg["models"]:
        for ds, subset in common.iter_targets(cfg):
            undisc = None
            for flag in cfg["select"]["weighted"]:
                p = (common.undisc_dir(cfg, model, ds, subset)
                     / f"weighted_{'true' if flag else 'false'}.tsv")
                if p.exists():
                    t = pd.read_csv(p, sep="\t")
                    undisc = t if undisc is None else pd.concat([undisc, t])
            if undisc is None:
                print(f"skip (no undisc): {model}/{ds}/{subset}")
                continue

            for conv in cfg["select"]["by"]:
                red_name = "svd_valsel.tsv" if conv == "val" else "svd.tsv"
                red_path = common.reduced_dir(cfg, model, ds, subset) / red_name
                reduced = pd.read_csv(red_path, sep="\t") if red_path.exists() else None

                rows = {"svd": [], "crr": []}
                for seed in cfg["seeds"]:
                    u = undisc[(undisc["seed"] == seed) & (undisc["sel_by"] == conv)]
                    if not u.empty:
                        pick = _pick_pooling(u, f"step_acc_{conv}", f"agent_acc_{conv}")
                        rows["svd"].append(pick)
                        per_seed.append({
                            "model": model, "dataset": ds, "subset": subset,
                            "convention": conv, "stage": "svd", "seed": seed,
                            "pooling": pick["pooling"],
                            "step_acc_test": pick["step_acc_test"],
                            "agent_acc_test": pick["agent_acc_test"]})
                    if reduced is not None:
                        r = reduced[(reduced["seed"] == seed)
                                    & (reduced["sel_by"] == conv)]
                        if not r.empty:
                            pick = _pick_pooling(r, f"disc_step_acc_{conv}",
                                                 f"disc_agent_acc_{conv}")
                            rows["crr"].append(pick)
                            per_seed.append({
                                "model": model, "dataset": ds, "subset": subset,
                                "convention": conv, "stage": "crr", "seed": seed,
                                "pooling": pick["pooling"],
                                "layer_range": pick["layer_range"],
                                "gamma": pick["gamma"], "w": pick["w"],
                                "svd_orient": pick["svd_orient"],
                                "step_acc_test": pick["disc_step_acc_test"],
                                "agent_acc_test": pick["disc_agent_acc_test"]})

                for stage, picks in rows.items():
                    if not picks:
                        continue
                    prefix = "" if stage == "svd" else "disc_"
                    steps = np.array([p[f"{prefix}step_acc_test"] for p in picks],
                                     dtype=float)
                    agents = np.array([p[f"{prefix}agent_acc_test"] for p in picks],
                                      dtype=float)
                    agg.append({
                        "model": model, "dataset": ds, "subset": subset,
                        "convention": conv, "stage": stage, "n_seeds": len(picks),
                        "step_acc_test_mean": steps.mean(),
                        "step_acc_test_std": steps.std(ddof=1) if len(steps) > 1 else 0.0,
                        "agent_acc_test_mean": agents.mean(),
                        "agent_acc_test_std": agents.std(ddof=1) if len(agents) > 1 else 0.0,
                    })

    out_dir = common.summary_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agg).to_csv(out_dir / "results_table.tsv", sep="\t", index=False)
    pd.DataFrame(per_seed).to_csv(out_dir / "per_seed.tsv", sep="\t", index=False)
    print(f"wrote {out_dir / 'results_table.tsv'}  ({len(agg)} rows)")
    print(f"wrote {out_dir / 'per_seed.tsv'}  ({len(per_seed)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    p.add_argument("--stage", required=True, choices=["undisc", "disc", "summary"])
    p.add_argument("--validate", action="store_true",
                   help="undisc only: recompute metrics from stored tensors")
    args = p.parse_args()
    cfg = common.load_cfg(args.config, args.overrides)

    if args.stage == "undisc":
        stage_undisc(cfg, validate=args.validate)
    elif args.stage == "disc":
        stage_disc(cfg)
    else:
        stage_summary(cfg)


if __name__ == "__main__":
    main()
