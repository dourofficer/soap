"""Rescore (CRR) sweep runner — one long/tidy TSV per (model, subset).

Reads the reduced base table (``base_table``), reproduces each base row's val+test
per-step scores, then sweeps orient x score_norm x layer_range x w x strategy and,
for ALL gammas at once (vectorized W matmuls + batched metrics), records undiscounted
and discounted step@k/agent@k on val + test.

Weights are built ONCE per (model, subset, seed) split (WCache); orientation is
auto-restricted per row (distance methods use ``none``). Writes
``outputs/<ds>/rescore/<tag>/<model>/<subset>/sweep.tsv``.

    # from v2/  (run the base reduction first)
    python -m src.reports.reduce --config configs/reduce/correct-full.yaml --set stage=base
    python -m src.rescore.run    --config configs/rescore/correct-full.yaml
"""
from __future__ import annotations

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..metrics import KeeperContext, compute_metrics_batch
from ..stores import load_representations, split_files, list_rep_files
from ..score.svd import fit_one, score_config, N_COMPONENTS
from ..score.ensemble import member_positions, ens_score_vec, ENSEMBLE_POSITION
from .weights import aggregate_attn, WCache, coerce_w
from .strategies import orient, allowed_orients, normalize_scores, STRATEGIES, discount_loop
from ..score.scorers import native_direction

SWEEP_COLS = [
    "seed", "pooling", "position", "method", "c_begin", "c_end", "centered", "weighted",
    "direction", "orient", "score_norm", "strategy", "layer_range", "gamma", "w",
    "undisc_step_acc_val", "undisc_agent_acc_val", "undisc_step_acc_test", "undisc_agent_acc_test",
    "disc_step_acc_val", "disc_agent_acc_val", "disc_step_acc_test", "disc_agent_acc_test",
]


def _base_rows(cfg, model, subset) -> pd.DataFrame | None:
    p = paths.reduced_root(cfg) / model / subset / cfg.get("base_table", "base_test.tsv")
    if not p.exists():
        return None
    return pd.read_csv(p, sep="\t")


def _undisc_from_row(row) -> dict:
    """Undiscounted reference = the base row's OWN recorded metrics.

    Deliberately NOT recomputed from the oriented score: orientation is part of the
    rescoring method, not of the baseline, and a saturating orient (sigmoid on
    large-magnitude scores) collapses the ranking to a tie, which would make the
    "no correction" reference meaningless. Being orient-independent also means the
    improvement column compares every orient against the same baseline."""
    return {
        "undisc_step_acc_val":   row["step_acc_val"],
        "undisc_agent_acc_val":  row["agent_acc_val"],
        "undisc_step_acc_test":  row["step_acc_test"],
        "undisc_agent_acc_test": row["agent_acc_test"],
    }


def run_pair(cfg, model, subset, rec) -> None:
    device = cfg.get("device", "cuda")
    ks = cfg["ks"]
    gammas = list(cfg["gammas"])
    ws = list(cfg["ws"])
    orients = list(cfg["orients"])
    score_norms = list(cfg.get("score_norms", ["none"]))
    strategies = list(cfg.get("strategies", ["discount"]))
    n_ranges = cfg.get("n_ranges", 4)
    poolings = cfg["poolings"]

    base = _base_rows(cfg, model, subset)
    if base is None or len(base) == 0:
        print(f"[skip] no base table for {model}/{subset}")
        return
    out = paths.rescore_root(cfg) / model / subset / "sweep.tsv"
    if out.exists() and not cfg.get("force"):
        print(f"[skip] {out}")
        return

    rep_dir = paths.reps_root(cfg) / model / subset
    data_dir = paths.data_root(cfg) / subset
    files = list_rep_files(rep_dir)
    weightings, bounds = aggregate_attn(paths.attn_root(cfg), model, subset,
                                        n_ranges=n_ranges, device=device)
    range_labels = [f"{lo}-{hi}" for lo, hi in bounds]

    # ── The sweep ────────────────────────────────────────────────────────────
    # Nesting is ordered by COST, outermost = most expensive to recompute, so that each
    # expensive artifact is built once and amortised over everything inside it:
    #
    #   seed          -> reloads representations and re-splits (seconds of I/O + GPU)
    #     base row    -> refits/reproduces the base score vector
    #       orient    -> a cheap elementwise map of that vector
    #         norm    -> a cheap per-trajectory affine map
    #           range -> selects which cached W set to use
    #             w   -> selects which cached W set to use
    #               strategy -> one matmul
    #                 gamma  -> NOT a loop: all gammas are one broadcast (see below)
    #
    # Two things deliberately sit OUTSIDE the row loop because they do not depend on the
    # base score at all: the attention aggregation (per model/subset) and the W matrices
    # (per seed x range x w). Rebuilding W inside the sweep — as a naive implementation
    # does — repeats the same ragged Python construction for every (row, orient, norm,
    # gamma) combination; hoisting it is the single biggest win in the stage.
    #
    # Gamma is not a loop level: discount/backprop are affine in gamma, so one matmul
    # yields the whole gamma curve at once and the metric pass scores all of them in a
    # single batched call. The output is still written LONG (one row per gamma) so the
    # sweep table keeps a flat, groupable schema.
    records = []
    for seed, seed_rows in base.groupby("seed"):
        parts = split_files(files, cfg["splits"], int(seed))
        load = lambda fl: load_representations(rep_dir, data_dir, poolings=poolings,
                                               files=fl, device=device)
        train, val, test = load(parts["train"]), load(parts["val"]), load(parts["test"])
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        val_WC = WCache(weightings, val.keeper, ws, device=device)
        test_WC = WCache(weightings, test.keeper, ws, device=device)
        fit_cache: dict = {}

        for _, row in tqdm(list(seed_rows.iterrows()), desc=f"{model}/{subset} s{seed}", leave=False):
            pooling, position, method = row["pooling"], row["position"], row["method"]
            cb, ce = int(row["c_begin"]), int(row["c_end"])
            cen, wt = bool(row["centered"]), bool(row["weighted"])

            def _fit(pos):
                k = (pooling, pos)
                if k not in fit_cache:
                    fit_cache[k] = fit_one(train.stores[k].R, N_COMPONENTS)
                return fit_cache[k]

            if position == ENSEMBLE_POSITION:
                members = member_positions(train.positions())
                fits = {p: _fit(p) for p in members}
                trainR = {p: train.stores[(pooling, p)].R for p in members}
                s_val = ens_score_vec(method, cb, ce, cen, wt, members, fits, trainR,
                                      {p: val.stores[(pooling, p)].R for p in members})
                s_test = ens_score_vec(method, cb, ce, cen, wt, members, fits, trainR,
                                       {p: test.stores[(pooling, p)].R for p in members})
            else:
                entry = _fit(position)
                s_val = score_config(val.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
                s_test = score_config(test.stores[(pooling, position)].R, entry, method, cb, ce, cen, wt)
            is_ens = position == ENSEMBLE_POSITION
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
                                # (N, G) -> transpose to (G, N): gammas are the batch dim.
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
        del train, val, test, val_WC, test_WC
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)[SWEEP_COLS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    rec.add_output(out)
    print(f"  wrote {out}  ({len(df)} rows)")


def run(cfg) -> None:
    with RunTimer(cfg, "rescore") as rec:
        rec.note(gammas=cfg["gammas"], ws=cfg["ws"], orients=cfg["orients"],
                 score_norms=cfg.get("score_norms", ["none"]),
                 strategies=cfg.get("strategies", ["discount"]))
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                if cfg.get("dry_run"):
                    print(f"[dry] rescore {model}/{subset}")
                    continue
                run_pair(cfg, model, subset, rec)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
