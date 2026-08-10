"""The selection protocol: seed-window ("triples") selection + comparison tables.

For EVERY consecutive window of ``triples.window`` seeds over ``triples.seeds``
(declared in ``configs/protocol/<ds>.yaml``), per (model, subset):

1. SVD (proj): filter the recorded score rows by ``base.fixed``; pick the
   ``base.swept`` config maximizing mean test step-acc over the window's seeds
   (config must exist in all of them; tiebreak agent acc, then highest config key).
2. Rescoring rows, one per strategy in ``rescore.strategies`` (backprop /
   succ-strong / succ-near): filter the sweep (``src.rescore.run``) by the strategy,
   ``rescore.fixed`` and the window's chosen base config; pick the ``rescore.swept``
   config the same way. Empty until the sweep exists — the stage is TWO-PASS:
   run it once before rescoring (its SVD rows tell the rescore stage which base
   configs to sweep) and once after (fills the rescore rows).
3. Baselines: scored from the recorded prediction JSONLs on the window's seeds.

Selection is test-selected over the window (the protocol's convention). Outputs
(``tables/<tag>/``):

* ``triples_selection.tsv`` — one row per (model, subset, window, method) with ALL
  hyperparameters (fixed and swept), the window's seeds, and the accuracies.
* ``triples_summary.tsv``  — one row per (model, subset, window): SVD / backprop /
  succ-strong / succ-near step-acc side by side, backprop's diff vs SVD and each
  succ variant's diff vs backprop, plus the agent columns. With
  ``average_subsets: true``, per-(model, window) macro-average rows over subsets.

    # from repo root — before AND after `python -m src.rescore.run`
    python -m src.reports.triples --config configs/protocol/<ds>.yaml
"""
from __future__ import annotations

import csv

import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..stores import list_rep_files
from .baselines import (SPLIT_MODEL, MODEL_DISPLAY, SUBSET_DISPLAY, ROW_TO_PRED,
                        load_scores, baseline_cell, fmt)

BASE_ROW = "SVD (proj)"


def windows(seeds: list[int], k: int) -> list[list[int]]:
    return [seeds[i:i + k] for i in range(len(seeds) - k + 1)]


def norm_val(v) -> str:
    """String form that survives pandas' nullable-int-to-float round-trip
    (selection columns with blank baseline rows read back as float: 9 -> '9.0')."""
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


def select_shared(df, fixed, swept, step_col, agent_col, seeds):
    """One `swept` config over fixed ``seeds``: filter by `fixed`, then argmax of the
    mean step metric (config must be present in every seed). Tiebreak: agent acc,
    then the HIGHEST config key (sorted groupby -> deterministic ties)."""
    for col, val in fixed.items():
        df = df[df[col].astype(str).map(norm_val) == norm_val(val)]
    df = df[df["seed"].isin(seeds)]
    best = None
    for _, g in df.groupby(swept, sort=True):
        if len(g) < len(seeds):
            continue
        cand = {"config": {**fixed, **{ax: g.iloc[0][ax] for ax in swept}},
                "step": float(g[step_col].mean()),
                "agent": float(g[agent_col].mean())}
        # >= : on full ties the later (highest) config key wins
        if best is None or (cand["step"], cand["agent"]) >= (best["step"], best["agent"]):
            best = cand
    return best


def run(cfg) -> None:
    dataset = cfg.get("dataset") or cfg.get("name")
    subsets = SUBSET_DISPLAY[dataset]
    bf, bs = cfg["base"]["fixed"], cfg["base"]["swept"]
    rc = cfg["rescore"]
    strategies = list(rc["strategies"])
    wins = windows(cfg["triples"]["seeds"], cfg["triples"].get("window", 3))
    cfg_cols = list(bf) + bs + ["strategy"] + list(rc["fixed"]) + rc["swept"]
    baseline_rows = [r for r in ROW_TO_PRED]

    with RunTimer(cfg, "tables") as rec:
        selection, summary = [], []
        avg = {}   # (model, seeds_str) -> list of per-subset metric tuples
        for _, mk in MODEL_DISPLAY:
            for _, sk in subsets:
                # the protocol's seed universe is triples.seeds, not the manifest's
                # (possibly narrower) `seeds` list load_scores would default to
                scores = load_scores({**cfg, "seeds": cfg["triples"]["seeds"]}, mk, sk)
                sweep_f = paths.rescore_root(cfg) / mk / sk / "sweep.tsv"
                sweep = pd.read_csv(sweep_f, sep="\t") if sweep_f.exists() else None
                reps_files = list_rep_files(paths.reps_root(cfg) / SPLIT_MODEL / sk)

                for win in wins:
                    seeds_str = ",".join(map(str, win))
                    cells = {}
                    svd = (select_shared(scores, bf, bs, "step_acc_test",
                                         "agent_acc_test", seeds=win)
                           if scores is not None else None)
                    cells[BASE_ROW] = svd
                    for strat in strategies:
                        sel = None
                        if svd is not None and sweep is not None:
                            sw = sweep[sweep["strategy"] == strat]
                            for ax in bs:          # this window's chosen base config
                                sw = sw[sw[ax].astype(str).map(norm_val)
                                        == norm_val(svd["config"][ax])]
                            sel = select_shared(sw, rc["fixed"], rc["swept"],
                                                "disc_step_acc_test",
                                                "disc_agent_acc_test", seeds=win)
                            if sel is not None:
                                sel["config"] = {**svd["config"], "strategy": strat,
                                                 **sel["config"]}
                        cells[strat] = sel
                    for label in baseline_rows:
                        root, method = ROW_TO_PRED[label]
                        acc = (baseline_cell(cfg, mk, sk, root, method, win, reps_files)
                               if svd is not None else None)
                        cells[label] = ({"config": {}, "step": acc[0], "agent": acc[1]}
                                        if acc else None)

                    for label in baseline_rows + [BASE_ROW] + strategies:
                        sel = cells[label]
                        if sel:
                            selection.append(
                                [mk, sk, seeds_str, label]
                                + [norm_val(sel["config"][c]) if c in sel["config"]
                                   else "" for c in cfg_cols]
                                + [fmt(sel["step"]), fmt(sel["agent"])])

                    bp = cells.get("backprop")
                    strong, near = cells.get("succ-strong"), cells.get("succ-near")
                    if svd:
                        summary.append(
                            [mk, sk, seeds_str, fmt(svd["step"]),
                             fmt(bp["step"]) if bp else "",
                             fmt(strong["step"]) if strong else "",
                             fmt(near["step"]) if near else "",
                             fmt(bp["step"] - svd["step"]) if bp else "",
                             fmt(strong["step"] - bp["step"]) if strong and bp else "",
                             fmt(near["step"] - bp["step"]) if near and bp else "",
                             fmt(svd["agent"]),
                             fmt(bp["agent"]) if bp else "",
                             fmt(strong["agent"]) if strong else "",
                             fmt(near["agent"]) if near else ""])
                    if cfg.get("average_subsets") and svd and bp and strong and near:
                        avg.setdefault((mk, seeds_str), []).append(
                            (svd["step"], bp["step"], strong["step"], near["step"],
                             svd["agent"], bp["agent"], strong["agent"], near["agent"]))

        # Macro-average rows: only when ALL subsets have all four method rows.
        if cfg.get("average_subsets"):
            for (mk, seeds_str), vals in avg.items():
                if len(vals) != len(subsets):
                    continue
                m = [sum(v[i] for v in vals) / len(vals) for i in range(8)]
                summary.append([mk, "average", seeds_str, fmt(m[0]), fmt(m[1]),
                                fmt(m[2]), fmt(m[3]), fmt(m[1] - m[0]),
                                fmt(m[2] - m[1]), fmt(m[3] - m[1]),
                                fmt(m[4]), fmt(m[5]), fmt(m[6]), fmt(m[7])])

        out = paths.tables_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "triples_selection.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds", "row"] + cfg_cols
                       + ["step_acc_test", "agent_acc_test"])
            w.writerows(selection)
        with open(out / "triples_summary.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds",
                        "svd_step", "backprop_step", "succ_strong_step", "succ_near_step",
                        "d_backprop_step", "d_strong_step", "d_near_step",
                        "svd_agent", "backprop_agent", "succ_strong_agent",
                        "succ_near_agent"])
            w.writerows(summary)
        rec.add_output(out / "triples_selection.tsv")
        rec.add_output(out / "triples_summary.tsv")
        print(f"[triples] {dataset} windows={len(wins)} -> {out / 'triples_summary.tsv'}")


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
