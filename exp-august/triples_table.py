"""Seed-window ("triples") protocol (exp-august) — every consecutive seed window.

Alongside the shared-config protocol (focused_table.py), this runs the SAME
selection for EVERY consecutive window of ``triples.window`` seeds over
``triples.seeds`` (both declared in exp-august/configs/<ds>.yaml), with the
window's seeds FIXED instead of chosen:

Per (model, subset) x window [i, i+1, i+2]:

1. SVD (proj): filter score rows by ``base.fixed``; pick the ``base.swept`` config
   maximizing mean test step-acc over the window's seeds (config must exist in all
   of them; tiebreak agent acc, then highest config key).
2. SOAP (full): filter the window sweep (run_rescore_triples.py) by
   ``rescore.strategy`` + ``rescore.fixed`` + the window's chosen base config;
   pick the ``rescore.swept`` config the same way. Empty until the sweep exists.
3. Baselines: scored from the recorded prediction JSONLs on the window's seeds.

Selection is test-selected (optimistic). Outputs (exp-august/outputs/<ds>/tables/<tag>/):

* ``triples_selection.tsv`` — one row per (model, subset, window, method) with ALL
  hyperparameters (fixed and swept), the window's seeds, and the accuracies.
* ``triples_summary.tsv`` — one row per (model, subset, window): SVD and SOAP
  step-acc side by side with their difference, the two criteria for choosing seeds.

    # from repo root
    python exp-august/triples_table.py --config exp-august/configs/ww.yaml
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from focused_table import (EXP, ROWS, exp_out, arts_cfg, select_shared, baseline_cell)

from src.common import paths
from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.stores import list_rep_files
from src.reports.main_table import SPLIT_MODEL, MODEL_DISPLAY, SUBSET_DISPLAY, _fmt
from src.reports.reduce import _load_scores


def windows(seeds: list[int], k: int) -> list[list[int]]:
    return [seeds[i:i + k] for i in range(len(seeds) - k + 1)]


def run(cfg) -> None:
    dataset = cfg.get("dataset") or cfg.get("name")
    subsets = SUBSET_DISPLAY[dataset]
    bf, bs = cfg["base"]["fixed"], cfg["base"]["swept"]
    rc = cfg["rescore"]
    wins = windows(cfg["triples"]["seeds"], cfg["triples"].get("window", 3))
    cfg_cols = list(bf) + bs + ["strategy"] + list(rc["fixed"]) + rc["swept"]
    rec_cfg = {**cfg, "outputs_base": str(exp_out(cfg))}   # provenance inside exp-august

    with RunTimer(rec_cfg, "tables") as rec:
        selection, summary = [], []
        avg = {}   # (model, seeds) -> [(svd_step, soap_step, svd_agent, soap_agent)]
        for _, mk in MODEL_DISPLAY:
            for _, sk in subsets:
                scores = _load_scores({**arts_cfg(cfg), "seeds": cfg["triples"]["seeds"]},
                                      mk, sk)
                sweep_f = (exp_out(cfg) / "rescore" / paths.tag(cfg) / mk / sk
                           / "sweep_triples.tsv")
                sweep = pd.read_csv(sweep_f, sep="\t") if sweep_f.exists() else None
                if sweep is not None:
                    sweep = sweep[sweep["strategy"] == rc["strategy"]]
                # gt-aware artifact tree; baseline_cell below keeps the plain cfg
                reps_files = list_rep_files(paths.reps_root(arts_cfg(cfg)) / SPLIT_MODEL / sk)

                for win in wins:
                    cells = {}
                    svd = (select_shared(scores, bf, bs, "step_acc_test",
                                         "agent_acc_test", seeds=win)
                           if scores is not None else None)
                    cells["SVD (proj)"] = svd
                    soap = None
                    if svd is not None and sweep is not None:
                        sw = sweep
                        for ax in bs:              # this window's chosen base config
                            sw = sw[sw[ax] == svd["config"][ax]]
                        soap = select_shared(sw, rc["fixed"], rc["swept"],
                                             "disc_step_acc_test",
                                             "disc_agent_acc_test", seeds=win)
                        if soap is not None:
                            soap["config"] = {**svd["config"],
                                              "strategy": rc["strategy"],
                                              **soap["config"]}
                    cells["SOAP (full)"] = soap
                    for label in ROWS[:-2]:        # the baselines
                        cells[label] = (baseline_cell(cfg, dataset, mk, sk, label,
                                                      win, reps_files)
                                        if svd is not None else None)

                    seeds_str = ",".join(map(str, win))
                    if cfg.get("average_subsets") and svd and soap:
                        avg.setdefault((mk, seeds_str), []).append(
                            (svd["step"], soap["step"], svd["agent"], soap["agent"]))
                    for label in ROWS:
                        sel = cells[label]
                        if sel:
                            selection.append(
                                [mk, sk, seeds_str, label]
                                + [str(sel["config"].get(c, "")) for c in cfg_cols]
                                + [_fmt(sel["step"]), _fmt(sel["agent"])])
                    if svd:
                        summary.append([mk, sk, seeds_str, _fmt(svd["step"]),
                                        _fmt(soap["step"]) if soap else "",
                                        _fmt(soap["step"] - svd["step"]) if soap else "",
                                        _fmt(svd["agent"]),
                                        _fmt(soap["agent"]) if soap else ""])

        # Macro-average rows (config-gated): one subset="average" row per (model,
        # window), every metric averaged over the dataset's subsets with equal weight.
        # A window only gets a row when ALL subsets have both SVD and SOAP for it.
        if cfg.get("average_subsets"):
            for (mk, seeds_str), vals in avg.items():
                if len(vals) != len(subsets):
                    continue
                sv, so, sva, soa = (sum(v[i] for v in vals) / len(vals) for i in range(4))
                summary.append([mk, "average", seeds_str, _fmt(sv), _fmt(so),
                                _fmt(so - sv), _fmt(sva), _fmt(soa)])

        out = exp_out(cfg) / "tables" / paths.tag(cfg)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "triples_selection.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds", "row"] + cfg_cols
                       + ["step_acc_test", "agent_acc_test"])
            w.writerows(selection)
        with open(out / "triples_summary.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds", "svd_step", "soap_step",
                        "diff_step", "svd_agent", "soap_agent"])
            w.writerows(summary)
        rec.add_output(out / "triples_selection.tsv")
        rec.add_output(out / "triples_summary.tsv")
        print(f"[triples_table] {dataset} windows={len(wins)} -> {out/'triples_summary.tsv'}")


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
