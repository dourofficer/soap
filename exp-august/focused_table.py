"""Focused table (exp-august) — shared-config / shared-seed protocol, config-driven.

Which axes are FIXED (row filters) and which are SWEPT (selected) is declared in
``exp-august/configs/<ds>.yaml`` under ``base:`` (the SVD (proj) score) and
``rescore:`` (SOAP (full)) — nothing is hardcoded here.

Per (model, subset) cell:

1. SVD (proj): filter the recorded score rows by ``base.fixed``; choose the
   ``base.swept`` config and 3 seeds JOINTLY — per config, mean test step-acc over
   its own best-3 seeds; argmax (tiebreak agent acc, then lowest config key).
2. The winning config's 3 seeds are the cell's SHARED seeds for every other row.
3. SOAP (full): filter the sweep (from run_rescore.py) by ``rescore.strategy`` and
   ``rescore.fixed``; choose the ``rescore.swept`` config as argmax of mean test
   step-acc over the shared seeds. Empty until the sweep exists.
4. Baselines: scored from the recorded prediction JSONLs on the shared seeds.

``focused_selection.tsv`` states ALL hyperparameters per row, fixed and swept.
Selection is test-selected (optimistic), matching the rest of the table. Reads
scores/baselines from the main ``outputs/`` tree; writes into ``exp-august/outputs/``.

    # from repo root
    python exp-august/focused_table.py --config exp-august/configs/ww.yaml
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parent))            # repo root, so `import src.*` works

import pandas as pd

from src.common import paths
from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.stores import list_rep_files
from src.reports.main_table import (SPLIT_MODEL, MODEL_DISPLAY, SUBSET_DISPLAY,
                                    ROW_TO_PRED, _fmt, _baseline_cell)
from src.reports.reduce import _load_scores

ROWS = ["All-at-once", "Step-by-step", "Binary search", "CORRECT", "CHIEF",
        "SVD (proj)", "SOAP (full)"]


def exp_out(cfg) -> Path:
    """Protocol write root; ``gt: true`` in the config lands in exp-august/outputs-gt/."""
    sub = "outputs-gt" if cfg.get("gt") else "outputs"
    return EXP / sub / (cfg.get("dataset") or cfg["name"])


def arts_cfg(cfg) -> dict:
    """Cfg whose derived paths point at the extracted/scored artifact tree.

    Plain mode: unchanged (paths derive to the repo's outputs/<ds>). With ``gt: true``:
    the forward-pass artifacts and score files live under the repo's outputs-gt/<ds>.
    Baseline predictions always keep the PLAIN cfg — their GT-ness is a corpus
    property (ww/TE prompts embed the answer), not a knob of this experiment.
    """
    if not cfg.get("gt"):
        return cfg
    ds = cfg.get("dataset") or cfg["name"]
    return {**cfg, "outputs_base": str(EXP.parent / "outputs-gt" / ds)}


def select_shared(df, fixed, swept, step_col, agent_col, seeds=None, n_seeds=3):
    """One shared `swept` config: filter by `fixed`, then argmax of mean step-acc.

    With ``seeds`` given, average over exactly those seeds (all must be present);
    otherwise choose config and seeds jointly (each config's own best-n seeds).
    Tiebreak: agent acc, then the HIGHEST config key.
    """
    for col, val in fixed.items():
        df = df[df[col] == val]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]
    best = None
    for _, g in df.groupby(swept, sort=True):  # sorted keys -> deterministic ties
        if seeds is None:
            g = g.sort_values([step_col, agent_col, "seed"],
                              ascending=[False, False, True],
                              kind="mergesort").head(n_seeds)
        if len(g) < (n_seeds if seeds is None else len(seeds)):
            continue
        cand = {"config": {**fixed, **{ax: g.iloc[0][ax] for ax in swept}},
                "seeds": seeds or sorted(int(s) for s in g["seed"]),
                "step": float(g[step_col].mean()),
                "agent": float(g[agent_col].mean())}
        # >= : on full ties the later (highest) config key wins
        if best is None or (cand["step"], cand["agent"]) >= (best["step"], best["agent"]):
            best = cand
    return best


def svd_cell(cfg, model, subset):
    df = _load_scores(cfg, model, subset)      # seed-*.tsv concat, seeds kept, k==1
    if df is None:
        return None
    return select_shared(df, cfg["base"]["fixed"], cfg["base"]["swept"],
                         "step_acc_test", "agent_acc_test", n_seeds=cfg["n_seeds"])


def soap_full_cell(cfg, model, subset, svd):
    if svd is None:
        return None
    f = exp_out(cfg) / "rescore" / paths.tag(cfg) / model / subset / "sweep_focused.tsv"
    if not f.exists():                         # run_rescore.py not run yet
        return None
    rc = cfg["rescore"]
    df = pd.read_csv(f, sep="\t")
    df = df[df["strategy"] == rc["strategy"]]
    for ax in cfg["base"]["swept"]:            # guard against a stale sweep: only rows
        df = df[df[ax] == svd["config"][ax]]   # of the cell's current base config count
    sel = select_shared(df, rc["fixed"], rc["swept"],
                        "disc_step_acc_test", "disc_agent_acc_test", seeds=svd["seeds"])
    if sel is not None:
        sel["config"] = {**svd["config"], "strategy": rc["strategy"], **sel["config"]}
    return sel


def baseline_cell(cfg, dataset, model, subset, label, seeds, reps_files):
    root, method = ROW_TO_PRED[label]
    acc = _baseline_cell(cfg, dataset, model, subset, root, method, seeds, reps_files)
    if acc is None:                            # predictions missing for this cell
        return None
    return {"config": {}, "seeds": seeds, "step": acc[0], "agent": acc[1]}


def run(cfg) -> None:
    dataset = cfg.get("dataset") or cfg.get("name")
    subsets = SUBSET_DISPLAY[dataset]
    # every hyperparameter column, fixed and swept, in declaration order
    cfg_cols = (list(cfg["base"]["fixed"]) + cfg["base"]["swept"] + ["strategy"]
                + list(cfg["rescore"]["fixed"]) + cfg["rescore"]["swept"])
    rec_cfg = {**cfg, "outputs_base": str(exp_out(cfg))}   # provenance inside exp-august

    with RunTimer(rec_cfg, "tables") as rec:
        # Artifact reads (reps for split identity, recorded scores) follow the gt flag;
        # baseline_cell below deliberately keeps the plain cfg.
        reps_files = {sk: list_rep_files(paths.reps_root(arts_cfg(cfg)) / SPLIT_MODEL / sk)
                      for _, sk in subsets}

        # SVD (proj) selection first: it fixes each (model, subset)'s shared seeds.
        svd = {(mk, sk): svd_cell(arts_cfg(cfg), mk, sk)
               for _, mk in MODEL_DISPLAY for _, sk in subsets}

        header_a = ["Backbone", "Method"] + [x for disp, _ in subsets for x in (disp, "")]
        header_b = ["", ""] + ["Step-level", "Agent-level"] * len(subsets)
        lines, selection = [header_a, header_b], []

        for disp_model, mk in MODEL_DISPLAY:
            for i, label in enumerate(ROWS):
                cells = []
                for _, sk in subsets:
                    if label == "SVD (proj)":
                        sel = svd[(mk, sk)]
                    elif label == "SOAP (full)":
                        sel = soap_full_cell(cfg, mk, sk, svd[(mk, sk)])
                    elif svd[(mk, sk)] is None:    # no SVD scores -> no shared seeds
                        sel = None
                    else:
                        sel = baseline_cell(cfg, dataset, mk, sk, label,
                                            svd[(mk, sk)]["seeds"], reps_files[sk])
                    cells += [_fmt(sel["step"]), _fmt(sel["agent"])] if sel else ["", ""]
                    if sel:
                        selection.append([mk, sk, label]
                                         + [str(sel["config"].get(c, "")) for c in cfg_cols]
                                         + [",".join(map(str, sel["seeds"])),
                                            _fmt(sel["step"]), _fmt(sel["agent"])])
                lines.append([disp_model if i == 0 else "", label] + cells)

        out = exp_out(cfg) / "tables" / paths.tag(cfg)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results_focused.tsv", "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(lines)
        with open(out / "focused_selection.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "row"] + cfg_cols
                       + ["seeds", "step_acc_test", "agent_acc_test"])
            w.writerows(selection)
        rec.add_output(out / "results_focused.tsv")
        rec.add_output(out / "focused_selection.tsv")
        print(f"[focused_table] {dataset} -> {out/'results_focused.tsv'}")


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
