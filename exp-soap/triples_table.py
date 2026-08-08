"""Seed-window ("triples") comparison table for the successor-side variants (exp-soap).

Per (model, subset) x window:

1. SVD (proj) and SOAP (full) are COPIED from exp-august's ``triples_selection.tsv``
   (protocol 2's frozen numbers) — any drift would be a wiring bug, not a result.
2. Each succ variant is selected from its exp-soap sweep exactly the way SOAP (full)
   was: filter by ``rescore.fixed`` + the window's chosen base config, then argmax of
   mean test step-acc over the window's seeds for ``rescore.swept``. Test-selected
   (optimistic), matching the protocol.

Outputs (exp-soap/outputs/<ds>/tables/<tag>/):

* ``triples_succ_selection.tsv`` — one row per (model, subset, window, method) with
  ALL hyperparameters (fixed and swept), the window's seeds, and the accuracies.
* ``triples_succ_summary.tsv``  — one row per (model, subset, window): SVD / SOAP /
  succ-strong / succ-near step-acc side by side with each variant's diff vs SOAP,
  plus the agent-acc columns. With ``average_subsets: true``: macro-average rows.

    # from repo root, after run_rescore_triples.py
    python exp-soap/triples_table.py --config exp-soap/configs/ww.yaml
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP))

import pandas as pd

from run_rescore_triples import exp_out, aug_out          # importing applies the path patch

from src.common import paths
from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.reports.main_table import MODEL_DISPLAY, SUBSET_DISPLAY, _fmt

SUCC_ROWS = ["succ-strong", "succ-near"]


def windows(seeds: list[int], k: int) -> list[list[int]]:
    return [seeds[i:i + k] for i in range(len(seeds) - k + 1)]


def _norm(v) -> str:
    """String form that survives pandas' nullable-int-to-float round-trip
    (aug selection columns with blank baseline rows read as float: 9 -> '9.0')."""
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


def select_shared(df, fixed, swept, step_col, agent_col, seeds):
    """One shared `swept` config over fixed seeds (copied from
    exp-august/focused_table.py, seeds-given branch). Tiebreak: agent acc, then the
    HIGHEST config key."""
    for col, val in fixed.items():
        df = df[df[col].astype(str) == str(val)]
    df = df[df["seed"].isin(seeds)]
    best = None
    for _, g in df.groupby(swept, sort=True):          # sorted keys -> deterministic ties
        if len(g) < len(seeds):
            continue
        cand = {"config": {**fixed, **{ax: g.iloc[0][ax] for ax in swept}},
                "step": float(g[step_col].mean()),
                "agent": float(g[agent_col].mean())}
        if best is None or (cand["step"], cand["agent"]) >= (best["step"], best["agent"]):
            best = cand
    return best


def run(cfg) -> None:
    dataset = cfg.get("dataset") or cfg.get("name")
    subsets = SUBSET_DISPLAY[dataset]
    bf, bs = cfg["base"]["fixed"], cfg["base"]["swept"]
    rc = cfg["rescore"]
    wins = windows(cfg["triples"]["seeds"], cfg["triples"].get("window", 3))
    cfg_cols = list(bf) + bs + ["strategy"] + list(rc["fixed"]) + rc["swept"]
    rec_cfg = {**cfg, "outputs_base": str(exp_out(cfg))}

    aug_sel = pd.read_csv(aug_out(cfg) / "tables" / paths.tag(cfg) / "triples_selection.tsv",
                          sep="\t")

    with RunTimer(rec_cfg, "tables") as rec:
        selection, summary = [], []
        avg = {}   # (model, seeds_str) -> [(svd, soap, strong, near, agents...)]
        for _, mk in MODEL_DISPLAY:
            for _, sk in subsets:
                cell_sel = aug_sel[(aug_sel["model"] == mk) & (aug_sel["subset"] == sk)]
                sweeps = {}
                f = (exp_out(cfg) / "rescore" / paths.tag(cfg) / mk / sk
                     / "sweep_triples-succ.tsv")
                if f.exists():
                    df = pd.read_csv(f, sep="\t")
                    sweeps = {strat: df[df["strategy"] == strat] for strat in SUCC_ROWS}

                for win in wins:
                    seeds_str = ",".join(map(str, win))
                    rows = cell_sel[cell_sel["seeds"] == seeds_str].set_index("row")

                    def _copied(label):
                        if label not in rows.index:
                            return None
                        r = rows.loc[label]
                        return {"config": {c: r[c] for c in cfg_cols if c in rows.columns},
                                "step": float(r["step_acc_test"]),
                                "agent": float(r["agent_acc_test"])}

                    cells = {"SVD (proj)": _copied("SVD (proj)"),
                             "SOAP (full)": _copied("SOAP (full)")}
                    svd = cells["SVD (proj)"]
                    for strat in SUCC_ROWS:
                        sel = None
                        if svd is not None and strat in sweeps:
                            sw = sweeps[strat]
                            for ax in bs:              # this window's chosen base config
                                sw = sw[sw[ax].astype(str).map(_norm)
                                        == _norm(svd["config"][ax])]
                            sel = select_shared(sw, rc["fixed"], rc["swept"],
                                                "disc_step_acc_test",
                                                "disc_agent_acc_test", seeds=win)
                            if sel is not None:
                                sel["config"] = {**{k: svd["config"].get(k, "") for k in
                                                    list(bf) + bs},
                                                 "strategy": strat, **sel["config"]}
                        cells[strat] = sel

                    for label in ["SVD (proj)", "SOAP (full)"] + SUCC_ROWS:
                        sel = cells[label]
                        if sel:
                            selection.append(
                                [mk, sk, seeds_str, label]
                                + ["" if pd.isna(sel["config"].get(c, ""))
                                   else _norm(sel["config"].get(c, "")) for c in cfg_cols]
                                + [_fmt(sel["step"]), _fmt(sel["agent"])])

                    svd, soap = cells["SVD (proj)"], cells["SOAP (full)"]
                    strong, near = cells["succ-strong"], cells["succ-near"]
                    if svd:
                        summary.append(
                            [mk, sk, seeds_str, _fmt(svd["step"]),
                             _fmt(soap["step"]) if soap else "",
                             _fmt(strong["step"]) if strong else "",
                             _fmt(near["step"]) if near else "",
                             _fmt(strong["step"] - soap["step"]) if strong and soap else "",
                             _fmt(near["step"] - soap["step"]) if near and soap else "",
                             _fmt(svd["agent"]),
                             _fmt(soap["agent"]) if soap else "",
                             _fmt(strong["agent"]) if strong else "",
                             _fmt(near["agent"]) if near else ""])
                    if cfg.get("average_subsets") and svd and soap and strong and near:
                        avg.setdefault((mk, seeds_str), []).append(
                            (svd["step"], soap["step"], strong["step"], near["step"],
                             svd["agent"], soap["agent"], strong["agent"], near["agent"]))

        # Macro-average rows: only when ALL subsets have all four methods for the window.
        if cfg.get("average_subsets"):
            for (mk, seeds_str), vals in avg.items():
                if len(vals) != len(subsets):
                    continue
                m = [sum(v[i] for v in vals) / len(vals) for i in range(8)]
                summary.append([mk, "average", seeds_str, _fmt(m[0]), _fmt(m[1]),
                                _fmt(m[2]), _fmt(m[3]), _fmt(m[2] - m[1]),
                                _fmt(m[3] - m[1]), _fmt(m[4]), _fmt(m[5]),
                                _fmt(m[6]), _fmt(m[7])])

        out = exp_out(cfg) / "tables" / paths.tag(cfg)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "triples_succ_selection.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds", "row"] + cfg_cols
                       + ["step_acc_test", "agent_acc_test"])
            w.writerows(selection)
        with open(out / "triples_succ_summary.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["model", "subset", "seeds", "svd_step", "soap_step",
                        "succ_strong_step", "succ_near_step",
                        "d_strong_step", "d_near_step",
                        "svd_agent", "soap_agent", "succ_strong_agent", "succ_near_agent"])
            w.writerows(summary)
        rec.add_output(out / "triples_succ_selection.tsv")
        rec.add_output(out / "triples_succ_summary.tsv")
        print(f"[triples_table] {dataset} windows={len(wins)} -> "
              f"{out / 'triples_succ_summary.tsv'}")


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
