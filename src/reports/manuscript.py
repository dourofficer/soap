"""Manuscript-shaped main tables from the triples protocol, one per GT setting.

Emits manuscript tables 1 (without-GT) and 2 (with-GT) as TSV grids: per backbone,
Step-level / Agent-level accuracy across the five manuscript columns (WW-AG, WW-HC,
CE = correct-error macro-average, TE-Cap, TE-Mag), with rows for the prompting
baselines (all-at-once / step-by-step / binary-search per judge in ``--baselines``)
and the SOAP row computed by ``--strategy``. ``--baselines ''`` drops the prompting
rows (SOAP-only tables, different output stems).

Per (backbone, column) cell the reported window follows the manuscript's
seed-selection rule — among the cell's windows prefer those where the strategy
strictly beats SVD (proj); largest strategy step-acc, tiebreak larger diff, then
earliest window — and EVERY row of the cell (prompting included) is evaluated on
that same window's seed triple. Judge predictions are read from
``outputs[-gt]/<ds>/baselines/prompting/<judge>/<subset>/`` (imported by
``scripts/import_prompting.py``); the split universe is the PLAIN tree's rep file
list (the GT tree mirrors it exactly — check_gt_parity). Missing protocol runs or
predictions leave blank cells and are listed. A companion ``*_selection.tsv``
records every reported cell's window seeds (and, for SOAP rows, hyperparameters).

    # from repo root, after the protocol runs
    python -m src.reports.manuscript --strategy backprop                 # main tables
    python -m src.reports.manuscript --strategy succ-near --baselines '' # SOAP-only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml

from .baselines import SPLIT_MODEL, SUBSET_DISPLAY, baseline_cell

REPO = Path(__file__).resolve().parents[2]
TAG = "325"

COLUMNS = [
    ("WW-AG",  "ww",            "algorithm-generated"),
    ("WW-HC",  "ww",            "hand-crafted"),
    ("CE",     "correct-error", "average"),
    ("TE-Cap", "traceelephant", "captain"),
    ("TE-Mag", "traceelephant", "magentic"),
]
MODELS = [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")]
METHODS = [("All-at-once", "all_at_once"), ("Step-by-step", "step_by_step"),
           ("Binary search", "binary_search")]
HPARAM_COLS = ["pooling", "method", "centered", "weighted", "direction",
               "position", "c_begin", "c_end",
               "strategy", "orient", "score_norm", "layer_range", "gamma", "w"]
TABLES = [("table1", "outputs", "without-GT"), ("table2", "outputs-gt", "with-GT")]


def load_tsv(root: Path, dataset: str, name: str) -> pd.DataFrame | None:
    path = root / dataset / "tables" / TAG / name
    return pd.read_csv(path, sep="\t") if path.exists() else None


def pick_window(summary, model, subset, step_col, agent_col):
    """The manuscript seed-selection rule for one cell (see module docstring)."""
    cell = summary[(summary["model"] == model) & (summary["subset"] == subset)]
    cell = cell.dropna(subset=["svd_step", step_col]).copy()
    if cell.empty:
        return None
    cell["diff"] = cell[step_col] - cell["svd_step"]
    positive = cell[cell["diff"] > 0]
    pool = (positive if not positive.empty else cell).copy()
    pool["first_seed"] = pool["seeds"].str.split(",").str[0].astype(int)
    pool = pool.sort_values([step_col, "diff", "first_seed"],
                            ascending=[False, False, True], kind="mergesort")
    r = pool.iloc[0]
    return {"seeds": r["seeds"], "step": float(r[step_col]),
            "agent": float(r[agent_col]), "diff": float(r["diff"])}


def _plain_reps(ds: str, subset: str) -> list[str]:
    """Canonical split universe: the plain tree's rep file names, numerically sorted."""
    d = REPO / "outputs" / ds / "activations" / SPLIT_MODEL / subset
    return [p.name for p in sorted(d.glob("*.safetensors"), key=lambda p: int(p.stem))]


def _judge_cell(root: Path, ds: str, subset: str, judge: str, method: str,
                seeds: list[int]):
    """(step, agent) of one judge x method on the window seeds; CE 'average' is the
    macro-average over the 7 subsets (all must be present)."""
    man = yaml.safe_load(open(REPO / "configs" / "datasets" / f"{ds}.yaml"))
    cfg = {"outputs_base": str(root / ds), "splits": man["splits"]}
    subsets = ([sk for _, sk in SUBSET_DISPLAY[ds]] if subset == "average" else [subset])
    steps, agents = [], []
    for sk in subsets:
        acc = baseline_cell(cfg, judge, sk, "prompting", method, seeds, _plain_reps(ds, sk))
        if acc is None:
            return None
        steps.append(acc[0])
        agents.append(acc[1])
    return sum(steps) / len(steps), sum(agents) / len(agents)


def build(root: Path, strategy: str, judges: list[str]):
    """One table: grid rows + per-cell selection bookkeeping + missing-cell list."""
    scol = strategy.replace("-", "_")
    step_col, agent_col = f"{scol}_step", f"{scol}_agent"
    grid, book, missing = {}, [], []
    for _, ds, subset in COLUMNS:
        summary = load_tsv(root, ds, "triples_summary.tsv")
        selection = load_tsv(root, ds, "triples_selection.tsv")
        for _, mk in MODELS:
            cell = (pick_window(summary, mk, subset, step_col, agent_col)
                    if summary is not None and step_col in summary.columns else None)
            grid[(mk, ds, subset, "SOAP")] = cell
            if cell is None:
                missing.append(f"{mk}/{ds}/{subset}")
                continue
            seeds = [int(s) for s in cell["seeds"].split(",")]
            hp = {}
            if selection is not None and subset != "average":
                hit = selection[(selection["model"] == mk) & (selection["subset"] == subset)
                                & (selection["seeds"] == cell["seeds"])
                                & (selection["row"] == strategy)]
                if not hit.empty:
                    hp = {c: hit.iloc[0][c] for c in HPARAM_COLS}
            book.append({"model": mk, "dataset": ds, "subset": subset,
                         "row": f"SOAP ({strategy})", "seeds": cell["seeds"],
                         "step_acc_test": f"{cell['step']:.4f}",
                         "agent_acc_test": f"{cell['agent']:.4f}",
                         "diff_vs_svd": f"{cell['diff']:.4f}", **hp})
            for judge in judges:
                for label, method in METHODS:
                    acc = _judge_cell(root, ds, subset, judge, method, seeds)
                    grid[(mk, ds, subset, f"{label} ({judge})")] = (
                        {"step": acc[0], "agent": acc[1]} if acc else None)
                    if acc:
                        book.append({"model": mk, "dataset": ds, "subset": subset,
                                     "row": f"{label} ({judge})", "seeds": cell["seeds"],
                                     "step_acc_test": f"{acc[0]:.4f}",
                                     "agent_acc_test": f"{acc[1]:.4f}"})
    return grid, book, missing


def write_table(out_dir: Path, stem: str, grid, book, strategy: str,
                judges: list[str]) -> None:
    rows = [f"{label} ({j})" for label, _ in METHODS for j in judges] + ["SOAP"]
    display = {"SOAP": f"SOAP ({strategy})"}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Backbone", "Method", "Metric"] + [c for c, _, _ in COLUMNS])
        for disp, mk in MODELS:
            for row in rows:
                for metric, key in (("Step-level", "step"), ("Agent-level", "agent")):
                    w.writerow([disp, display.get(row, row), metric]
                               + [f"{grid[k][key]:.4f}"
                                  if (k := (mk, ds, sub, row)) in grid and grid[k]
                                  else "" for _, ds, sub in COLUMNS])
    if book:
        pd.DataFrame(book).to_csv(out_dir / f"{stem}_selection.tsv", sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="backprop",
                   choices=["backprop", "succ-strong", "succ-near"])
    p.add_argument("--baselines", default="gpt-5,gpt-4o",
                   help="comma-separated judge dirs under baselines/prompting/ "
                        "('' = SOAP-only tables)")
    p.add_argument("--out-dir", default=str(REPO / "outputs" / "manuscript-tables"))
    args = p.parse_args()
    judges = [j for j in args.baselines.split(",") if j]

    out_dir = Path(args.out_dir)
    for stem, sub, setting in TABLES:
        grid, book, missing = build(REPO / sub, args.strategy, judges)
        gt_part = "_main_gt" if sub == "outputs-gt" else "_main"
        name = (f"{stem}{gt_part}_{args.strategy}" if judges
                else f"{stem}_soap{'_gt' if sub == 'outputs-gt' else ''}_{args.strategy}")
        write_table(out_dir, name, grid, book, args.strategy, judges)
        note = f" (missing: {', '.join(missing)})" if missing else ""
        print(f"[manuscript] {setting}: {out_dir / (name + '.tsv')}{note}")


if __name__ == "__main__":
    main()
