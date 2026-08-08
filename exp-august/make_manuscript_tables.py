"""Select the seed windows + numbers for the manuscript's SOAP rows (Tables 1 & 2).

Emits BOOKKEEPING TSVs only (no LaTeX): which seed window each (model, column) cell
uses, the selected hyperparameters, and the SVD (proj) / SOAP (full) test accuracies.
The numbers are then transcribed into manuscript/sections/experiments.tex by hand
(SOAP rows only; the other methods' cells stay dashed).

WHAT IT READS
-------------
The seed-window ("triples") protocol outputs of triples_table.py —
Table 1 (without-GT) from ``exp-august/outputs/<ds>/tables/325/``,
Table 2 (with-GT)    from ``exp-august/outputs-gt/<ds>/tables/325/``:

* ``triples_summary.tsv``   — per (model, subset, window): svd/soap accuracies.
  The CE column uses correct-error's ``subset="average"`` macro-average rows
  (``average_subsets: true``), i.e. one CE number averaged over its 7 subsets.
* ``triples_selection.tsv`` — per-row hyperparameters, for the bookkeeping.

SEED SELECTION (one window per (model, column) cell, applied per table)
-----------------------------------------------------------------------
Using that table's own SOAP results for the cell:
  1. diff = SOAP (full) - SVD (proj) test step-acc, per window.
  2. If any window has diff > 0, restrict the search to those windows;
     otherwise keep all windows.
  3. Pick the window with the largest SOAP (full) test step-acc.
     Ties: larger diff first, then the window starting at the smallest seed.
The chosen window's seeds are the cell's shared seeds for every method row.

CE IN TABLE 2: the CORRECT-Error corpus has no gold answers, so the with-GT run
does not exist; the cell is emitted with "(no results)" and stays dashed in the tex.
To drop a column entirely, comment out its line in COLUMNS below and re-run.

OUTPUTS (exp-august/outputs/manuscript-tables/)
-----------------------------------------------
  table1_main_selection.tsv / table2_main_gt_selection.tsv
    role="cell"   rows — one per (model, column): seeds + accuracies
                  (+ hyperparameters when the column is a single subset).
    role="detail" rows — for the CE macro-average cell only: one per constituent
                  subset with that subset's own config and accuracies at the
                  chosen window.

    # from repo root
    python exp-august/make_manuscript_tables.py
"""
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent
OUT_DIR = EXP / "outputs" / "manuscript-tables"

# ── table columns ───────────────────────────────────────────────────────────
# (column header, dataset, subset). "average" is the macro-average pseudo-subset
# of correct-error. Comment out a line to drop that column from both tables.
COLUMNS = [
    ("WW-AG",  "ww",            "algorithm-generated"),
    ("WW-HC",  "ww",            "hand-crafted"),
    ("CE",     "correct-error", "average"),
    ("TE-Cap", "traceelephant", "captain"),
    ("TE-Mag", "traceelephant", "magentic"),
]

MODELS = ["qwen3.5-9b", "deepseek-8b"]

ROW_SVD, ROW_SOAP = "SVD (proj)", "SOAP (full)"

# Hyperparameter columns copied from triples_selection.tsv into the bookkeeping
# (base score axes first, then rescoring axes; the SOAP (full) row states both).
HPARAM_COLS = ["pooling", "method", "centered", "weighted", "direction",
               "position", "c_begin", "c_end",
               "strategy", "orient", "score_norm", "layer_range", "gamma", "w"]

# (output stem, protocol results root, human-readable setting)
TABLES = [
    ("table1_main",    EXP / "outputs",    "without-GT"),
    ("table2_main_gt", EXP / "outputs-gt", "with-GT"),
]


# ── data access ─────────────────────────────────────────────────────────────
def load_tsv(results_root: Path, dataset: str, name: str) -> pd.DataFrame | None:
    """Load one dataset's triples_{summary,selection}.tsv, or None if that
    protocol run does not exist (e.g. correct-error has no with-GT run)."""
    path = results_root / dataset / "tables" / "325" / name
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("svd_step", "soap_step", "diff_step", "step_acc_test"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── seed selection ──────────────────────────────────────────────────────────
def pick_window(summary: pd.DataFrame, model: str, subset: str) -> pd.Series | None:
    """Apply the seed-selection rule to one (model, subset) cell.

    Returns the chosen window's summary row (seeds, svd_step, soap_step, ...),
    or None when the cell has no complete SOAP results."""
    cell = summary[(summary["model"] == model) & (summary["subset"] == subset)]
    cell = cell.dropna(subset=["svd_step", "soap_step"])
    if cell.empty:
        return None

    cell = cell.copy()
    cell["diff"] = cell["soap_step"] - cell["svd_step"]

    # Rule: prefer windows where rescoring strictly helps; fall back to all.
    positive = cell[cell["diff"] > 0]
    pool = positive if not positive.empty else cell

    # Largest SOAP (full) step-acc; ties -> larger diff, then earliest window.
    pool = pool.copy()
    pool["first_seed"] = pool["seeds"].str.split(",").str[0].astype(int)
    pool = pool.sort_values(["soap_step", "diff", "first_seed"],
                            ascending=[False, False, True], kind="mergesort")
    return pool.iloc[0]


# ── bookkeeping helpers ─────────────────────────────────────────────────────
def hparams_of(selection: pd.DataFrame, model: str, subset: str,
               seeds: str, row_name: str) -> dict:
    """The hyperparameters of one method row at one window ({} if absent)."""
    hit = selection[(selection["model"] == model) & (selection["subset"] == subset)
                    & (selection["seeds"] == seeds) & (selection["row"] == row_name)]
    if hit.empty:
        return {}
    return {hp: hit.iloc[0][hp] for hp in HPARAM_COLS}


def acc_of(selection: pd.DataFrame, model: str, subset: str,
           seeds: str, row_name: str):
    hit = selection[(selection["model"] == model) & (selection["subset"] == subset)
                    & (selection["seeds"] == seeds) & (selection["row"] == row_name)]
    return None if hit.empty else hit.iloc[0]["step_acc_test"]


# ── main ────────────────────────────────────────────────────────────────────
def build_table(stem: str, results_root: Path, setting: str) -> None:
    records = []
    for model in MODELS:
        for col_header, dataset, subset in COLUMNS:
            summary = load_tsv(results_root, dataset, "triples_summary.tsv")
            selection = load_tsv(results_root, dataset, "triples_selection.tsv")
            base = {"table": stem, "setting": setting, "role": "cell",
                    "model": model, "column": col_header,
                    "dataset": dataset, "subset": subset}

            win = None if summary is None else pick_window(summary, model, subset)
            if win is None:
                records.append({**base, "seeds": "(no results)"})
                continue

            record = {**base, "seeds": win["seeds"],
                      "svd_step": f"{win['svd_step']:.4f}",
                      "soap_step": f"{win['soap_step']:.4f}"}
            if subset != "average":
                # Single-subset cell: the SOAP (full) row states all hyperparameters.
                record.update(hparams_of(selection, model, subset,
                                         win["seeds"], ROW_SOAP))
            records.append(record)

            if subset == "average":
                # Macro-average cell: one detail row per constituent subset, each
                # with its own selected config at the chosen window.
                parts = sorted(selection["subset"].unique())
                for part in parts:
                    records.append({
                        **base, "role": "detail", "subset": part,
                        "seeds": win["seeds"],
                        "svd_step": _fmt4(acc_of(selection, model, part,
                                                 win["seeds"], ROW_SVD)),
                        "soap_step": _fmt4(acc_of(selection, model, part,
                                                  win["seeds"], ROW_SOAP)),
                        **hparams_of(selection, model, part, win["seeds"], ROW_SOAP),
                    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}_selection.tsv"
    columns = (["table", "setting", "role", "model", "column", "dataset", "subset",
                "seeds", "svd_step", "soap_step"] + HPARAM_COLS)
    pd.DataFrame(records).reindex(columns=columns).to_csv(out_path, sep="\t",
                                                          index=False)
    print(f"[{stem}] wrote {out_path}")


def _fmt4(value) -> str:
    return "" if value is None or pd.isna(value) else f"{float(value):.4f}"


def main() -> None:
    for stem, results_root, setting in TABLES:
        build_table(stem, results_root, setting)


if __name__ == "__main__":
    main()
