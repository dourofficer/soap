"""Score the representation-based baselines (OAT, StepFinder) on SOAP's frozen test splits.

The baselines live in `../attrib-prompting/outputs-rb-{nogt,gt}/`, one JSON per
trajectory with the same keys the prompting baselines write. This runner reuses
`scripts/prompting/evaluate.py` — same split code, same step/agent rules, a missing
prediction counts as wrong — so the numbers mean exactly what Tables 1–2 mean.

Two StepFinder families:
  * A (`stepfinder.s42..s46`) — trained on the paper's regenerated corpus; one model per
    TRAINING seed, predictions for every trajectory. Scored on the frozen triple, then
    averaged over the five training seeds. OAT (`oat.s42..s46`) is scored the same way.
  * B (`stepfinder.e<seed>`) — trained on split seed <seed>'s own 30% train partition;
    predictions only for that seed's val+test ids. Scored on seed <seed>'s test split
    where the frozen triple's seeds were trained (WW-AG, WW-HC, TE-Mag); elsewhere
    the cell is reported as unavailable.

Outputs, under results-ablations/b1_rb_baselines/:
  by_seed.tsv, by_cell.tsv, by_column.tsv   — families A + OAT, both GT settings
  family_b.tsv                               — family B, without-GT only
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from main.config import load_config, seeds_for  # noqa: E402

spec = importlib.util.spec_from_file_location("ev", REPO / "scripts/prompting/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OUT = REPO / "results-ablations" / "b1_rb_baselines"
TRAIN_SEEDS = range(42, 47)
COLS = ["WW-AG", "WW-HC", "CE", "TE-Cap", "TE-Mag"]


def family_a() -> None:
    ev.SETTINGS = [("outputs-rb-nogt", False, "results-nogt", ""),
                   ("outputs-rb-gt", True, "results-gt", "-gt")]
    ev.JUDGES = ["qwen3.5-9b", "deepseek-8b", "qwen3-embedding-0.6b"]
    ev.METHODS = [f"{m}.s{s}" for m in ("oat", "stepfinder") for s in TRAIN_SEEDS]
    df = ev.evaluate(ev.DATASETS)
    cell = ev.by_cell(df)
    col = ev.by_column(cell)
    for frame in (df, cell, col):
        frame["family"] = frame["method"].str.split(".").str[0]
        frame["train_seed"] = frame["method"].str.split(".s").str[1].astype(int)
    df.to_csv(OUT / "by_seed.tsv", sep="\t", index=False)
    cell.to_csv(OUT / "by_cell.tsv", sep="\t", index=False)
    col.to_csv(OUT / "by_column.tsv", sep="\t", index=False)

    agg = col.groupby(["with_gt", "judge", "family", "column"]).agg(
        step_mean=("step_acc", "mean"), step_min=("step_acc", "min"),
        step_max=("step_acc", "max"), agent_mean=("agent_acc", "mean"),
        n_runs=("method", "count")).reset_index()
    agg.to_csv(OUT / "by_column_mean_over_train_seeds.tsv", sep="\t", index=False)
    for val in ("step_mean", "step_max", "agent_mean"):
        piv = agg.pivot_table(index=["with_gt", "judge", "family"], columns="column",
                              values=val)[COLS] * 100
        print(f"\n=== {val} (%) ===\n{piv.round(2).to_string()}")
    assert cell[cell.judge != "qwen3-embedding-0.6b"].n_missing.sum() == 0


def family_b() -> None:
    rows = []
    for ds in ev.DATASETS:
        cfg = load_config(REPO / f"configs-main/{ds}.yaml")
        for subset in cfg["subsets"]:
            for seed in seeds_for(cfg, subset):
                preds = ev.read_cell("outputs-rb-nogt", ds, subset, "qwen3-embedding-0.6b",
                                     f"stepfinder.e{seed}")
                row = dict(dataset=ds, subset=subset, seed=seed, available=bool(preds))
                if preds:
                    ids = ev.test_ids(ds, subset, "results-nogt", cfg["splits"], seed)
                    row.update(ev.score(preds, ids))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "family_b.tsv", sep="\t", index=False)
    print("\n=== family B, frozen-triple mean (%) ===")
    print((df[df.available].groupby(["dataset", "subset"])[["step_acc", "agent_acc"]]
           .mean() * 100).round(2).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    family_a()
    family_b()
