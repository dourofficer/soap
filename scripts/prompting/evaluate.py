"""Score the prompt-based baselines on the same test splits SOAP uses.

The manuscript scores these baselines on the whole dataset. SOAP scores on a test split of
three seeds. The two numbers then count different trajectories, so a reader cannot compare
them. This script removes that gap.

It reads predictions from `../attrib-prompting` and writes scores to `results-prompting/`.
It runs no model. The predictions already exist.

THE GT FOLDER NAMES LOOK BACKWARDS. Read this table before you touch a path:

    ../attrib-prompting   setting       soap tree        soap config
    outputs-nogt/         without GT    results-nogt/    configs-main/<ds>.yaml
    outputs/              with GT       results-gt/      configs-main/<ds>-gt.yaml

In soap, `outputs/` means the opposite. It is the frozen `src/` tree and we never read it.

THE SPLIT. One seed triple belongs to each subset, frozen in `configs-main/`. The two GT
settings hold different triples, so we read the config that matches the setting. We split
the same file list SOAP splits: the activation names under `results-{nogt,gt}/`, through
`main.stores.list_rep_files`. A cell's score is the mean over its three seeds.

THE RULES. All three come from SOAP, so the numbers mean the same thing:
  * step  -- int(pred) == int(gold). `gold_step` is a string in ww and an int elsewhere.
  * agent -- standardize_role(pred).lower() == gold.lower(). Note the asymmetry: SOAP
             standardises the prediction and only lowercases the gold. We keep it.
  * a missing or null prediction counts as WRONG. The divisor stays the split size.

    python scripts/prompting/evaluate.py
    python scripts/prompting/evaluate.py --dataset ww          # one dataset
    python scripts/prompting/evaluate.py --judge qwen3.5-9b    # one judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main.config import load_config, seeds_for                      # noqa: E402
from main.metrics import standardize_role                           # noqa: E402
from main.stores import list_rep_files, split_files                 # noqa: E402

SRC = REPO.parent / "attrib-prompting"
OUT = REPO / "results-prompting"

# (attrib-prompting root, with_gt, soap results tree, config suffix)
SETTINGS = [("outputs-nogt", False, "results-nogt", ""),
            ("outputs", True, "results-gt", "-gt")]

# The judge (for prompt-based methods, the model that reads the log). The two
# closed-source ones answer the manuscript's GPT-4o block; the two open ones are
# the backbones SOAP itself runs on, so their prompting rows sit in the same
# block as SOAP and share its test splits.
JUDGES = ["gpt-4o", "gpt-5", "qwen3.5-9b", "deepseek-8b"]
# gpt-5 x correct-error has no `chief` run — it was excluded as too costly, so
# that cell stays blank rather than missing.
METHODS = ["all_at_once", "step_by_step", "binary_search", "correct", "chief",
           "raffles"]
DATASETS = ["ww", "traceelephant", "correct-error"]
SPLIT_MODEL = "qwen3.5-9b"      # id source; every backbone has the same file list

# The five manuscript columns. CE is the mean of its seven subsets.
COLUMNS = [("WW-AG", "ww", "algorithm-generated"), ("WW-HC", "ww", "hand-crafted"),
           ("CE", "correct-error", None), ("TE-Cap", "traceelephant", "captain"),
           ("TE-Mag", "traceelephant", "magentic")]


# ── the rules ───────────────────────────────────────────────────────────────
def step_hit(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return int(pred) == int(gold)
    except (TypeError, ValueError):
        return False


def agent_hit(pred, gold) -> bool:
    """SOAP's rule. It standardises the prediction and only lowercases the gold."""
    if pred is None or gold is None:
        return False
    g = str(gold).strip().lower()
    return bool(g) and standardize_role(str(pred)).strip().lower() == g


def agent_hit_substring(pred, gold) -> bool:
    """The older rule, kept for one diagnostic print. We do not report it."""
    if pred is None or gold is None:
        return False
    p = standardize_role(str(pred)).strip().lower()
    g = standardize_role(str(gold)).strip().lower()
    return bool(g) and (p == g or g in p)


# ── reading ─────────────────────────────────────────────────────────────────
def read_cell(root: str, ds: str, subset: str, judge: str, method: str) -> dict[str, dict]:
    """Every prediction of one cell, keyed by its id string.

    Ids are file stems. TraceElephant stems start at 0, the rest at 1, so we never build
    a range. Files whose stem is not all digits are run records, not predictions.
    """
    d = SRC / root / ds / subset / judge / method
    if not d.is_dir():
        return {}
    out = {}
    for p in d.glob("*.json"):
        if not p.stem.isdigit():
            continue
        out[p.stem] = json.loads(p.read_text())
    return out


def test_ids(ds: str, subset: str, tree: str, splits: dict, seed: int) -> list[str]:
    """The stems of SOAP's test split — the same call SOAP makes."""
    rep_dir = REPO / tree / ds / "activations" / SPLIT_MODEL / subset
    files = list_rep_files(rep_dir)
    return [Path(f).stem for f in split_files(files, splits, seed)["test"]]


def score(preds: dict, ids: list[str]) -> dict:
    """Accuracy over `ids`. A missing or null prediction counts as wrong."""
    step = agent = agent_sub = missing = null_step = 0
    for i in ids:
        row = preds.get(str(i))
        if row is None:
            missing += 1
            continue
        if row.get("predicted_step") is None:
            null_step += 1
        step += step_hit(row.get("predicted_step"), row.get("gold_step"))
        agent += agent_hit(row.get("predicted_agent"), row.get("gold_agent"))
        agent_sub += agent_hit_substring(row.get("predicted_agent"), row.get("gold_agent"))
    n = len(ids)
    return {"n_test": n, "n_missing": missing, "n_null_step": null_step,
            "step_acc": step / n if n else 0.0,
            "agent_acc": agent / n if n else 0.0,
            "agent_acc_substring": agent_sub / n if n else 0.0}


# ── driver ──────────────────────────────────────────────────────────────────
def evaluate(datasets: list[str], judges: list[str]) -> pd.DataFrame:
    rows = []
    for root, with_gt, tree, suffix in SETTINGS:
        for ds in datasets:
            cfg = load_config(REPO / "configs-main" / f"{ds}{suffix}.yaml")
            for subset in cfg["subsets"]:
                seeds = seeds_for(cfg, subset)
                ids_by_seed = {s: test_ids(ds, subset, tree, cfg["splits"], s) for s in seeds}
                for judge in judges:
                    for method in METHODS:
                        preds = read_cell(root, ds, subset, judge, method)
                        if not preds:
                            print(f"  [miss] {root}/{ds}/{subset}/{judge}/{method}")
                            continue
                        for seed in seeds:
                            rows.append({
                                "with_gt": with_gt, "judge": judge, "method": method,
                                "dataset": ds, "subset": subset, "seed": seed,
                                "seeds": ",".join(map(str, seeds)),
                                "n_preds": len(preds),
                                **score(preds, ids_by_seed[seed])})
    return pd.DataFrame(rows)


def by_cell(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["with_gt", "judge", "method", "dataset", "subset"]
    g = df.groupby(keys, as_index=False).agg(
        seeds=("seeds", "first"), n_preds=("n_preds", "first"),
        n_test=("n_test", "mean"), n_missing=("n_missing", "sum"),
        n_null_step=("n_null_step", "mean"),
        step_acc=("step_acc", "mean"), agent_acc=("agent_acc", "mean"),
        agent_acc_substring=("agent_acc_substring", "mean"))
    return g


def by_column(cell: pd.DataFrame) -> pd.DataFrame:
    """The five manuscript columns. CE is the macro-average over its seven subsets."""
    rows = []
    for name, ds, subset in COLUMNS:
        sel = cell[cell["dataset"] == ds]
        sel = sel if subset is None else sel[sel["subset"] == subset]
        keys = ["with_gt", "judge", "method"]
        g = sel.groupby(keys, as_index=False)[["step_acc", "agent_acc"]].mean()
        g.insert(0, "column", name)
        g["n_subsets"] = sel.groupby(keys, as_index=False).size()["size"].values
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", dest="datasets")
    p.add_argument("--judge", action="append", dest="judges",
                   help=f"restrict to one judge (repeatable); default {JUDGES}")
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"no prompting repo at {SRC}")
    datasets = args.datasets or DATASETS
    judges = args.judges or JUDGES

    df = evaluate(datasets, judges)
    if df.empty:
        raise SystemExit("no cells read")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cell = by_cell(df)
    col = by_column(cell)
    for name, frame in (("by_seed", df), ("by_cell", cell), ("by_column", col)):
        path = out / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        print(f"  {path}  {len(frame)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
