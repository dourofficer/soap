"""Pick the best discount config per row and assemble final discounted tables.

For each (model, subset), reads the two sweep TSVs
    outputs/tables/discounted/sweep/{model}__{subset}__svd.tsv
    outputs/tables/discounted/sweep/{model}__{subset}__classifier.tsv
and selects, per (strategy, pooling, seed), the row maximizing
(disc_step_acc_test, disc_agent_acc_test).

Output mirrors the undiscounted-table row layout (18 rows: 6 svd +
6 classifier_pseudo + 6 classifier_oracle, ordered by strategy then
pooling [last, mean] then seed [1, 2, 3]) plus columns recording which
discount hyperparameters won — `svd_orient` is blank for classifier rows.

  outputs/tables/discounted/reduced/{model}__{subset}.tsv

python -m attribscope.utils.build_discounted_tables \
    --sweep-root outputs/tables/discounted/sweep \
    --out-root outputs/tables/discounted/reduced
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SWEEP_ROOT_DEFAULT = Path("outputs/tables/discounted/sweep")
OUT_ROOT_DEFAULT   = Path("outputs/tables/discounted/reduced")

MODELS  = [
    "deepseek-8b", 
    "llama-3.1-8b", 
    "qwen3-8b", 
    "qwen3-14b"
]
SUBSETS = ["algorithm-generated", "hand-crafted"]

STRATEGY_ORDER = ["svd", "classifier_pseudo", "classifier_oracle"]
POOLING_ORDER  = ["last", "mean"]
GROUP_KEYS     = ["strategy", "pooling", "seed"]

OUT_COLS = [
    "strategy", "weight", "pooling", "method", "c_begin", "c_end", "centered",
    "threshold", "seed",
    "undisc_step_acc_val", "undisc_agent_acc_val",
    "undisc_step_acc_test", "undisc_agent_acc_test",
    "svd_orient", "layer_range", "gamma", "w",
    "disc_step_acc_val", "disc_agent_acc_val",
    "disc_step_acc_test", "disc_agent_acc_test",
    "diff_step_acc_test", "diff_agent_acc_test",
]


def _load(sweep_root: Path, model: str, subset: str, suffix: str) -> pd.DataFrame | None:
    path = sweep_root / f"{model}__{subset}__{suffix}.tsv"
    if not path.exists():
        print(f"[skip] missing sweep TSV: {path}")
        return None
    return pd.read_csv(path, sep="\t")


def _best_per_group(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values(["disc_step_acc_test", "disc_agent_acc_test"],
                           ascending=False, kind="mergesort")
              .groupby(GROUP_KEYS, as_index=False, sort=False)
              .first())


def _sort_section(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strategy"] = pd.Categorical(df["strategy"],
                                    categories=STRATEGY_ORDER, ordered=True)
    df["pooling"]  = pd.Categorical(df["pooling"],
                                    categories=POOLING_ORDER, ordered=True)
    out = (df.sort_values(["strategy", "pooling", "seed"], kind="mergesort")
             .reset_index(drop=True))
    out["strategy"] = out["strategy"].astype(str)
    out["pooling"]  = out["pooling"].astype(str)
    return out


def build_one(sweep_root: Path, model: str, subset: str) -> pd.DataFrame | None:
    svd = _load(sweep_root, model, subset, "svd")
    clf = _load(sweep_root, model, subset, "classifier")

    parts = []
    if svd is not None:
        parts.append(
            _best_per_group(svd).rename(columns={"orient": "svd_orient"})
        )
    if clf is not None:
        best = _best_per_group(clf)
        best["svd_orient"] = ""
        parts.append(best)
    if not parts:
        return None

    full = pd.concat(parts, ignore_index=True)
    full["diff_step_acc_test"]  = full["disc_step_acc_test"]  - full["undisc_step_acc_test"]
    full["diff_agent_acc_test"] = full["disc_agent_acc_test"] - full["undisc_agent_acc_test"]
    full = _sort_section(full)
    return full[OUT_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT_DEFAULT)
    ap.add_argument("--out-root",   type=Path, default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for subset in SUBSETS:
            table = build_one(args.sweep_root, model, subset)
            if table is None:
                print(f"[skip] {model}/{subset}: no sweep TSVs found")
                continue
            dst = args.out_root / f"{model}__{subset}.tsv"
            table.to_csv(dst, sep="\t", index=False, na_rep="")
            print(f"wrote {dst}  ({len(table)} rows)")


if __name__ == "__main__":
    main()