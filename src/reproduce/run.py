"""Reproduce frozen configs and write PER-STEP scores for inspection / plotting.

Point it at reduced tables; it re-runs each selected winner and writes, per
(table, model, subset, seed):

    reproductions/<tag>/<model>/<subset>/<table>_seed-<n>_<split>.steps.tsv   per-step
    reproductions/<tag>/<model>/<subset>/<table>_seed-<n>_<split>.preds.tsv   per-trajectory
    reproductions/<tag>/<model>/<subset>/<table>_seed-<n>_<split>.json        config+metrics

`.steps.tsv` is the plotting surface: one row per (trajectory, step) with the score at
every pipeline stage (base / oriented / normalized / final), its within-trajectory rank,
and the gold flag. Because every table writes the same schema plus a `table` column,
concatenating several files gives a method-vs-method comparison on identical rows.

    # from v2/
    # reproduce the CRR winners on the test split (default)
    python -m src.reproduce.run --config configs/reproduce/correct-full.yaml

    # compare base vs CRR vs backprop, all seeds, on every trajectory
    python -m src.reproduce.run --config configs/reproduce/correct-full.yaml \
        --set tables=[base_test,crr_test,backprop_test] --set split=all

    # one specific cell
    python -m src.reproduce.run --config configs/reproduce/correct-full.yaml \
        --model qwen3.5-9b --seed 1 --set tables=[crr_test]

Verification: unless `--set verify=false`, a reproduction whose split matches the
table's selection convention must reproduce that row's recorded accuracy exactly; a
mismatch raises (the pipeline is meant to be reproducible, so this is a real failure).
"""
from __future__ import annotations

import json

import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from .core import ReproContext, reproduce_row

# Which recorded accuracy a table's rows should reproduce, per split. base_* rows store
# the base score's own accuracy; crr_/backprop_ rows store the post-rescore accuracy.
_EXPECTED = {
    "base":     {"val": "step_acc_val",      "test": "step_acc_test"},
    "crr":      {"val": "disc_step_acc_val", "test": "disc_step_acc_test"},
    "backprop": {"val": "disc_step_acc_val", "test": "disc_step_acc_test"},
}


def _family(table: str) -> str:
    return table.split("_")[0]


def _select(df: pd.DataFrame, how, seeds) -> pd.DataFrame:
    """Rows to reproduce: 'all' (every seed's winner) or 'best' (single best seed)."""
    if seeds:
        df = df[df["seed"].isin(seeds)]
    if how == "best":
        metric = ("disc_step_acc_test" if "disc_step_acc_test" in df.columns
                  else "step_acc_test")
        return df.sort_values(metric, ascending=False, kind="mergesort").head(1)
    return df


def run(cfg: dict) -> None:
    tables = cfg.get("tables", ["crr_test"])
    split = cfg.get("split", "test")
    how = cfg.get("select", "all")
    verify = cfg.get("verify", True)
    ks = cfg.get("ks", [1])

    with RunTimer(cfg, "reproductions") as rec:
        rec.note(tables=tables, split=split, select=how, verify=verify)
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                # One context per (model, subset): attention is aggregated once and
                # representations/SVD fits are cached per seed across all tables below.
                ctx = ReproContext(cfg, model, subset, n_ranges=cfg.get("n_ranges", 4))
                out_dir = paths.repro_root(cfg) / model / subset
                for table in tables:
                    src = paths.reduced_root(cfg) / model / subset / f"{table}.tsv"
                    if not src.exists():
                        print(f"[skip] missing {src}")
                        continue
                    rows = _select(pd.read_csv(src, sep="\t"), how, cfg.get("seeds"))
                    for _, row in rows.iterrows():
                        r = reproduce_row(ctx, row.to_dict(), split=split, ks=ks)
                        seed = int(row["seed"])
                        stem = out_dir / f"{table}_seed-{seed}_{split}"
                        stem.parent.mkdir(parents=True, exist_ok=True)

                        steps = r.per_step.assign(table=table)
                        preds = r.predictions.assign(table=table)
                        steps.to_csv(f"{stem}.steps.tsv", sep="\t", index=False)
                        preds.to_csv(f"{stem}.preds.tsv", sep="\t", index=False)
                        payload = {"config": r.config, "metrics": r.metrics,
                                   "n_trajectories": int(len(r.keeper.traj_ranges)),
                                   "n_steps": int(len(r.per_step))}

                        # Reproducibility check against the value the sweep recorded.
                        exp_col = _EXPECTED[_family(table)].get(split)
                        if verify and exp_col and exp_col in row and pd.notna(row[exp_col]):
                            got = r.metrics[f"step@{ks[0]}_{r.config['rank_direction']}"]
                            payload["recorded_step_acc"] = float(row[exp_col])
                            payload["reproduced_step_acc"] = float(got)
                            if abs(got - float(row[exp_col])) > 1e-9:
                                raise AssertionError(
                                    f"{table} {model}/{subset} seed{seed}: reproduced "
                                    f"{got:.6f} != recorded {float(row[exp_col]):.6f}")
                        with open(f"{stem}.json", "w") as fh:
                            json.dump(payload, fh, indent=2, default=str)
                        for suf in (".steps.tsv", ".preds.tsv", ".json"):
                            rec.add_output(f"{stem}{suf}")
                        acc = payload.get("reproduced_step_acc")
                        print(f"[repro] {table} {model}/{subset} seed{seed} {split}: "
                              f"{len(steps)} steps, {len(preds)} trajs"
                              + (f", step@1={acc:.4f} (verified)" if acc is not None else ""))


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
