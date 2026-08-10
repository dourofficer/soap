"""Reproduce selected protocol rows and write PER-STEP scores for inspection.

Point it at the triples selection (``tables/<tag>/triples_selection.tsv``); for each
requested row label x (model, subset) x seed window it re-runs the frozen config on
every window seed and writes:

    reproductions/<tag>/<model>/<subset>/<label>_win-<s0>_seed-<n>_<split>.steps.tsv
    reproductions/<tag>/<model>/<subset>/<label>_win-<s0>_seed-<n>_<split>.preds.tsv
    reproductions/<tag>/<model>/<subset>/<label>_win-<s0>_seed-<n>_<split>.json

`.steps.tsv` is the plotting surface: one row per (trajectory, step) with the score at
every pipeline stage (base / oriented / normalized / final), its within-trajectory
rank, and the gold flag. All labels share the schema (plus a `row` column), so
concatenating files gives a method-vs-method comparison on identical rows.

Verification: unless ``verify: false``, the reproduced test step-accuracies averaged
over the window's seeds must match the selection row's recorded ``step_acc_test`` to
within its 4-decimal rounding; a mismatch raises (the pipeline is meant to be
reproducible, so this is a real failure).

    # from repo root — reproduce SVD (proj) + backprop winners of one window
    python -m src.reproduce.run --config configs/reproduce/correct-full.yaml \
        --set windows=[1] --model deepseek-8b
"""
from __future__ import annotations

import json

import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from .core import ReproContext, reproduce_row

BASE_ROW = "SVD (proj)"
# hyperparameter columns of triples_selection.tsv, with coercions back from TSV text
_COERCE = {"c_begin": int, "c_end": int,
           "centered": lambda v: str(v) == "True", "weighted": lambda v: str(v) == "True",
           "gamma": float}
_HPARAMS = ["pooling", "position", "method", "c_begin", "c_end", "centered", "weighted",
            "direction", "orient", "score_norm", "strategy", "layer_range", "gamma", "w"]


def _row_dict(sel_row: pd.Series, seed: int) -> dict:
    """One reproduce_row input: the selection row's hyperparameters + a window seed."""
    d = {"seed": seed}
    for c in _HPARAMS:
        v = sel_row.get(c)
        if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) == "":
            continue
        d[c] = _COERCE[c](v) if c in _COERCE else v
    if sel_row["row"] == BASE_ROW:
        d.pop("strategy", None)                     # base rows have no rescoring stage
    return d


def run(cfg: dict) -> None:
    labels = cfg.get("rows", [BASE_ROW, "backprop"])
    split = cfg.get("split", "test")
    verify = cfg.get("verify", True)
    ks = cfg.get("ks", [1])
    want = cfg.get("windows", "all")                # "all" | list of first seeds

    sel_file = paths.tables_root(cfg) / "triples_selection.tsv"
    sel = pd.read_csv(sel_file, sep="\t")

    with RunTimer(cfg, "reproductions") as rec:
        rec.note(rows=labels, split=split, verify=verify, windows=want)
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                cell = sel[(sel["model"] == model) & (sel["subset"] == subset)
                           & (sel["row"].isin(labels))]
                if want != "all":
                    cell = cell[cell["seeds"].str.split(",").str[0].astype(int).isin(want)]
                if cell.empty:
                    print(f"[skip] no selection rows for {model}/{subset}")
                    continue
                # One context per (model, subset): attention is aggregated once and
                # representations/SVD fits are cached per seed across all rows below.
                ctx = ReproContext(cfg, model, subset, n_ranges=cfg.get("n_ranges", 4))
                out_dir = paths.repro_root(cfg) / model / subset
                for _, srow in cell.iterrows():
                    seeds = [int(s) for s in str(srow["seeds"]).split(",")]
                    label = srow["row"].replace(" ", "").replace("(", "-").rstrip(")")
                    accs = []
                    for seed in seeds:
                        r = reproduce_row(ctx, _row_dict(srow, seed), split=split, ks=ks)
                        stem = out_dir / f"{label}_win-{seeds[0]}_seed-{seed}_{split}"
                        stem.parent.mkdir(parents=True, exist_ok=True)
                        r.per_step.assign(row=srow["row"]).to_csv(
                            f"{stem}.steps.tsv", sep="\t", index=False)
                        r.predictions.assign(row=srow["row"]).to_csv(
                            f"{stem}.preds.tsv", sep="\t", index=False)
                        acc = r.metrics[f"step@{ks[0]}_{r.config['rank_direction']}"]
                        accs.append(float(acc))
                        with open(f"{stem}.json", "w") as fh:
                            json.dump({"config": r.config, "metrics": r.metrics,
                                       "n_trajectories": int(len(r.keeper.traj_ranges)),
                                       "n_steps": int(len(r.per_step))},
                                      fh, indent=2, default=str)
                        for suf in (".steps.tsv", ".preds.tsv", ".json"):
                            rec.add_output(f"{stem}{suf}")
                    got = sum(accs) / len(accs)
                    recorded = float(srow["step_acc_test"])
                    status = ""
                    if verify and split == "test":
                        # recorded value is the window mean rounded to 4 decimals
                        if abs(got - recorded) > 5.1e-5:
                            raise AssertionError(
                                f"{srow['row']} {model}/{subset} win {srow['seeds']}: "
                                f"reproduced {got:.6f} != recorded {recorded:.4f}")
                        status = " (verified)"
                    print(f"[repro] {srow['row']} {model}/{subset} win {srow['seeds']} "
                          f"{split}: step@1={got:.4f}{status}")


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
