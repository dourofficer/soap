"""Reproduce / apply a frozen (optimal) CRR config.

For each (model, subset) it takes the winning configs from the reduced discounted
table (``discounted-splits/reduced/<tag>/<model>/<subset>/svd.tsv`` — already the
best per (pooling, seed)) and re-runs the full SVD→orient→discount pipeline via
``src.rescore.reproduce.reproduce_crr``.

Modes (``split``):
  - test / val : recompute step@1 / agent@1 on the frozen split and CHECK they
    match the reduced table (validate).
  - all        : score every trajectory in the subset and emit per-trajectory
    predicted decisive step + agent (apply / inference).

Config source (``select``):
  - best     : every row of the reduced table (optionally narrowed to the top-K
    seeds by mean disc_step_acc_test).
  - explicit : a single hand-specified config under the ``explicit:`` key (or via
    ``--set explicit.gamma=… explicit.layer_range=…`` overrides of a table row).

Outputs go to ``outputs-<ds>/reproductions/<tag>/<split>/<model>/<subset>/``:
    predictions.tsv, step_scores.tsv, metrics.tsv

    python -m experiments.reproduce.run \
        --config experiments/reproduce/configs/correct-error.yaml \
        --set split=test --set select=best
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments._common.config import load_stage_config
from experiments._common import paths
from experiments._common.sweep import CONSOLE
from src.rescore.reproduce import reproduce_crr

# Constant SVD columns the reduced discounted table drops (the discount sweep
# consumes weighted_false.tsv → method='proj', weighted=False). Overridable in
# the config if a different undiscounted source was used.
DEFAULT_METHOD   = "proj"
DEFAULT_WEIGHTED = False


def _select_rows(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pick which reduced-table rows to reproduce."""
    if cfg.get("select", "best") == "explicit":
        return pd.DataFrame([cfg["explicit"]])

    rows = table
    top_k = cfg.get("top_k")
    if top_k:
        seed_rank = (rows.groupby("seed")["disc_step_acc_test"].mean()
                         .sort_values(ascending=False))
        keep = list(seed_rank.head(int(top_k)).index)
        rows = rows[rows["seed"].isin(keep)]
    return rows.reset_index(drop=True)


def _inject_constants(row: dict, cfg: dict) -> dict:
    row = dict(row)
    row.setdefault("method",   cfg.get("method",   DEFAULT_METHOD))
    row.setdefault("weighted", cfg.get("weighted", DEFAULT_WEIGHTED))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_stage_config(args.config, args.overrides)

    split     = cfg.get("split", "test")
    device    = cfg.get("device", "cuda")
    n_ranges  = cfg.get("n_ranges", 4)
    splits    = cfg["splits"]
    disc_root = paths.disc_root(cfg)
    attn_root = paths.attn_root(cfg)
    reps_root = paths.reps_root(cfg)
    data_root = cfg["data_root"]
    out_base  = paths.reproductions_root(cfg, split)

    # For validate mode, which reduced-table column the reproduced step@1 must match.
    match_col = {"val": "disc_step_acc_val", "test": "disc_step_acc_test"}.get(split)

    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            table_path = disc_root / model / subset / "svd.tsv"
            if not table_path.exists():
                print(f"skip (missing): {table_path}")
                continue
            table = pd.read_csv(table_path, sep="\t")
            rows = _select_rows(table, cfg)
            if rows.empty:
                print(f"skip (no rows): {table_path}")
                continue

            CONSOLE.rule(f"[bold]{model}/{subset} — split={split} "
                         f"({len(rows)} config(s))[/]")

            pred_frames, score_frames, metric_rows = [], [], []
            for _, r in rows.iterrows():
                row = _inject_constants(r.to_dict(), cfg)
                res = reproduce_crr(
                    row, model, subset, reps_root, data_root, attn_root,
                    device=device, splits=splits, split=split, n_ranges=n_ranges,
                )
                tag = {"pooling": row.get("pooling"), "seed": row.get("seed")}

                preds = pd.DataFrame(res.predictions).assign(**tag)
                pred_frames.append(preds)

                sc = pd.DataFrame({
                    "traj_idx":  [e.traj_idx for e in res.keeper.index],
                    "step_idx":  [e.step_idx for e in res.keeper.index],
                    "role":      [e.role     for e in res.keeper.index],
                    "s_tilde":   res.s_tilde.tolist(),
                }).assign(**tag)
                score_frames.append(sc)

                m = {**tag,
                     "step@1":  res.metrics["step@1_desc"],
                     "agent@1": res.metrics["agent@1_desc"]}
                if match_col and match_col in row:
                    m["table_step@1"] = row[match_col]
                    m["match"] = abs(res.metrics["step@1_desc"] - float(row[match_col])) < 1e-6
                metric_rows.append(m)

            out_dir = out_base / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(pred_frames, ignore_index=True).to_csv(
                out_dir / "predictions.tsv", sep="\t", index=False)
            pd.concat(score_frames, ignore_index=True).to_csv(
                out_dir / "step_scores.tsv", sep="\t", index=False)
            metrics_df = pd.DataFrame(metric_rows)
            metrics_df.to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

            if match_col and "match" in metrics_df:
                n_ok = int(metrics_df["match"].sum())
                print(f"  validate: {n_ok}/{len(metrics_df)} configs match the reduced table")
            print(f"  wrote {out_dir}/ (predictions, step_scores, metrics)")


if __name__ == "__main__":
    main()
