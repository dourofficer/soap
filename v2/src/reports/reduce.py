"""Reduce sweep tables to best-config-per-seed, in BOTH selection conventions.

Two stages, selected by ``stage`` (base | crr | both):

* base — reduce the score TSVs (all seeds) to the best base config per seed.
  Groups by SEED ONLY, so pooling competes as an ordinary column alongside
  position/method/band. Writes, under reduced/<tag>/<model>/<subset>/:
      base_test.tsv        best per seed by (step_acc_test,  agent_acc_test)
      base_val.tsv         best per seed by (step_acc_val,   agent_acc_val)   [leak-free]
      base_by_method_*.tsv best per (seed, method)  [norms included; diagnostics]
  ``base_top_k`` keeps the top-K rows per seed (rescore-input breadth knob).

* crr — reduce the rescore sweep (added in the rescore milestone).

    # from v2/
    python -m src.reports.reduce --config configs/reduce/correct-full.yaml --set stage=base
"""
from __future__ import annotations

import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer

DEFAULT_HEADLINE = ["proj", "resid", "angres", "maha"]


def best_per_group(df: pd.DataFrame, metrics: list[str], group_keys: list[str],
                   top_k: int = 1) -> pd.DataFrame:
    """Top-``top_k`` rows per group by ``metrics`` (lexicographic, stable). Adds ``rank``.

    ``group_keys`` defines what is held FIXED while everything else competes, and that
    choice is the whole statistical content of a reduction. Grouping by ``seed`` alone
    (what the base reduction does) means pooling, position, method, band and centering
    all compete against each other within a seed — one winner per seed, then seeds are
    averaged. Adding a key (e.g. ``method``) instead reports the best each family can do,
    which is a diagnostic, not a headline.

    ``mergesort`` is required: it is the only stable sort in pandas, so ties resolve by
    the incoming row order rather than arbitrarily, making reductions deterministic
    across runs.

    Selection metric matters as much as the grouping. Selecting on a ``*_test`` column
    means the reported number is the maximum over the sweep, which is optimistic by
    construction; selecting on ``*_val`` and reporting test is the leak-free protocol.
    Both are emitted so the gap between them is always visible.
    """
    d = df.sort_values(metrics, ascending=False, kind="mergesort").copy()
    d["rank"] = d.groupby(group_keys, sort=False).cumcount() + 1
    return d[d["rank"] <= top_k].reset_index(drop=True)


def _load_scores(cfg, model, subset) -> pd.DataFrame | None:
    d = paths.scores_root(cfg) / model / subset
    files = sorted(d.glob("seed-*.tsv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    return df[df["k"] == 1].reset_index(drop=True)


def reduce_base(cfg) -> list:
    headline = cfg.get("headline_methods", DEFAULT_HEADLINE)
    top_k = cfg.get("base_top_k", 1)
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            df = _load_scores(cfg, model, subset)
            if df is None:
                print(f"[skip] no scores for {model}/{subset}")
                continue
            head = df[df["method"].isin(headline)]
            out_dir = paths.reduced_root(cfg) / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            tables = {
                "base_test": best_per_group(head, ["step_acc_test", "agent_acc_test"], ["seed"], top_k),
                "base_val":  best_per_group(head, ["step_acc_val", "agent_acc_val"], ["seed"], top_k),
                "base_by_method_test": best_per_group(df, ["step_acc_test", "agent_acc_test"], ["seed", "method"], 1),
                "base_by_method_val":  best_per_group(df, ["step_acc_val", "agent_acc_val"], ["seed", "method"], 1),
            }
            for name, t in tables.items():
                p = out_dir / f"{name}.tsv"
                t.to_csv(p, sep="\t", index=False)
                written.append(p)
            print(f"[base] {model}/{subset}: {len(tables['base_test'])} test rows, "
                  f"{len(tables['base_val'])} val rows")
    return written


def reduce_crr(cfg) -> list:
    """Reduce the rescore sweep to CRR + backprop tables, both conventions.

    Best per SEED (pooling and all discount hyperparams compete). ``discount`` rows
    -> crr_{test,val}.tsv; ``backprop`` rows -> backprop_{test,val}.tsv (kept apart so
    the ablation never contaminates the headline CRR cell). Adds diff = disc - undisc.
    """
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            sweep = paths.rescore_root(cfg) / model / subset / "sweep.tsv"
            if not sweep.exists():
                print(f"[skip] no sweep for {model}/{subset}")
                continue
            df = pd.read_csv(sweep, sep="\t")
            df["diff_step_acc_test"] = df["disc_step_acc_test"] - df["undisc_step_acc_test"]
            df["diff_agent_acc_test"] = df["disc_agent_acc_test"] - df["undisc_agent_acc_test"]
            out_dir = paths.reduced_root(cfg) / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            for strat, prefix in (("discount", "crr"), ("backprop", "backprop")):
                sub = df[df["strategy"] == strat]
                if sub.empty:
                    continue
                for conv, metrics in (("test", ["disc_step_acc_test", "disc_agent_acc_test"]),
                                      ("val", ["disc_step_acc_val", "disc_agent_acc_val"])):
                    best = best_per_group(sub, metrics, ["seed"], 1)
                    p = out_dir / f"{prefix}_{conv}.tsv"
                    best.to_csv(p, sep="\t", index=False)
                    written.append(p)
            print(f"[crr] {model}/{subset}: reduced {len(df)} sweep rows")
    return written


def run(cfg) -> None:
    stage = cfg.get("stage", "both")
    with RunTimer(cfg, "reduced") as rec:
        rec.note(reduce_stage=stage)
        if stage in ("base", "both"):
            for p in reduce_base(cfg):
                rec.add_output(p)
        if stage in ("crr", "both"):
            for p in reduce_crr(cfg):
                rec.add_output(p)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
