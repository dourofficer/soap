"""Successor-side SOAP sweeps under the seed-window ("triples") protocol (exp-soap).

Reads exp-august's ``triples_selection.tsv`` (protocol 2: per consecutive seed window,
the window's chosen SVD (proj) base config) and, per (model, subset), builds the SAME
union base table as exp-august/run_rescore_triples.py — one row per deduplicated
(window-chosen base config, seed) — then runs the UNMODIFIED sweep loop
(``src.rescore.run.run_pair``) ONCE for both successor-side variants, with two
module-attribute patches (no src/ edits, same style as exp-august's path patch):

* ``rescore_run.WCache``     -> SuccWCache: column-masked FULL dependency weights for
  both variants instead of the row-trimmed ones (see succ.py for the math)
* ``rescore_run.STRATEGIES`` -> adds ``succ-strong`` / ``succ-near``, each the
  unchanged backprop arithmetic on its own masked matrices

Both variants share one pass so the expensive per-seed representation loading and SVD
refits are paid once. The sweep covers the grid declared in the config; unlike
exp-august, ``orients``/``score_norms`` are TRIMMED to the protocol's fixed values
(orient=inverse, score_norm=none) — selection never reads the other combinations, so
sweeping them would be pure waste. The swept axes (layer ranges x gammas x ws) stay
full, and their fixed/swept split is applied later by triples_table.py. Sweeps land in
``exp-soap/outputs/<ds>/rescore/<tag>/<model>/<subset>/sweep_triples-succ.tsv`` (both
strategy names in the ``strategy`` column) and are skipped if present (``--force``
re-runs).

    # from repo root, after exp-august's triples protocol has produced its selection
    python exp-soap/run_rescore_triples.py --config exp-soap/configs/ww.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
sys.path.insert(0, str(REPO))                  # repo root, so `import src.*` works
sys.path.insert(0, str(EXP))

import pandas as pd

from succ import SuccWCache, STRATEGIES_SUCC

from src.common import paths
from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.rescore import run as rescore_run
from src.rescore.strategies import STRATEGIES as SRC_STRATEGIES

AUG = REPO / "exp-august"
BASE_TABLE = "base_triples_test.tsv"


# ── path patch (copied from exp-august/run_rescore.py): extracted artifacts always
# come from the main tree (or its GT mirror); everything written stays in exp-soap. ──
def _art_base(cfg):
    sub = "outputs-gt" if cfg.get("gt") else "outputs"
    return REPO / sub / (cfg.get("dataset") or cfg["name"])


paths.reps_root = lambda cfg: _art_base(cfg) / "activations"
paths.attn_root = lambda cfg: _art_base(cfg) / "attention"


def exp_out(cfg) -> Path:
    """Protocol write root; ``gt: true`` lands in exp-soap/outputs-gt/."""
    sub = "outputs-gt" if cfg.get("gt") else "outputs"
    return EXP / sub / (cfg.get("dataset") or cfg["name"])


def aug_out(cfg) -> Path:
    """exp-august's write root — source of the window selections (and frozen sweeps)."""
    sub = "outputs-gt" if cfg.get("gt") else "outputs"
    return AUG / sub / (cfg.get("dataset") or cfg["name"])


def arts_cfg(cfg) -> dict:
    """Cfg whose derived paths point at the scored-artifact tree (copied from
    exp-august/focused_table.py)."""
    if not cfg.get("gt"):
        return cfg
    ds = cfg.get("dataset") or cfg["name"]
    return {**cfg, "outputs_base": str(REPO / "outputs-gt" / ds)}


def base_rows(cfg, model, subset, sel_row, seeds) -> pd.DataFrame:
    """One score row per window seed for the window's config (copied from
    exp-august/run_rescore.py)."""
    d = paths.scores_root(arts_cfg(cfg)) / model / subset
    df = pd.concat([pd.read_csv(d / f"seed-{s}.tsv", sep="\t") for s in seeds])
    df = df[df["k"] == 1]
    for col, val in cfg["base"]["fixed"].items():
        df = df[df[col] == val]
    for col in cfg["base"]["swept"]:
        val = sel_row[col]
        df = df[df[col] == (int(val) if str(val).lstrip("-").isdigit() else val)]
    assert len(df) == len(seeds), \
        f"{model}/{subset}: selected config found in {len(df)}/{len(seeds)} seed files"
    return df.reset_index(drop=True)


def main() -> None:
    cfg = load_and_narrow(base_parser(__doc__).parse_args())
    out_base = exp_out(cfg)

    sel_file = aug_out(cfg) / "tables" / paths.tag(cfg) / "triples_selection.tsv"
    sel = pd.read_csv(sel_file, sep="\t")
    sel = sel[sel["row"] == "SVD (proj)"]

    assert all(s in STRATEGIES_SUCC for s in cfg["strategies"]), cfg["strategies"]
    cfg2 = {**cfg,
            "outputs_base": str(out_base),     # reduced/rescore/runs land in exp-soap
            "data_base": str(REPO / "data"),
            "variant": "triples-succ",         # -> sweep_triples-succ.tsv
            "base_table": BASE_TABLE}
    # The two patches that turn run_pair's backprop into the successor-side variants.
    rescore_run.WCache = SuccWCache
    rescore_run.STRATEGIES = {**SRC_STRATEGIES, **STRATEGIES_SUCC}

    with RunTimer(cfg2, "rescore") as rec:
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                rows = sel[(sel["model"] == model) & (sel["subset"] == subset)]
                if rows.empty:
                    print(f"[skip] no window selections for {model}/{subset}")
                    continue
                parts = [base_rows(cfg, model, subset, r,
                                   [int(s) for s in str(r["seeds"]).split(",")])
                         for r in rows.to_dict("records")]
                base = (pd.concat(parts)
                        .drop_duplicates(["seed"] + cfg["base"]["swept"])
                        .sort_values("seed").reset_index(drop=True))
                bt = paths.reduced_root(cfg2) / model / subset / BASE_TABLE
                bt.parent.mkdir(parents=True, exist_ok=True)
                base.to_csv(bt, sep="\t", index=False)
                print(f"[base] {model}/{subset}: {len(base)} (config, seed) rows "
                      f"from {len(rows)} windows")
                rescore_run.run_pair(cfg2, model, subset, rec)


if __name__ == "__main__":
    main()
