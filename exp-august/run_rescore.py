"""SOAP (full) rescoring sweeps under the shared-config protocol (exp-august).

For every (model, subset): take the SVD (proj) winning config and its three shared
seeds from ``focused_selection.tsv`` (written by focused_table.py), build a 3-row
base table containing exactly that config (one row per shared seed, base metrics
copied from the recorded score files), and run the existing rescore sweep
(``src.rescore.run.run_pair``) on it.

The sweep covers the FULL grid declared in the exp-august config (orients x
score_norms x layer ranges x gammas x ws x strategies) for the fixed base config in
all three seeds — SOAP (full)'s fixed/swept split is applied later at selection
time by focused_table.py, so moving an axis between fixed and swept needs no re-run.

Sweeps land in ``exp-august/outputs/<ds>/rescore/<tag>/<model>/<subset>/
sweep_focused.tsv`` and are skipped if already present (``--force`` re-runs).

Patch (no src/ edits): ``reps_root``/``attn_root`` are pointed at the main tree's
extracted artifacts, while ``outputs_base`` points inside exp-august so everything
the run writes stays here.

    # from repo root, after focused_table.py has produced focused_selection.tsv
    python exp-august/run_rescore.py --config exp-august/configs/ww.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
sys.path.insert(0, str(REPO))                  # repo root, so `import src.*` works

import pandas as pd

from focused_table import exp_out, arts_cfg    # gt-aware roots shared by both protocols

from src.common import paths
from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.rescore import run as rescore_run


def _art_base(cfg):
    """Extracted-artifact tree (forward-pass outputs): the main outputs/ tree, or its
    outputs-gt/ mirror when the config carries ``gt: true``."""
    sub = "outputs-gt" if cfg.get("gt") else "outputs"
    return REPO / sub / (cfg.get("dataset") or cfg["name"])


# Extracted artifacts always come from the main tree (or its GT mirror), even though
# cfg2 below points outputs_base into exp-august for everything written. The lambdas
# read the cfg at call time, so the gt flag decides per invocation.
paths.reps_root = lambda cfg: _art_base(cfg) / "activations"
paths.attn_root = lambda cfg: _art_base(cfg) / "attention"

BASE_TABLE = "base_focused_test.tsv"


def base_rows(cfg, model, subset, sel_row, seeds) -> pd.DataFrame:
    """One score row per shared seed for the cell's config (undisc reference included)."""
    d = paths.scores_root(arts_cfg(cfg)) / model / subset   # main tree, or outputs-gt under gt: true
    df = pd.concat([pd.read_csv(d / f"seed-{s}.tsv", sep="\t") for s in seeds])
    df = df[df["k"] == 1]
    for col, val in cfg["base"]["fixed"].items():
        df = df[df[col] == val]
    for col in cfg["base"]["swept"]:                     # the cell's selected values
        val = sel_row[col]
        df = df[df[col] == (int(val) if str(val).lstrip("-").isdigit() else val)]
    assert len(df) == len(seeds), \
        f"{model}/{subset}: selected config found in {len(df)}/{len(seeds)} seed files"
    return df.reset_index(drop=True)


def main() -> None:
    cfg = load_and_narrow(base_parser(__doc__).parse_args())
    ds = cfg.get("dataset") or cfg["name"]
    out_base = exp_out(cfg)

    sel_file = out_base / "tables" / paths.tag(cfg) / "focused_selection.tsv"
    sel = pd.read_csv(sel_file, sep="\t")
    sel = sel[sel["row"] == "SVD (proj)"].set_index(["model", "subset"])

    cfg2 = {**cfg,
            "outputs_base": str(out_base),     # reduced/rescore/runs land in exp-august
            "data_base": str(REPO / "data"),
            "variant": "focused",              # -> sweep_focused.tsv
            "base_table": BASE_TABLE}

    with RunTimer(cfg2, "rescore") as rec:
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                if (model, subset) not in sel.index:
                    print(f"[skip] no SVD (proj) selection for {model}/{subset}")
                    continue
                r = sel.loc[(model, subset)]
                seeds = [int(s) for s in str(r["seeds"]).split(",")]
                base = base_rows(cfg, model, subset, r, seeds)
                bt = paths.reduced_root(cfg2) / model / subset / BASE_TABLE
                bt.parent.mkdir(parents=True, exist_ok=True)
                base.to_csv(bt, sep="\t", index=False)
                rescore_run.run_pair(cfg2, model, subset, rec)


if __name__ == "__main__":
    main()
