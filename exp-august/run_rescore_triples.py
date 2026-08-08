"""Rescoring sweeps for the seed-window ("triples") protocol (exp-august).

Reads ``triples_selection.tsv`` (written by triples_table.py) and, per (model,
subset), builds ONE base table holding the union over windows of (chosen base
config x the window's seeds) — deduplicated — then runs the existing rescore sweep
(``src.rescore.run.run_pair``) on it. The sweep covers the FULL grid declared in
the config, so the fixed/swept split is applied later at selection time.

Sweeps land in ``exp-august/outputs/<ds>/rescore/<tag>/<model>/<subset>/
sweep_triples.tsv`` and are skipped if already present. NOTE: the skip is per
cell — after adding windows or changing the base selection, re-run with
``--force`` so the sweep is rebuilt to cover the new (config, seed) pairs.

Reuses run_rescore.py's path patch (main-tree reps/attention, exp-august writes).

    # from repo root, after triples_table.py has produced triples_selection.tsv
    python exp-august/run_rescore_triples.py --config exp-august/configs/ww.yaml
"""
from __future__ import annotations

import pandas as pd

from run_rescore import EXP, REPO, base_rows, paths, exp_out   # importing applies the path patch

from src.common.cli import base_parser, load_and_narrow
from src.common.provenance import RunTimer
from src.rescore import run as rescore_run

BASE_TABLE = "base_triples_test.tsv"


def main() -> None:
    cfg = load_and_narrow(base_parser(__doc__).parse_args())
    ds = cfg.get("dataset") or cfg["name"]
    out_base = exp_out(cfg)

    sel_file = out_base / "tables" / paths.tag(cfg) / "triples_selection.tsv"
    sel = pd.read_csv(sel_file, sep="\t")
    sel = sel[sel["row"] == "SVD (proj)"]

    cfg2 = {**cfg,
            "outputs_base": str(out_base),     # reduced/rescore/runs land in exp-august
            "data_base": str(REPO / "data"),
            "variant": "triples",              # -> sweep_triples.tsv
            "base_table": BASE_TABLE}

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
