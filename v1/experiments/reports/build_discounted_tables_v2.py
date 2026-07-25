"""Build discounted result tables (SVD only) — one TSV per (model, subset).

Reduces the discount sweep to the best config per (pooling, seed), mirroring
build_undiscounted_tables_v2.py.

Per (model, subset), reads
    sweep_root/{model}/{subset}/svd.tsv          (src.rescore.run output)
and writes
    out_root/{model}/{subset}/svd.tsv

Best per (pooling, seed) = argmax (disc_step_acc_test, disc_agent_acc_test)
lexicographically. Rows sorted (pooling [last, mean], seed), annotated with the
winning discount hyperparameters (svd_orient, layer_range, gamma, w) plus
undisc/disc/diff accuracies.

Roots come from the dataset manifest when the config declares ``dataset:``:
    sweep_root = discounted-splits/sweep/<tag>,  out_root = discounted-splits/reduced/<tag>.
Explicit ``sweep_root`` / ``out_root`` / ``models`` / ``subsets`` still override.

Usage:
    python -m experiments.reports.build_discounted_tables_v2 \
        --config experiments/reports/configs/discounted_v2_correct-error.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments._common.config import load_yaml, resolve
from experiments._common import paths
from experiments.reports._common import best_per_group, sort_section

OUT_COLS = [
    "seed", "pooling",
    "position", "c_begin", "c_end", "centered",
    "undisc_step_acc_val", "undisc_agent_acc_val",
    "undisc_step_acc_test", "undisc_agent_acc_test",
    "svd_orient", "layer_range", "gamma", "w",
    "disc_step_acc_val", "disc_agent_acc_val",
    "disc_step_acc_test", "disc_agent_acc_test",
    "diff_step_acc_test", "diff_agent_acc_test",
]

SORT_METRICS = ["disc_step_acc_test", "disc_agent_acc_test"]
GROUP_KEYS   = ["pooling", "seed"]


def build_table(svd_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(svd_path, sep="\t")
    if "orient" in raw.columns:
        raw = raw.rename(columns={"orient": "svd_orient"})

    best = best_per_group(raw, SORT_METRICS, GROUP_KEYS)
    best["diff_step_acc_test"]  = best["disc_step_acc_test"]  - best["undisc_step_acc_test"]
    best["diff_agent_acc_test"] = best["disc_agent_acc_test"] - best["undisc_agent_acc_test"]
    best = sort_section(best)

    cols = [c for c in OUT_COLS if c in best.columns]
    return best[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = resolve(load_yaml(args.config, args.overrides))

    sweep_root = Path(cfg["sweep_root"]) if cfg.get("sweep_root") else paths.rescore_sweep_root(cfg)
    out_root   = Path(cfg["out_root"])   if cfg.get("out_root")   else paths.disc_root(cfg)
    models     = cfg["models"]
    subsets    = cfg["subsets"]

    out_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        for subset in subsets:
            svd_path = sweep_root / model / subset / "svd.tsv"
            if not svd_path.exists():
                print(f"skip (missing): {svd_path}")
                continue

            print(f"{model}/{subset}")
            table = build_table(svd_path)

            dst_dir = out_root / model / subset
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "svd.tsv"
            table.to_csv(dst, sep="\t", index=False, na_rep="")
            print(f"  wrote {dst}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
