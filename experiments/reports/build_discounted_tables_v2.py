"""Build discounted result tables (SVD only) — one TSV per (model, subset).

Reduces the discount sweep to the best config per (pooling, seed), mirroring
the layout/argument handling of build_undiscounted_tables_v2.py.

Per (model, subset), reads
    sweep_root/{model}/{subset}/svd.tsv          (experiments.rescore.sweep output)
and writes
    out_root/{model}/{subset}/svd.tsv

Each output has 6 rows (2 poolings x 3 seeds). Best per (pooling, seed) =
argmax (disc_step_acc_test, disc_agent_acc_test) lexicographically. Rows are
sorted (pooling [last, mean], seed) and annotated with the winning discount
hyperparameters (svd_orient, layer_range, gamma, w) plus undisc/disc/diff
accuracies.

Usage:
    python -m experiments.reports.build_discounted_tables_v2
    python -m experiments.reports.build_discounted_tables_v2 \
        --config experiments/reports/configs/discounted_v2.yaml \
        --set sweep_root="outputs-1006/discounted-splits/sweep/424" \
        --set out_root="outputs-1006/discounted-splits/reduced/424"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# ── Defaults (overridable via config / CLI) ──────────────────────────────────
SWEEP_ROOT_DEFAULT = Path("outputs/tables/discounted/sweep")
OUT_ROOT_DEFAULT   = Path("outputs/tables/discounted/reduced")

MODELS_DEFAULT  = ["deepseek-8b", "llama-3.1-8b", "qwen3-8b", "qwen3-14b"]
SUBSETS_DEFAULT = ["algorithm-generated", "hand-crafted"]

POOLING_ORDER = ["last", "mean"]
GROUP_KEYS    = ["pooling", "seed"]

OUT_COLS = [
    "strategy", "position", "pooling", "method", "c_begin", "c_end",
    "centered", "weighted", "threshold", "seed",
    "undisc_step_acc_val", "undisc_agent_acc_val",
    "undisc_step_acc_test", "undisc_agent_acc_test",
    "svd_orient", "layer_range", "gamma", "w",
    "disc_step_acc_val", "disc_agent_acc_val",
    "disc_step_acc_test", "disc_agent_acc_test",
    "diff_step_acc_test", "diff_agent_acc_test",
]


# ── Core logic ───────────────────────────────────────────────────────────────
def _best_per_pooling_seed(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values(["disc_step_acc_test", "disc_agent_acc_test"],
                           ascending=False, kind="mergesort")
              .groupby(GROUP_KEYS, as_index=False, sort=False)
              .first())


def _sort_section(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pooling"] = pd.Categorical(df["pooling"],
                                   categories=POOLING_ORDER, ordered=True)
    out = (df.sort_values(["pooling", "seed"], kind="mergesort")
             .reset_index(drop=True))
    out["pooling"] = out["pooling"].astype(str)
    return out


def build_table(svd_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(svd_path, sep="\t")
    if "orient" in raw.columns:
        raw = raw.rename(columns={"orient": "svd_orient"})

    best = _best_per_pooling_seed(raw)
    best["diff_step_acc_test"]  = best["disc_step_acc_test"]  - best["undisc_step_acc_test"]
    best["diff_agent_acc_test"] = best["disc_agent_acc_test"] - best["undisc_agent_acc_test"]
    best = _sort_section(best)

    cols = [c for c in OUT_COLS if c in best.columns]
    return best[cols]


# ── Config / CLI ─────────────────────────────────────────────────────────────
def load_cfg(path: Path | None, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text()) if path else {}
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config file (all keys optional)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    sweep_root = Path(cfg.get("sweep_root", SWEEP_ROOT_DEFAULT))
    out_root   = Path(cfg.get("out_root", OUT_ROOT_DEFAULT))
    models     = cfg.get("models",  MODELS_DEFAULT)
    subsets    = cfg.get("subsets", SUBSETS_DEFAULT)

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