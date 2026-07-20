"""Final results table — SVD + CRR (+ backprop ablation), SVD/CRR only (no baselines).

Reads the reduced per-seed tables and aggregates over seeds. Default: mean +/- std
over ALL seeds (honest); ``select_seeds: K`` selects the top-K seeds
(ranked by the chosen convention's CRR step-acc). ``convention`` picks which split's
selection feeds the table (test = headline, val = leak-free).

Writes under tables/<tag>/:
    results.tsv       Backbone x Method grid (step / agent, mean+/-std per subset)
    per_seed.tsv      the per-seed values behind every cell
    seed_ranking.tsv  seed ranking + which were selected

    # from v2/
    python -m src.reports.tables --config configs/tables/correct-full.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer

# (row label, reduced-file prefix, step col, agent col)
ROWS = [
    ("SVD",      "base", "step_acc_{c}",       "agent_acc_{c}"),
    ("CRR",      "crr",  "disc_step_acc_{c}",  "disc_agent_acc_{c}"),
    ("Backprop", "backprop", "disc_step_acc_{c}", "disc_agent_acc_{c}"),
]


def _read(cfg, model, subset, prefix, conv):
    p = paths.reduced_root(cfg) / model / subset / f"{prefix}_{conv}.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


def _fmt(vals: np.ndarray) -> str:
    if len(vals) == 0:
        return ""
    return f"{vals.mean():.4f}±{vals.std():.4f}" if len(vals) > 1 else f"{vals.mean():.4f}"


def run(cfg) -> None:
    conv = cfg.get("convention", "test")
    select_k = cfg.get("select_seeds", None)
    subsets = cfg["subsets"]

    with RunTimer(cfg, "tables") as rec:
        rec.note(convention=conv, select_seeds=select_k)
        grid_rows, per_seed, ranking_rows = [], [], []

        for model in cfg["models"]:
            # choose seeds from CRR ranking on this convention (per model), if requested
            crr = _read(cfg, model, subsets[0], "crr", conv)
            chosen = None
            if select_k and crr is not None:
                order = crr.groupby("seed")[f"disc_step_acc_{conv}"].mean().sort_values(ascending=False)
                chosen = set(order.index[:select_k])
                for rank, (seed, val) in enumerate(order.items(), 1):
                    ranking_rows.append({"model": model, "seed": seed, "mean_crr_step": val,
                                         "rank": rank, "selected": int(seed in chosen)})

            for label, prefix, stepc, agentc in ROWS:
                row = {"Backbone": model, "Method": label}
                for sk in subsets:
                    df = _read(cfg, model, sk, prefix, conv)
                    if df is None:
                        row[f"{sk} step"], row[f"{sk} agent"] = "", ""
                        continue
                    if chosen is not None:
                        df = df[df["seed"].isin(chosen)]
                    step = df[stepc.format(c=conv)].to_numpy()
                    agent = df[agentc.format(c=conv)].to_numpy()
                    row[f"{sk} step"], row[f"{sk} agent"] = _fmt(step), _fmt(agent)
                    for _, r in df.iterrows():
                        per_seed.append({"model": model, "method": label, "subset": sk,
                                         "seed": int(r["seed"]),
                                         "step": r[stepc.format(c=conv)], "agent": r[agentc.format(c=conv)]})
                grid_rows.append(row)

        out = paths.tables_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(grid_rows).to_csv(out / "results.tsv", sep="\t", index=False)
        pd.DataFrame(per_seed).to_csv(out / "per_seed.tsv", sep="\t", index=False)
        if ranking_rows:
            pd.DataFrame(ranking_rows).to_csv(out / "seed_ranking.tsv", sep="\t", index=False)
        for p in ("results.tsv", "per_seed.tsv"):
            rec.add_output(out / p)
        print(f"[tables] wrote {out}/results.tsv  (convention={conv}, "
              f"seeds={'all' if not select_k else f'top-{select_k}'})")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
