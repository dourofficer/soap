"""A4 / tab:attnsel — sensitivity to the context window w.

Vary w in {1, 2, 3, 4, 5, all} with everything else at anchor (position, band,
attention layers, gamma, strategy=backprop). Pure filter of the existing sweep grid —
nothing is recomputed. The anchor's own w must reproduce the selection row exactly.

    python scripts/ablations/a4_window.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_DIR, anchor_filter, anchor_rows, assert_close,  # noqa: E402
                    iter_cells, load_selection, norm_val, seed_mean)
from main import config as C                                                # noqa: E402

W_ORDER = ["1", "2", "3", "4", "5", "all"]
OUT = RESULTS_DIR / "a4_window.tsv"


def main() -> int:
    rows = []
    for cfg, model, subset in iter_cells():
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        sweep = pd.read_csv(C.sweep_dir(cfg, model, subset) / "sweep.tsv", sep="\t")

        cell = sweep[sweep.strategy == "backprop"]
        cell = anchor_filter(cell, bp_row,
                             ["position", "c_begin", "c_end", "layer_range", "gamma"])
        agg = seed_mean(cell, seeds, ["w"])
        agg["w"] = agg["w"].astype(str).map(norm_val)
        assert sorted(agg["w"]) == sorted(W_ORDER), f"{model}/{subset}: missing w values"

        anchor_w = norm_val(bp_row["w"])
        got = float(agg.loc[agg.w == anchor_w, "step_acc_test@1"].iloc[0])
        assert_close(got, float(bp_row["step_acc_test"]),
                     f"{cfg['dataset']}/{model}/{subset} anchor w={anchor_w}")

        for w in W_ORDER:
            r = agg[agg.w == w].iloc[0]
            rows.append({"dataset": cfg["dataset"], "model": model, "subset": subset,
                         "seeds": ",".join(map(str, seeds)),
                         "position": svd_row["position"],
                         "c_begin": int(svd_row["c_begin"]), "c_end": int(svd_row["c_end"]),
                         "layer_range": bp_row["layer_range"],
                         "gamma": float(bp_row["gamma"]), "w": w,
                         "is_anchor": w == anchor_w,
                         "step_acc_test": r["step_acc_test@1"],
                         "agent_acc_test": r["agent_acc_test@1"],
                         "step_acc_val": r["step_acc_val@1"],
                         "agent_acc_val": r["agent_acc_val@1"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    df["cell"] = df["dataset"] + "/" + df["subset"]
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index="w", columns="cell", values="step_acc_test") * 100
        print(f"\n=== {model} (step acc %, test) ===")
        print(pivot.reindex(W_ORDER).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
