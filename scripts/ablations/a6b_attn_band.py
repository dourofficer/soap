"""A6(b) / fig:layers — which attention layer band carries the dependency signal.

Vary layer_range over the four equal attention bands (Qwen3.5: 0-2, 2-4, 4-6, 6-8;
DeepSeek: 0-8, 8-16, 16-24, 24-32), everything else at anchor (strategy=backprop).
Pure filter of the existing sweep grid. The anchor's own band must reproduce the
selection row exactly.

    python scripts/ablations/a6b_attn_band.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_DIR, anchor_filter, anchor_rows, assert_close,  # noqa: E402
                    iter_cells, load_selection, seed_mean)
from main import config as C                                                # noqa: E402

OUT = RESULTS_DIR / "a6b_attn_band.tsv"


def band_key(label: str) -> int:
    return int(label.split("-")[0])


def main() -> int:
    rows = []
    for cfg, model, subset in iter_cells():
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        sweep = pd.read_csv(C.sweep_dir(cfg, model, subset) / "sweep.tsv", sep="\t")

        cell = sweep[sweep.strategy == "backprop"]
        cell = anchor_filter(cell, bp_row,
                             ["position", "c_begin", "c_end", "gamma", "w"])
        agg = seed_mean(cell, seeds, ["layer_range"])
        bands = sorted(agg["layer_range"], key=band_key)
        assert len(bands) == cfg["n_ranges"], f"{model}/{subset}: got bands {bands}"

        anchor_band = str(bp_row["layer_range"])
        got = float(agg.loc[agg.layer_range == anchor_band, "step_acc_test@1"].iloc[0])
        assert_close(got, float(bp_row["step_acc_test"]),
                     f"{cfg['dataset']}/{model}/{subset} anchor band={anchor_band}")

        for band in bands:
            r = agg[agg.layer_range == band].iloc[0]
            rows.append({"dataset": cfg["dataset"], "model": model, "subset": subset,
                         "seeds": ",".join(map(str, seeds)),
                         "position": svd_row["position"],
                         "c_begin": int(svd_row["c_begin"]), "c_end": int(svd_row["c_end"]),
                         "layer_range": band,
                         "gamma": float(bp_row["gamma"]), "w": bp_row["w"],
                         "is_anchor": band == anchor_band,
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
        pivot = g.pivot_table(index="layer_range", columns="cell", values="step_acc_test") * 100
        pivot = pivot.reindex(sorted(pivot.index, key=band_key))
        print(f"\n=== {model} (step acc %, test) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
