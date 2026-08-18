"""A5 / fig:gamma — sensitivity to the propagation strength gamma.

gamma in {0, 0.1, ..., 1.0} at anchor (position, band, attention band, w,
strategy=backprop); gamma=0 is the base scorer and the reference level. The sweep
grid holds seven of the eleven values; all eleven are recomputed here in one batched
pass, and the overlapping seven must match the grid per seed — a parity check that
the recomputation scores exactly as the sweep did.

Output is PER SEED, so the figure can shade +-1 std over the triple.

    python scripts/ablations/a5_gamma.py [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, RESULTS_DIR, anchor_filter, anchor_rows,  # noqa: E402
                    assert_close, base_scores, cell_paths, iter_cells,
                    load_selection, position_load_names)
from main import config as C                                           # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch          # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W       # noqa: E402
from main.stores import load_representations, split_files              # noqa: E402

GAMMAS = [round(0.1 * i, 1) for i in range(11)]
OUT = RESULTS_DIR / "a5_gamma.tsv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device

    rows = []
    for cfg, model, subset in iter_cells():
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        position = svd_row["position"]
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        members, names = position_load_names(rep_dir, files, position)
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={position} [{cb},{ce}) "
              f"attn={bp_row['layer_range']} w={bp_row['w']}")

        weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                            n_ranges=cfg["n_ranges"], device=device)
        labels = [f"{lo}-{hi}" for lo, hi in bounds]
        weighting = weightings[labels.index(str(bp_row["layer_range"]))]

        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "position": position, "c_begin": cb, "c_end": ce,
                  "layer_range": bp_row["layer_range"], "w": bp_row["w"]}
        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=names, files=parts[sp],
                                              device=device)
                     for sp in ("train", "val", "test")}
            per_split = {}
            for sp in ("val", "test"):
                split = loads[sp]
                s = base_scores(cfg, position, cb, ce, loads["train"], split, members)
                mats = {"backprop": build_W(split.keeper, weighting, bp_row["w"], device)}
                St = apply_strategy(s, split.keeper, mats, "backprop",
                                    GAMMAS).T.contiguous()
                per_split[sp] = compute_metrics_batch(St, None, [1],
                                                      ctx=KeeperContext(split.keeper))
            for gi, gamma in enumerate(GAMMAS):
                rows.append({**common, "seed": seed, "gamma": gamma,
                             "step_acc_test": float(per_split["test"]["step@1"][gi]),
                             "agent_acc_test": float(per_split["test"]["agent@1"][gi]),
                             "step_acc_val": float(per_split["val"]["step@1"][gi]),
                             "agent_acc_val": float(per_split["val"]["agent@1"][gi])})
            del loads
            if device == "cuda":
                torch.cuda.empty_cache()

        # Parity: the seven grid gammas must match the sweep rows per seed.
        sweep = pd.read_csv(C.sweep_dir(cfg, model, subset) / "sweep.tsv", sep="\t")
        grid = anchor_filter(sweep[sweep.strategy == "backprop"], bp_row,
                             ["position", "c_begin", "c_end", "layer_range", "w"])
        checked = 0
        for _, g in grid[grid.seed.isin(seeds)].iterrows():
            mine = [r for r in rows
                    if r["dataset"] == cfg["dataset"] and r["model"] == model
                    and r["subset"] == subset and r["seed"] == g["seed"]
                    and abs(r["gamma"] - float(g["gamma"])) < 1e-9]
            assert len(mine) == 1
            assert_close(mine[0]["step_acc_test"], float(g["step_acc_test@1"]),
                         f"{model}/{subset} seed={g['seed']} gamma={g['gamma']}")
            checked += 1
        assert checked == len(seeds) * 7, f"{model}/{subset}: parity covered {checked} rows"
        print(f"  parity ok ({checked} grid rows)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    df["cell"] = df["dataset"] + "/" + df["subset"]
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index="gamma", columns="cell", values="step_acc_test") * 100
        print(f"\n=== {model} (mean step acc %, test) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
