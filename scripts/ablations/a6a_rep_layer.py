"""A6(a)/(a') / fig:layers — which representation layer carries the anomaly signal.

Three views over the position axis, everything else at anchor:

  anchor-base   base score per layer at the ANCHOR band — recomputed, then verified
                against the sweep grid's base rows.
  anchor-soap   the same, plus backprop rescoring at the anchor's attention band,
                gamma and w. Not in the grid (the sweep expanded the rescore grid
                only for the winning layer), so it is computed here.
  best-band     base score per layer at that layer's own BEST spectral band,
                test-selected over the triple by the standard rule. Pure grid read.

The anchor position's anchor-base / anchor-soap rows must reproduce the selection.

    python scripts/ablations/a6a_rep_layer.py [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    base_scores, cell_paths, iter_cells, load_selection,
                    seed_mean, select_config)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import ENSEMBLE_POSITION, member_positions            # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

OUT = RESULTS_DIR / "a6a_rep_layer.tsv"


def pos_key(p: str):
    """embed < act/N ascending < act/N_normed < ens-mid3."""
    if p == "embed":
        return (0, 0)
    if p == ENSEMBLE_POSITION:
        return (3, 0)
    n = p.split("/")[1]
    if n.endswith("_normed"):
        return (2, int(n[:-7]))
    return (1, int(n))


def compute_anchor_variants(cfg, model, subset, svd_row, bp_row, device) -> list[dict]:
    """Per-seed base + rescored metrics for EVERY position at the anchor band."""
    seeds = C.seeds_for(cfg, subset)
    cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
    gamma = float(bp_row["gamma"])
    rep_dir, data_dir, files = cell_paths(cfg, model, subset)

    weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                        n_ranges=cfg["n_ranges"], device=device)
    labels = [f"{lo}-{hi}" for lo, hi in bounds]
    r_idx = labels.index(str(bp_row["layer_range"]))
    weighting = weightings[r_idx]

    rows = []
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                          files=parts[sp], device=device)
                 for sp in ("train", "val", "test")}
        train = loads["train"]
        members = member_positions(train.positions())
        positions = sorted(train.positions(), key=pos_key)
        if len(members) >= 2:
            positions.append(ENSEMBLE_POSITION)

        for sp in ("val", "test"):
            split = loads[sp]
            ctx = KeeperContext(split.keeper)
            mats = {"backprop": build_W(split.keeper, weighting, bp_row["w"], device)}
            base_s, soap_s = [], []
            for p in positions:
                s = base_scores(cfg, p, cb, ce, train, split,
                                members if p == ENSEMBLE_POSITION else None)
                base_s.append(s)
                soap_s.append(apply_strategy(s, split.keeper, mats, "backprop",
                                             [gamma])[:, 0])
            mb = compute_metrics_batch(torch.stack(base_s), None, [1], ctx=ctx)
            ms = compute_metrics_batch(torch.stack(soap_s), None, [1], ctx=ctx)
            for i, p in enumerate(positions):
                for variant, m in (("anchor-base", mb), ("anchor-soap", ms)):
                    rows.append({"seed": seed, "variant": variant, "position": p,
                                 f"step_acc_{sp}@1": float(m["step@1"][i]),
                                 f"agent_acc_{sp}@1": float(m["agent@1"][i])})
        del loads
        if device == "cuda":
            torch.cuda.empty_cache()

    # Merge the val/test halves of each (seed, variant, position).
    df = pd.DataFrame(rows)
    df = df.groupby(["seed", "variant", "position"], as_index=False).first()
    return df, seeds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_rows = []
    for cfg, model, subset in iter_cells():
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        sweep = pd.read_csv(C.sweep_dir(cfg, model, subset) / "sweep.tsv", sep="\t")
        base_grid = sweep[sweep.strategy == "base"]
        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "seeds": ",".join(map(str, seeds)),
                  "layer_range": bp_row["layer_range"],
                  "gamma": float(bp_row["gamma"]), "w": bp_row["w"]}
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={svd_row['position']} "
              f"[{svd_row['c_begin']},{svd_row['c_end']}) "
              f"attn={bp_row['layer_range']} gamma={bp_row['gamma']} w={bp_row['w']}")

        # best-band: pure grid read, per position.
        for pos, g in base_grid.groupby("position"):
            best = select_config(g, ["c_begin", "c_end"], seeds,
                                 "step_acc_test@1", "agent_acc_test@1")
            if best is None:
                continue
            out_rows.append({**common, "variant": "best-band", "position": pos,
                             "c_begin": int(best["config"]["c_begin"]),
                             "c_end": int(best["config"]["c_end"]),
                             "is_anchor": pos == svd_row["position"],
                             "step_acc_test": best["step"], "agent_acc_test": best["agent"],
                             "step_acc_val": best["step_val"],
                             "agent_acc_val": best["agent_val"]})

        # anchor-base / anchor-soap: computed, then verified.
        df, seeds = compute_anchor_variants(cfg, model, subset, svd_row, bp_row,
                                            args.device)
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        grid_anchor = base_grid[(base_grid.c_begin == cb) & (base_grid.c_end == ce)]
        for variant in ("anchor-base", "anchor-soap"):
            agg = seed_mean(df[df.variant == variant], seeds, ["position"])
            for _, r in agg.iterrows():
                if variant == "anchor-base":
                    ref = grid_anchor[(grid_anchor.position == r["position"])
                                      & grid_anchor.seed.isin(seeds)]
                    assert len(ref) == len(seeds), f"grid missing {r['position']}"
                    assert_close(r["step_acc_test@1"], float(ref["step_acc_test@1"].mean()),
                                 f"{model}/{subset} {r['position']} base vs grid")
                out_rows.append({**common, "variant": variant, "position": r["position"],
                                 "c_begin": cb, "c_end": ce,
                                 "is_anchor": r["position"] == svd_row["position"],
                                 "step_acc_test": r["step_acc_test@1"],
                                 "agent_acc_test": r["agent_acc_test@1"],
                                 "step_acc_val": r["step_acc_val@1"],
                                 "agent_acc_val": r["agent_acc_val@1"]})

        # The anchor position must reproduce the selection rows exactly.
        for variant, want in (("anchor-base", svd_row), ("anchor-soap", bp_row)):
            got = [r for r in out_rows
                   if r["model"] == model and r["subset"] == subset
                   and r["dataset"] == cfg["dataset"]
                   and r["variant"] == variant and r["position"] == svd_row["position"]]
            assert_close(got[-1]["step_acc_test"], float(want["step_acc_test"]),
                         f"{model}/{subset} {variant} anchor position")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df["_k"] = df["position"].map(pos_key)
    df = df.sort_values(["dataset", "model", "subset", "variant", "_k"]).drop(columns="_k")
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    df["cell"] = df["dataset"] + "/" + df["subset"]
    for (model, cell), g in df.groupby(["model", "cell"]):
        pivot = g.pivot_table(index="position", columns="variant",
                              values="step_acc_test") * 100
        pivot = pivot.reindex(sorted(pivot.index, key=pos_key))
        print(f"\n=== {model} {cell} (step acc %, test) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
