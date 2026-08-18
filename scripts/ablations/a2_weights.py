"""tab:weights — the effect of attention-guided rescoring.

The table compares four rows per (backbone, subset): the base score alone, the two
content-agnostic uniform aggregations, and SOAP. It absorbs the main tables' "SOAP
(w/o rescoring)" row, so it runs on all five subsets, both backbones, both GT trees.

Only the uniform rows are computed here. The base and SOAP rows already exist in
`results-{nogt,gt}/<ds>/select/selection.tsv` (verified against Tables 1-2); the base
row is still recomputed per seed as a self-check that this script scores steps exactly
as the sweep did, and the run aborts on any mismatch.

A uniform row keeps the anchor's base config and gamma and replaces only the weights.
With every weight equal, the correction needs no attention and no matrices:

    uniform (normalized)    B_i = mean of the base scores of ALL successors of i
    uniform (unnormalized)  B_i = their raw SUM  (earlier steps aggregate more terms)
    S~ = S + gamma * B      (last step: no successors, B = 0, score passes through)

    python scripts/ablations/a2_weights.py                  # all six configs
    python scripts/ablations/a2_weights.py --configs configs-main/ww.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main import config as C                                        # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch       # noqa: E402
from main.score import (ENSEMBLE_POSITION, ens_score_steps, fit_svd,  # noqa: E402
                        member_positions, score_steps)
from main.stores import (list_rep_files, load_representations,      # noqa: E402
                         rep_names, split_files)

POOLING = "mean"
CONFIGS = ["configs-main/ww.yaml", "configs-main/ww-gt.yaml",
           "configs-main/correct-error.yaml", "configs-main/correct-error-gt.yaml",
           "configs-main/traceelephant.yaml", "configs-main/traceelephant-gt.yaml"]
OUT = REPO / "results-ablations" / "a2_weights.tsv"


def successor_B(s: torch.Tensor, keeper) -> tuple[torch.Tensor, torch.Tensor]:
    """Per step: (sum, mean) of the base scores of all later steps in its trajectory.

    Computed in float64: the folded-inverse scores span many orders of magnitude, and
    these are new numbers with no bit-parity constraint to preserve.
    """
    s = s.double()
    B_sum, B_mean = torch.zeros_like(s), torch.zeros_like(s)
    for start, end in keeper.traj_ranges:
        seg = s[start:end]
        T = seg.numel()
        suffix = torch.flip(torch.cumsum(torch.flip(seg, [0]), 0), [0])
        sums = torch.cat([suffix[1:], seg.new_zeros(1)])            # sum over t > i
        counts = torch.arange(T - 1, -1, -1, device=s.device, dtype=s.dtype)
        B_sum[start:end] = sums
        B_mean[start:end] = sums / counts.clamp(min=1)
    return B_sum, B_mean


def anchor_rows(sel: pd.DataFrame, model: str, subset: str) -> tuple[pd.Series, pd.Series]:
    """The svd (base) and backprop (SOAP) selection rows for one cell."""
    cell = sel[(sel["model"] == model) & (sel["subset"] == subset)]
    svd = cell[cell["row"] == "svd"]
    bp = cell[cell["row"] == "backprop"]
    assert len(svd) == 1 and len(bp) == 1, f"incomplete selection for {model}/{subset}"
    return svd.iloc[0], bp.iloc[0]


def base_scores(cfg, position, cb, ce, train, split, members=None):
    """The sweep's base score for one split, bit-identical to `_base_pass`.

    ``members`` must be the middle third of the FULL position list. It cannot be
    recomputed from ``train.positions()`` here: only the members are loaded, and
    re-applying the rule to that restricted list would take the middle third twice.
    """
    if position == ENSEMBLE_POSITION:
        fits = {p: fit_svd(train.stores[(POOLING, p)].R, cfg["n_components"])
                for p in members}
        tr = {p: train.stores[(POOLING, p)].R for p in members}
        ev = {p: split.stores[(POOLING, p)].R for p in members}
        return ens_score_steps(cb, ce, members, fits, tr, ev)
    V = fit_svd(train.stores[(POOLING, position)].R, cfg["n_components"])
    return score_steps(split.stores[(POOLING, position)].R, V, cb, ce)


def run_cell(cfg, model, subset, svd_row, bp_row, device) -> list[dict]:
    seeds = C.seeds_for(cfg, subset)
    position = svd_row["position"]
    cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
    gamma = float(bp_row["gamma"])
    rep_dir = C.reps_root(cfg) / model / subset
    data_dir = C.data_root(cfg) / subset
    files = list_rep_files(rep_dir)

    if position == ENSEMBLE_POSITION:
        members = member_positions(rep_names(rep_dir / files[0]))
        names = members
    else:
        members, names = None, [position]

    acc = {r: {"step_t": 0.0, "agent_t": 0.0, "step_v": 0.0, "agent_v": 0.0}
           for r in ("base", "uniform-unnorm", "uniform-norm")}
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                          weight_names=names, files=parts[sp],
                                          device=device)
                 for sp in ("train", "val", "test")}
        for sp, key_t, key_a in (("test", "step_t", "agent_t"), ("val", "step_v", "agent_v")):
            split = loads[sp]
            s = base_scores(cfg, position, cb, ce, loads["train"], split, members)
            B_sum, B_mean = successor_B(s, split.keeper)
            uni = torch.stack([s.double() + gamma * B_sum,
                               s.double() + gamma * B_mean])
            ctx = KeeperContext(split.keeper)
            mb = compute_metrics_batch(s, None, [1], ctx=ctx)
            mu = compute_metrics_batch(uni, None, [1], ctx=ctx)
            acc["base"][key_t] += float(mb["step@1"][0]) / len(seeds)
            acc["base"][key_a] += float(mb["agent@1"][0]) / len(seeds)
            for i, r in enumerate(("uniform-unnorm", "uniform-norm")):
                acc[r][key_t] += float(mu["step@1"][i]) / len(seeds)
                acc[r][key_a] += float(mu["agent@1"][i]) / len(seeds)
        del loads
        if device == "cuda":
            torch.cuda.empty_cache()

    drift = abs(acc["base"]["step_t"] - float(svd_row["step_acc_test"]))
    assert drift < 1e-9, (f"{model}/{subset}: recomputed base step acc "
                          f"{acc['base']['step_t']:.12f} != selection "
                          f"{float(svd_row['step_acc_test']):.12f}")

    common = {"with_gt": cfg["gt"], "dataset": cfg["dataset"], "model": model,
              "subset": subset, "seeds": ",".join(map(str, seeds)),
              "position": position, "c_begin": cb, "c_end": ce}
    rows = [{**common, "row": "base", "gamma": 0.0,
             "step_acc_test": acc["base"]["step_t"], "agent_acc_test": acc["base"]["agent_t"],
             "step_acc_val": acc["base"]["step_v"], "agent_acc_val": acc["base"]["agent_v"]}]
    for r in ("uniform-unnorm", "uniform-norm"):
        rows.append({**common, "row": r, "gamma": gamma,
                     "step_acc_test": acc[r]["step_t"], "agent_acc_test": acc[r]["agent_t"],
                     "step_acc_val": acc[r]["step_v"], "agent_acc_val": acc[r]["agent_v"]})
    rows.append({**common, "row": "soap", "gamma": gamma,
                 "step_acc_test": float(bp_row["step_acc_test"]),
                 "agent_acc_test": float(bp_row["agent_acc_test"]),
                 "step_acc_val": float(bp_row["step_acc_val"]),
                 "agent_acc_val": float(bp_row["agent_acc_val"])})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=CONFIGS)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows: list[dict] = []
    for cfg_path in args.configs:
        cfg = C.load_config(REPO / cfg_path)
        sel = pd.read_csv(C.select_dir(cfg) / "selection.tsv", sep="\t")
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                svd_row, bp_row = anchor_rows(sel, model, subset)
                print(f"[{cfg['dataset']}{'-gt' if cfg['gt'] else ''}] {model}/{subset} "
                      f"anchor={svd_row['position']} [{svd_row['c_begin']},"
                      f"{svd_row['c_end']}) gamma={bp_row['gamma']}")
                rows.extend(run_cell(cfg, model, subset, svd_row, bp_row, args.device))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    # A quick look in the manuscript's layout: rows x subsets, step accuracy (%).
    for (gt, model), g in df.groupby(["with_gt", "model"]):
        g = g.copy()
        g["cell"] = g["dataset"] + "/" + g["subset"]
        pivot = g.pivot_table(index="row", columns="cell", values="step_acc_test") * 100
        print(f"\n=== with_gt={gt} model={model} (step acc %, CE per-subset) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
