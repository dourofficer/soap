"""A10 — does step length alone find the decisive step?

The base score is an unnormalized projection of a mean-pooled vector, so a reviewer
asked whether it tracks step length or vector norm. Rows, nothing selected:

  longest     predict the step with the most tokens
  shortest    predict the step with the fewest
  random      expected accuracy of a uniform guess (mean of 1/T)
  unit-norm   the anchor base score with every step vector scaled to norm 1, reference
              and evaluation alike (anchor layer and band unchanged)
  unit-norm-reselect
              the same, with the band re-selected over all 210 bands at the anchor
              layer by the standard rule — the base score chose its band on
              unnormalized vectors, so holding that band fixed understates this row
  base, soap  the anchors, for reference

Two diagnostics per cell, on the test split: the mean within-trajectory Spearman
correlation of the base score with the step's token count and with its vector norm.
Token counts are A1's (`results-ablations/a1_scorefn/nll`), so nothing here needs a
forward pass.

    python scripts/ablations/a10_trivial.py [--device cuda] [--select-rule test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BACKBONES, POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    base_scores, cell_paths, iter_cells, load_ntokens,
                    load_selection, ntoken_vector, position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.score import band_bounds, fit_svd, score_steps                          # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

OUT = RESULTS_DIR / "a10_trivial.tsv"
ROWS = ["longest", "shortest", "unit-norm", "base"]


def unit(R: torch.Tensor) -> torch.Tensor:
    R = R.float()
    return R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)


def random_expectation(ctx: KeeperContext) -> tuple[float, float]:
    """Expected step and agent accuracy of a uniform guess, on the metrics' divisor."""
    step = sum(1.0 / (t["end"] - t["start"]) for t in ctx.trajs)
    agent = sum(float(t["role_match"].sum()) / (t["end"] - t["start"]) for t in ctx.trajs)
    return step / ctx.total, agent / ctx.total


def mean_spearman(a: torch.Tensor, b: torch.Tensor, keeper) -> float:
    """Mean over trajectories (3+ steps, both non-constant) of Spearman(a, b)."""
    rhos = []
    for start, end in keeper.traj_ranges:
        x, y = a[start:end].cpu().numpy(), b[start:end].cpu().numpy()
        if end - start < 3 or x.std() == 0 or y.std() == 0:
            continue
        rhos.append(spearmanr(x, y).statistic)
    return float(sum(rhos) / len(rhos))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--select-rule", default="test", choices=["test", "val"])
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    device = args.device

    rows = []
    for cfg, model, subset in iter_cells(overrides=[f"select_rule={args.select_rule}"],
                                         models=BACKBONES):
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        position = svd_row["position"]
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        members, names = position_load_names(rep_dir, files, position)
        assert members is None, "unit-norm row does not support ensemble anchors"
        ntok = load_ntokens(cfg, model, subset, data_dir)
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={position} [{cb},{ce})")

        acc = {r: {"step_t": 0.0, "agent_t": 0.0, "step_v": 0.0, "agent_v": 0.0}
               for r in ROWS + ["random"]}
        diag = {"rho_len": 0.0, "rho_norm": 0.0}
        bands = band_bounds(cfg["n_components"])
        band_acc = {sp: {"step": torch.zeros(len(bands)), "agent": torch.zeros(len(bands))}
                    for sp in ("test", "val")}
        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=names, files=parts[sp],
                                              device=device)
                     for sp in ("train", "val", "test")}
            V_unit = fit_svd(unit(loads["train"].stores[(POOLING, position)].R),
                             cfg["n_components"])
            for sp, kt, ka in (("test", "step_t", "agent_t"), ("val", "step_v", "agent_v")):
                split = loads[sp]
                ctx = KeeperContext(split.keeper)
                R = split.stores[(POOLING, position)].R
                s = base_scores(cfg, position, cb, ce, loads["train"], split, members)
                n = ntoken_vector(ntok, split.keeper).to(device)
                scores = {"longest": n, "shortest": -n,
                          "unit-norm": score_steps(unit(R), V_unit, cb, ce),
                          "base": s}
                S = torch.stack([scores[r].double() for r in ROWS])
                m = compute_metrics_batch(S, None, [1], ctx=ctx)
                for i, r in enumerate(ROWS):
                    acc[r][kt] += float(m["step@1"][i]) / len(seeds)
                    acc[r][ka] += float(m["agent@1"][i]) / len(seeds)
                Sb = torch.stack([score_steps(unit(R), V_unit, b0, b1).double()
                                  for b0, b1 in bands])
                mb = compute_metrics_batch(Sb, None, [1], ctx=ctx)
                band_acc[sp]["step"] += torch.as_tensor(mb["step@1"]) / len(seeds)
                band_acc[sp]["agent"] += torch.as_tensor(mb["agent@1"]) / len(seeds)
                rs, ra = random_expectation(ctx)
                acc["random"][kt] += rs / len(seeds)
                acc["random"][ka] += ra / len(seeds)
                if sp == "test":
                    diag["rho_len"] += mean_spearman(s, n, split.keeper) / len(seeds)
                    diag["rho_norm"] += mean_spearman(s, R.float().norm(dim=1),
                                                      split.keeper) / len(seeds)
            del loads
            if device == "cuda":
                torch.cuda.empty_cache()

        assert_close(acc["base"]["step_t"], float(svd_row["step_acc_test"]),
                     f"{model}/{subset} base self-check")
        # Standard rule over the bands: step accuracy, then agent accuracy, then the
        # highest band key (`>=` over the sorted grid, as main.sweep.select_config).
        rule = band_acc[args.select_rule]
        best = None
        for i in sorted(range(len(bands)), key=lambda i: bands[i]):
            key = (round(float(rule["step"][i]), 12), round(float(rule["agent"][i]), 12))
            if best is None or key >= best[0]:
                best = (key, i)
        bi = best[1]
        print(f"    unit-norm re-selected band: {bands[bi]}")
        acc["unit-norm-reselect"] = {
            "step_t": float(band_acc["test"]["step"][bi]),
            "agent_t": float(band_acc["test"]["agent"][bi]),
            "step_v": float(band_acc["val"]["step"][bi]),
            "agent_v": float(band_acc["val"]["agent"][bi]), "band": bands[bi]}
        acc["soap"] = {"step_t": float(bp_row["step_acc_test"]),
                       "agent_t": float(bp_row["agent_acc_test"]),
                       "step_v": float(bp_row["step_acc_val"]),
                       "agent_v": float(bp_row["agent_acc_val"])}
        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "seeds": ",".join(map(str, seeds)), "position": position,
                  "c_begin": cb, "c_end": ce,
                  "spearman_base_ntokens": diag["rho_len"],
                  "spearman_base_norm": diag["rho_norm"]}
        for r in ["longest", "shortest", "random", "unit-norm", "unit-norm-reselect",
                  "base", "soap"]:
            d = acc[r]
            band = d.get("band", (cb, ce) if r in ("unit-norm", "base", "soap") else ("", ""))
            rows.append({**common, "row": r, "row_c_begin": band[0], "row_c_end": band[1], "step_acc_test": d["step_t"],
                         "agent_acc_test": d["agent_t"], "step_acc_val": d["step_v"],
                         "agent_acc_val": d["agent_v"]})

    out = Path(args.out)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    df["cell"] = df["dataset"] + "/" + df["subset"]
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index="row", columns="cell", values="step_acc_test") * 100
        print(f"\n=== {model} (step acc %, test) ===")
        print(pivot.reindex(["longest", "shortest", "random", "unit-norm",
                             "unit-norm-reselect", "base", "soap"])
              .round(2).to_string())
        print(g.drop_duplicates("cell").set_index("cell")
              [["spearman_base_ntokens", "spearman_base_norm"]].round(3).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
