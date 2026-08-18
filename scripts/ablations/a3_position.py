"""A3 / tab:position — is SOAP's gain just a preference for early steps?

Three position-based baselines on top of the FIXED spectral base score (anchor
position and band), bracketed by the base and SOAP rows:

  temporal-z     z-score the base score within each trajectory, add -lambda * t/T
  temporal-raw   add -lambda * t/T to the raw base score (no z-scoring)
  earliest-top5  predict the earliest step among the 5 highest-scoring (no parameter)

t is the 1-based index of the step among its trajectory's scored steps, T the
trajectory length, so the bias spans (0, -lambda]. lambda in {0.1, ..., 1.0} for both
temporal variants, test-selected by the standard rule (mean test step accuracy over
the triple, tiebreak agent accuracy, then the larger lambda). The TSV records every
lambda; `selected` marks the winner. The base row is recomputed as a self-check
against the selection table; the SOAP row is copied from it.

    python scripts/ablations/a3_position.py [--device cuda]
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
                    position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

LAMBDAS = [round(0.1 * i, 1) for i in range(1, 11)]
Z_EPS = 1e-8
_TIE_DP = 12
OUT = RESULTS_DIR / "a3_position.tsv"


def variant_scores(s: torch.Tensor, keeper) -> dict[str, torch.Tensor]:
    """All baseline score vectors at once: {name: (C, N)} with C the lambda grid
    (C=1 for earliest-top5)."""
    s = s.double()
    lam = torch.tensor(LAMBDAS, dtype=s.dtype)
    z = torch.empty_like(s)
    frac = torch.empty_like(s)
    top5 = torch.empty_like(s)
    for start, end in keeper.traj_ranges:
        seg = s[start:end]
        T = seg.numel()
        z[start:end] = (seg - seg.mean()) / (seg.std(unbiased=False) + Z_EPS)
        frac[start:end] = torch.arange(1, T + 1, dtype=s.dtype) / T
        # earliest-of-top-5 via the metrics' own tie-stable rank identity:
        # rank(i) = 1 + #{j: s_j > s_i} + #{j < i: s_j == s_i}.
        gt = (seg[None, :] > seg[:, None]).sum(1)
        eq_earlier = ((seg[None, :] == seg[:, None]).long()
                      * torch.tril(torch.ones(T, T, dtype=torch.long), -1)).sum(1)
        rank = 1 + gt + eq_earlier
        idx = torch.arange(T, dtype=s.dtype)
        # Top-5 steps re-ranked earliest-first in (0, 1]; the rest strictly below,
        # keeping their original order.
        top5[start:end] = torch.where(rank <= 5, 1.0 - idx / (T + 1.0),
                                      -rank.to(s.dtype))
    return {"temporal-z": z[None, :] - lam[:, None] * frac[None, :],
            "temporal-raw": s[None, :] - lam[:, None] * frac[None, :],
            "earliest-top5": top5[None, :]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--configs", nargs="+", default=None,
                   help="config paths (default: the standard no-GT four-cell coverage)")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    device = args.device

    rows = []
    cells = iter_cells(args.configs) if args.configs else iter_cells()
    for cfg, model, subset in cells:
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        position = svd_row["position"]
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        members, names = position_load_names(rep_dir, files, position)
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={position} [{cb},{ce})")

        acc: dict[tuple, dict] = {}

        def bump(row, lam, key, val):
            d = acc.setdefault((row, lam), {"step_t": 0.0, "agent_t": 0.0,
                                            "step_v": 0.0, "agent_v": 0.0})
            d[key] += val / len(seeds)

        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=names, files=parts[sp],
                                              device=device)
                     for sp in ("train", "val", "test")}
            for sp, kt, ka in (("test", "step_t", "agent_t"), ("val", "step_v", "agent_v")):
                split = loads[sp]
                ctx = KeeperContext(split.keeper)
                s = base_scores(cfg, position, cb, ce, loads["train"], split, members)
                mb = compute_metrics_batch(s, None, [1], ctx=ctx)
                bump("base", 0.0, kt, float(mb["step@1"][0]))
                bump("base", 0.0, ka, float(mb["agent@1"][0]))
                for name, S in variant_scores(s.cpu(), split.keeper).items():
                    m = compute_metrics_batch(S.to(device), None, [1], ctx=ctx)
                    lams = LAMBDAS if name.startswith("temporal") else [0.0]
                    for i, lam in enumerate(lams):
                        bump(name, lam, kt, float(m["step@1"][i]))
                        bump(name, lam, ka, float(m["agent@1"][i]))
            del loads
            if device == "cuda":
                torch.cuda.empty_cache()

        assert_close(acc[("base", 0.0)]["step_t"], float(svd_row["step_acc_test"]),
                     f"{model}/{subset} base self-check")

        # Standard selection rule over lambda, larger lambda winning full ties.
        selected = {}
        for name in ("temporal-z", "temporal-raw"):
            best = None
            for lam in LAMBDAS:
                d = acc[(name, lam)]
                key = (round(d["step_t"], _TIE_DP), round(d["agent_t"], _TIE_DP))
                if best is None or key >= best[0]:
                    best = (key, lam)
            selected[name] = best[1]

        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "seeds": ",".join(map(str, seeds)), "position": position,
                  "c_begin": cb, "c_end": ce}
        soap = {"step_t": float(bp_row["step_acc_test"]), "agent_t": float(bp_row["agent_acc_test"]),
                "step_v": float(bp_row["step_acc_val"]), "agent_v": float(bp_row["agent_acc_val"])}
        order = ([("base", 0.0)] + [(n, l) for n in ("temporal-z", "temporal-raw")
                                    for l in LAMBDAS] + [("earliest-top5", 0.0)])
        for name, lam in order:
            d = acc[(name, lam)]
            rows.append({**common, "row": name, "lambda": lam,
                         "selected": selected.get(name) == lam if name in selected else True,
                         "step_acc_test": d["step_t"], "agent_acc_test": d["agent_t"],
                         "step_acc_val": d["step_v"], "agent_acc_val": d["agent_v"]})
        rows.append({**common, "row": "soap", "lambda": "", "selected": True,
                     "step_acc_test": soap["step_t"], "agent_acc_test": soap["agent_t"],
                     "step_acc_val": soap["step_v"], "agent_acc_val": soap["agent_v"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    sel = df[df.selected].copy()
    sel["cell"] = sel["dataset"] + "/" + sel["subset"]
    for model, g in sel.groupby("model"):
        pivot = g.pivot_table(index="row", columns="cell", values="step_acc_test") * 100
        pivot = pivot.reindex(["base", "temporal-z", "temporal-raw",
                               "earliest-top5", "soap"])
        print(f"\n=== {model} (selected rows, step acc %, test) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
