"""B4 — a supervised probe given the validation labels.

SOAP reads step labels on the validation split to select its configuration. This
runner hands the same labels to a supervised method: a logistic regression over step
representations, one example per step, label = "is the decisive step", trained on the
VALIDATION split only. The prediction is the within-trajectory argmax of the logit,
scored on the test split like every other row.

  probe-spectral   features = the step's 20 projections onto the reference split's
                   singular vectors (the coordinates SOAP's band is drawn from)
  probe-hidden     features = the raw mean-pooled hidden state
  probe-shuffled   probe-spectral with the decisive label moved to a random step of
                   the same trajectory — a sanity floor, not a result
  base, soap       the anchors, for reference

Layer and regularization strength C are chosen by leave-one-trajectory-out
cross-validation inside the validation split (held-out step accuracy, tiebreak mean
reciprocal rank), then the probe is refit on all of it. Nothing here reads a test
label before scoring — unlike the anchors, which stay test-selected for now.

An L2-regularized logistic regression is invariant to rotating its features, so the
4096-d hidden state is rotated onto the validation matrix's row space first (at most
n_steps dimensions). The fit is identical and much faster.

    python scripts/ablations/b4_probe.py [--device cuda]
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BACKBONES, POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    base_scores, cell_paths, iter_cells, load_selection,
                    position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.score import fit_svd                                        # noqa: E402
from main.stores import load_representations, rep_names, split_files  # noqa: E402

OUT = RESULTS_DIR / "b4_probe.tsv"
CS = [0.01, 0.1, 1.0, 10.0]
MAX_LAYERS = 12
ROWS = ["probe-spectral", "probe-hidden", "probe-shuffled"]
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def candidate_layers(rep_dir, files) -> list[str]:
    """Every stored position, thinned to at most MAX_LAYERS at an even stride."""
    names = rep_names(rep_dir / files[0])
    acts = sorted((n for n in names if n.startswith("act/") and n[4:].isdigit()),
                  key=lambda n: int(n[4:]))
    stride = math.ceil(len(acts) / MAX_LAYERS)
    return acts[::-1][::stride][::-1]          # always keeps the last layer


def labels_and_groups(keeper, shuffle_seed=None):
    """(y, group) per step; trajectories whose decisive step is unscored get group -1
    and are left out of training."""
    y = np.zeros(len(keeper.index), dtype=int)
    group = np.full(len(keeper.index), -1)
    rng = np.random.default_rng(shuffle_seed)
    for gi, (a, b) in enumerate(keeper.traj_ranges):
        pos = [i for i in range(a, b) if keeper.index[i].is_mistake]
        if not pos:
            continue
        group[a:b] = gi
        y[rng.integers(a, b) if shuffle_seed is not None else pos[0]] = 1
    return y, group


def fit(X, y, c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=200).fit(X, y)


def loto(X, y, group, c) -> tuple[float, float]:
    """Leave-one-trajectory-out (accuracy, MRR) of the within-trajectory argmax."""
    hits, rr = [], []
    for g in np.unique(group[group >= 0]):
        tr, te = (group >= 0) & (group != g), group == g
        z = fit(X[tr], y[tr], c).decision_function(X[te])
        rank = 1 + int((z > z[y[te] == 1][0]).sum())
        hits.append(rank == 1)
        rr.append(1.0 / rank)
    return float(np.mean(hits)), float(np.mean(rr))


def standardize(Xv, Xt):
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-8
    return (Xv - mu) / sd, (Xt - mu) / sd


def rotate(Xv, Xt):
    """Rotate onto the validation matrix's row space (lossless for an L2 probe)."""
    _, _, Vt = np.linalg.svd(Xv, full_matrices=False)
    return Xv @ Vt.T, Xt @ Vt.T


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--select-rule", default="test", choices=["test", "val"],
                   help="which selection tree the base/SOAP anchors are read from")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    device = args.device

    rows = []
    for cfg, model, subset in iter_cells(overrides=[f"select_rule={args.select_rule}"],
                                         models=BACKBONES):
        seeds = C.seeds_for(cfg, subset)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        layers = candidate_layers(rep_dir, files)
        position = svd_row["position"]
        members, names = position_load_names(rep_dir, files, position)
        load_names = sorted(set(layers) | set(names))
        print(f"[{cfg['dataset']}] {model}/{subset} layers={layers}")

        acc = {r: {"step_t": 0.0, "agent_t": 0.0, "step_v": 0.0} for r in ROWS + ["base"]}
        chosen = {r: [] for r in ROWS}
        n_val = []
        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=load_names, files=parts[sp],
                                              device=device)
                     for sp in ("train", "val", "test")}
            val, test = loads["val"], loads["test"]
            ctx = KeeperContext(test.keeper)
            y, group = labels_and_groups(val.keeper)
            y_sh, _ = labels_and_groups(val.keeper, shuffle_seed=seed)
            n_val.append(int((group >= 0).any() and len(np.unique(group[group >= 0]))))

            s = base_scores(cfg, position, int(svd_row["c_begin"]), int(svd_row["c_end"]),
                            loads["train"], test, members)
            mb = compute_metrics_batch(s, None, [1], ctx=ctx)
            acc["base"]["step_t"] += float(mb["step@1"][0]) / len(seeds)
            acc["base"]["agent_t"] += float(mb["agent@1"][0]) / len(seeds)

            feats = {}
            for layer in layers:
                Rv = val.stores[(POOLING, layer)].R.float()
                Rt = test.stores[(POOLING, layer)].R.float()
                V = fit_svd(loads["train"].stores[(POOLING, layer)].R, cfg["n_components"])
                feats[("probe-spectral", layer)] = standardize((Rv @ V).cpu().numpy(),
                                                               (Rt @ V).cpu().numpy())
                feats[("probe-hidden", layer)] = rotate(*standardize(Rv.cpu().numpy(),
                                                                     Rt.cpu().numpy()))
                feats[("probe-shuffled", layer)] = feats[("probe-spectral", layer)]

            for row in ROWS:
                yy = y_sh if row == "probe-shuffled" else y
                best = None
                for layer in layers:
                    Xv, _ = feats[(row, layer)]
                    for c in CS:
                        key = loto(Xv, yy, group, c)
                        if best is None or key > best[0]:
                            best = (key, layer, c)
                (cv_acc, _), layer, c = best
                Xv, Xt = feats[(row, layer)]
                keep = group >= 0
                z = fit(Xv[keep], yy[keep], c).decision_function(Xt)
                m = compute_metrics_batch(torch.as_tensor(z, device=device), None, [1], ctx=ctx)
                acc[row]["step_t"] += float(m["step@1"][0]) / len(seeds)
                acc[row]["agent_t"] += float(m["agent@1"][0]) / len(seeds)
                acc[row]["step_v"] += cv_acc / len(seeds)
                chosen[row].append(f"{layer}:C={c}")
                print(f"    seed {seed} {row:15s} {layer} C={c}  cv={cv_acc:.3f} "
                      f"test={float(m['step@1'][0]):.3f}")
            del loads, feats
            if device == "cuda":
                torch.cuda.empty_cache()

        assert_close(acc["base"]["step_t"], float(svd_row["step_acc_test"]),
                     f"{model}/{subset} base self-check")
        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "seeds": ",".join(map(str, seeds)),
                  "val_trajectories": ",".join(map(str, n_val))}
        for row in ROWS:
            d = acc[row]
            rows.append({**common, "row": row, "chosen": " | ".join(chosen[row]),
                         "step_acc_test": d["step_t"], "agent_acc_test": d["agent_t"],
                         "step_acc_val_cv": d["step_v"]})
        for row, src in (("base", svd_row), ("soap", bp_row)):
            rows.append({**common, "row": row, "chosen": "",
                         "step_acc_test": float(src["step_acc_test"]),
                         "agent_acc_test": float(src["agent_acc_test"]),
                         "step_acc_val_cv": float(src["step_acc_val"])})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")
    df["cell"] = df["dataset"] + "/" + df["subset"]
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index="row", columns="cell", values="step_acc_test") * 100
        print(f"\n=== {model} (step acc %, test) ===")
        print(pivot.reindex(ROWS + ["base", "soap"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
