"""Geometry probe — what do the (uncentered) singular vectors represent?

Per (model, subset, pooling, position, centered), fits the train SVD and reports:
  * cos(u_c, mean-direction) for c<5  — is u_0 ~ the mean (anisotropy / "typicalness")?
  * top-k energy fractions from the full spectrum
  * mistake vs non-mistake group means of {||v||, band energy ||Pv||^2, sin^2 = angres}
  * per-trajectory rank-AUC of each quantity for ranking the mistake step
    (P[q(mistake) > q(other step)]; 0.5 = chance, higher = more discriminative)

Answers the "similarity vs distance / norm vs alignment" question empirically, and
tells you which quantity carries the error signal (which distance form to prefer).

    # from v2/
    python -m src.analysis.geometry --config configs/datasets/correct-full.yaml \
        --seed 1 --set poolings=[mean] --set bands=[1,5,20]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..stores import load_representations, split_files, list_rep_files
from ..metrics import get_mistake_meta
from ..score.svd import fit_one, N_COMPONENTS
from ..score.scorers import proj, resid, angres


def _rank_auc(q: torch.Tensor, keeper) -> float:
    """Mean over mistake-bearing trajectories of P[q(mistake) > q(other step)]."""
    m_idx, _ = get_mistake_meta(keeper)
    aucs = []
    for (start, end), mstep in zip(keeper.traj_ranges, m_idx):
        if mstep is None:
            continue
        entries = keeper.index[start:end]
        seg = q[start:end]
        steps = [e.step_idx for e in entries]
        m = steps.index(mstep)
        others = torch.cat([seg[:m], seg[m + 1:]])
        if others.numel() == 0:
            continue
        aucs.append(((seg[m] > others).float().mean()).item())
    return float(np.mean(aucs)) if aucs else float("nan")


def _mistake_split(vals: torch.Tensor, keeper):
    m_idx, _ = get_mistake_meta(keeper)
    mis, non = [], []
    for (start, end), mstep in zip(keeper.traj_ranges, m_idx):
        entries = keeper.index[start:end]
        for i, e in enumerate(entries):
            (mis if (mstep is not None and e.step_idx == mstep) else non).append(vals[start + i].item())
    return (float(np.mean(mis)) if mis else float("nan"),
            float(np.mean(non)) if non else float("nan"))


def run(cfg) -> None:
    device = cfg.get("device", "cuda")
    seed = cfg["seeds"][0] if isinstance(cfg["seeds"], list) else cfg["seeds"]
    poolings = cfg.get("poolings", ["mean", "last"])
    bands = cfg.get("bands", [1, 5, 20])           # c_end values, c_begin=0
    positions_cfg = cfg.get("positions", "all")

    with RunTimer(cfg, "analysis") as rec:
        rec.note(seed=seed, poolings=poolings, bands=bands)
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                rep_dir = paths.reps_root(cfg) / model / subset
                data_dir = paths.data_root(cfg) / subset
                files = list_rep_files(rep_dir)
                parts = split_files(files, cfg["splits"], seed)
                train = load_representations(rep_dir, data_dir, poolings, files=parts["train"], device=device)
                ev_files = parts["val"] + parts["test"]
                ev = load_representations(rep_dir, data_dir, poolings, files=ev_files, device=device)
                positions = train.positions() if positions_cfg == "all" else positions_cfg

                rows = []
                for pooling in poolings:
                    for position in positions:
                        entry = fit_one(train.stores[(pooling, position)].R, N_COMPONENTS)
                        R = ev.stores[(pooling, position)].R.float()
                        for centered in (True, False):
                            V = entry["V_centered" if centered else "V_raw"]
                            S = entry["S_centered" if centered else "S_raw"]
                            ref = entry["ref"] if centered else None
                            # geometry: cos(u_c, mean-dir), energy fractions
                            mean_dir = entry["ref"] / (entry["ref"].norm() + 1e-12)
                            cos = [float((V[:, c] @ mean_dir).abs()) for c in range(min(5, V.shape[1]))]
                            e2 = S.float().square()
                            efrac = {k: float(e2[:k].sum() / e2.sum()) for k in (1, 2, 5, 20)}
                            norm_q = R.norm(dim=1) if ref is None else (R - ref).norm(dim=1)
                            for ce in bands:
                                be = proj(R, V, 0, ce, ref) * ce            # band energy ||Pv||^2 (sum)
                                rr = resid(R, V, 0, ce, ref)
                                sin2 = angres(R, V, 0, ce, ref)
                                row = {"model": model, "subset": subset, "pooling": pooling,
                                       "position": position, "centered": centered, "band_c_end": ce,
                                       **{f"cos_u{c}_mean": cos[c] for c in range(len(cos))},
                                       **{f"energy_top{k}_frac": efrac[k] for k in efrac}}
                                for qn, q in (("norm", norm_q), ("band_energy", be),
                                              ("resid", rr), ("sin2", sin2)):
                                    mm, nn = _mistake_split(q, ev.keeper)
                                    row[f"{qn}_mistake_mean"] = mm
                                    row[f"{qn}_nonmistake_mean"] = nn
                                    row[f"{qn}_rank_auc"] = _rank_auc(q, ev.keeper)
                                rows.append(row)

                out = paths.analysis_root(cfg) / "geometry" / model / subset / f"seed-{seed}.tsv"
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
                rec.add_output(out)
                # stdout summary: best-AUC quantity per pooling at the middle band
                df = pd.DataFrame(rows)
                for pooling in poolings:
                    sub = df[(df.pooling == pooling) & (df.band_c_end == bands[len(bands) // 2])]
                    aucs = {q: sub[f"{q}_rank_auc"].max() for q in ("norm", "band_energy", "resid", "sin2")}
                    best = max(aucs, key=aucs.get)
                    print(f"[geom] {model}/{subset} {pooling}: cos(u0,mean)~"
                          f"{sub['cos_u0_mean'].mean():.3f}  best-AUC={best} ({aucs[best]:.3f})")
                print(f"  wrote {out}")


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
