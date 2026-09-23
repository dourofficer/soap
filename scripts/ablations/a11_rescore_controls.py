"""A11 — is attention-guided rescoring more than a last-step penalty?

A reviewer argued that rescoring works because the final step collects nothing while
every other step gets a near-equal boost, and that at w=1 it mostly credits the next
step's strongest predecessor. Each control below keeps the anchor's base score and
replaces only the blame term B in  S~ = S + gamma * B:

  base            gamma = 0
  soap            B from the anchor's attention band and w (must reproduce Table 1)
  next-step       B_i = S_{i+1}; the final step gets 0
  const-boost     B_i = mean(S) for every non-final step, 0 for the final one —
                  a pure last-step penalty, no routing
  exclude-final   the base score with the final scored step removed from the argmax
  uniform-norm    B_i = mean of all successors' base scores (A2's row)
  soap-lennorm    soap, with the attention mass landing in each predecessor divided by
                  that predecessor's token count, then renormalized
  soap-nofirst    soap, with the trajectory's first turn (task description, attention
                  sink) removed from every step's dependency weights

Every row with a gamma co-selects it on the config's gamma grid by the standard rule,
so each control stands on its best gamma, as SOAP does. The TSV records every gamma;
`selected` marks the winner. A cell whose anchor chose gamma = 0 has no attention band
or w to inherit, so its three soap rows are omitted.

Diagnostics (test split, pooled over the triple): how often SOAP's prediction differs
from the base score's, how often the displaced prediction was the final step, how far
the prediction moved, and how many flips fixed or broke a correct answer; plus the
share of dependency weight landing on the first turn.

    python scripts/ablations/a11_rescore_controls.py [--device cuda] [--select-rule test]
    python scripts/ablations/a11_rescore_controls.py --configs configs-main/correct-error.yaml \
        --out results-ablations/a11_rescore_controls_ce.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BACKBONES, POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    base_scores, cell_paths, iter_cells, load_ntokens,
                    load_selection, pick, position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import WCache, aggregate_attn, apply_strategy       # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

OUT = RESULTS_DIR / "a11_rescore_controls.tsv"
CLOSED = ["next-step", "const-boost", "uniform-norm"]
ATTN = ["soap", "soap-lennorm", "soap-nofirst"]


def closed_form_B(s: torch.Tensor, keeper) -> dict[str, torch.Tensor]:
    """The blame term of every attention-free control, (N,) each."""
    s = s.double()
    B = {name: torch.zeros_like(s) for name in CLOSED}
    for start, end in keeper.traj_ranges:
        seg = s[start:end]
        T = seg.numel()
        if T < 2:
            continue
        B["next-step"][start:end - 1] = seg[1:]
        B["const-boost"][start:end - 1] = seg.mean()
        after = seg.flip(0).cumsum(0).flip(0) - seg            # sum over successors
        n_after = torch.arange(T - 1, -1, -1, dtype=s.dtype, device=s.device)
        B["uniform-norm"][start:end] = after / n_after.clamp_min(1)
    return B


def exclude_final(s: torch.Tensor, keeper) -> torch.Tensor:
    out = s.double().clone()
    for start, end in keeper.traj_ranges:
        if end - start > 1:
            out[end - 1] = float("-inf")
    return out


def reweight(weighting: dict, fn) -> dict:
    """A copy of one band's weighting with ``fn(traj_idx, ctx_ids, weights)`` applied
    to every step; ``fn`` returns the new (ctx_ids, weights), renormalized here."""
    out = {}
    for stem, steps in weighting.items():
        out[stem] = {}
        for step_idx, d in steps.items():
            ids, w = fn(int(stem), d["ctx_indices"], d["weights"])
            out[stem][step_idx] = {"ctx_indices": ids, "weights": w / (w.sum() + 1e-12)}
    return out


def first_turn_mass(weighting: dict) -> tuple[float, float]:
    """Mean dependency weight on the first turn, and what a uniform spread over the
    same predecessors would put there. Steps whose context lost turn 0 count as 0."""
    got, uniform = [], []
    for steps in weighting.values():
        for d in steps.values():
            ids, w = d["ctx_indices"], d["weights"]
            if ids.numel() == 0:
                continue
            first = ids == 0
            got.append(float(w[first].sum()))
            uniform.append(float(first.sum()) / ids.numel())
    return sum(got) / len(got), sum(uniform) / len(uniform)


def predictions(S: torch.Tensor, keeper) -> list[int]:
    """Local index of the top-ranked step per trajectory, earliest tie winning."""
    return [int((S[a:b] == S[a:b].max()).nonzero()[0]) for a, b in keeper.traj_ranges]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--select-rule", default="test", choices=["test", "val"])
    p.add_argument("--configs", nargs="+", default=None,
                   help="config paths (default: the standard no-GT four-cell coverage)")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    device = args.device
    diag_out = Path(args.out).with_name(Path(args.out).stem + "_diag.tsv")

    rows, diags = [], []
    kw = {"config_paths": args.configs} if args.configs else {}
    for cfg, model, subset in iter_cells(overrides=[f"select_rule={args.select_rule}"],
                                         models=BACKBONES, **kw):
        seeds = C.seeds_for(cfg, subset)
        gammas = [float(g) for g in cfg["gammas"]]
        assert gammas[0] == 0.0 and gammas == sorted(gammas)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        position = svd_row["position"]
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        anchor_gamma = float(bp_row["gamma"])
        has_attn = anchor_gamma > 0
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        members, names = position_load_names(rep_dir, files, position)
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={position} [{cb},{ce}) "
              f"L={bp_row['layer_range']} gamma={anchor_gamma} w={bp_row['w']}")

        bands = {}
        if has_attn:
            weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                                n_ranges=cfg["n_ranges"], device=device)
            labels = [f"{lo}-{hi}" for lo, hi in bounds]
            w = str(bp_row["w"]).removesuffix(".0")
            anchor_band = weightings[labels.index(str(bp_row["layer_range"]))]
            ntok = load_ntokens(cfg, model, subset, data_dir)

            def lennorm(traj, ids, wts):
                n = torch.tensor([ntok[traj].get(int(i), 1.0) for i in ids.tolist()],
                                 dtype=wts.dtype, device=wts.device)
                return ids, wts / n

            def nofirst(traj, ids, wts):
                keep = ids != 0
                return ids[keep], wts[keep]

            bands = {"soap": anchor_band,
                     "soap-lennorm": reweight(anchor_band, lennorm),
                     "soap-nofirst": reweight(anchor_band, nofirst)}
            mass, mass_uniform = first_turn_mass(anchor_band)

        acc: dict[tuple, dict] = {}

        def bump(row, gamma, key, val):
            d = acc.setdefault((row, gamma), {"step_t": 0.0, "agent_t": 0.0,
                                              "step_v": 0.0, "agent_v": 0.0})
            d[key] += val / len(seeds)

        flips = {"n_traj": 0, "n_flip": 0, "base_was_final": 0, "hops_earlier": 0.0,
                 "moved_earlier": 0, "fixed": 0, "broke": 0}
        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=names, files=parts[sp],
                                              device=device)
                     for sp in ("train", "val", "test")}
            for sp, kt, ka in (("test", "step_t", "agent_t"), ("val", "step_v", "agent_v")):
                split = loads[sp]
                keeper = split.keeper
                ctx = KeeperContext(keeper)
                s = base_scores(cfg, position, cb, ce, loads["train"], split, members)
                g = torch.tensor(gammas, dtype=torch.double, device=s.device)

                stacks = {"exclude-final": exclude_final(s, keeper)[None, :]}
                for name, B in closed_form_B(s, keeper).items():
                    stacks[name] = s.double()[None, :] + g[:, None] * B[None, :]
                for name, band in bands.items():
                    mats = WCache([band], keeper, [w], device=device).mats(0, w)
                    # (N, G) -> (G, N), exactly as the sweep's rescore pass does.
                    stacks[name] = apply_strategy(s, keeper, mats, "backprop",
                                                  gammas).T.contiguous()
                for name, S in stacks.items():
                    m = compute_metrics_batch(S, None, [1], ctx=ctx)
                    grid = [0.0] if name == "exclude-final" else gammas
                    for i, gamma in enumerate(grid):
                        bump(name, gamma, kt, float(m["step@1"][i]))
                        bump(name, gamma, ka, float(m["agent@1"][i]))
                        if gamma == 0.0 and name != "exclude-final":
                            assert float(m["step@1"][i]) == float(
                                compute_metrics_batch(s, None, [1], ctx=ctx)["step@1"][0]), \
                                f"{name}: gamma=0 is not the base score"
                mb = compute_metrics_batch(s, None, [1], ctx=ctx)
                bump("base", 0.0, kt, float(mb["step@1"][0]))
                bump("base", 0.0, ka, float(mb["agent@1"][0]))

                if sp == "test" and has_attn:
                    soap_S = stacks["soap"][gammas.index(anchor_gamma)]
                    pb, ps = predictions(s, keeper), predictions(soap_S, keeper)
                    gold = {(t["start"], t["end"]): t["m"] for t in ctx.trajs}
                    for (a, b), i_b, i_s in zip(keeper.traj_ranges, pb, ps):
                        flips["n_traj"] += 1
                        if i_b == i_s:
                            continue
                        flips["n_flip"] += 1
                        flips["base_was_final"] += int(i_b == b - a - 1)
                        flips["hops_earlier"] += i_b - i_s
                        flips["moved_earlier"] += int(i_s < i_b)
                        m_loc = gold.get((a, b))
                        flips["fixed"] += int(m_loc is not None and i_s == m_loc)
                        flips["broke"] += int(m_loc is not None and i_b == m_loc)
            del loads
            if device == "cuda":
                torch.cuda.empty_cache()

        assert_close(acc[("base", 0.0)]["step_t"], float(svd_row["step_acc_test"]),
                     f"{model}/{subset} base self-check")
        if has_attn:
            assert_close(acc[("soap", anchor_gamma)]["step_t"],
                         float(bp_row["step_acc_test"]), f"{model}/{subset} SOAP self-check")

        common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
                  "seeds": ",".join(map(str, seeds)), "position": position,
                  "c_begin": cb, "c_end": ce,
                  "layer_range": bp_row["layer_range"], "w": bp_row["w"]}
        order = ["base", "exclude-final"] + CLOSED + (ATTN if has_attn else [])
        for name in order:
            grid = [0.0] if name in ("base", "exclude-final") else gammas
            chosen = pick(acc, name, grid, split=args.select_rule)
            for gamma in grid:
                d = acc[(name, gamma)]
                rows.append({**common, "row": name, "gamma": gamma,
                             "selected": gamma == chosen,
                             "step_acc_test": d["step_t"], "agent_acc_test": d["agent_t"],
                             "step_acc_val": d["step_v"], "agent_acc_val": d["agent_v"]})
        if has_attn:
            n = max(flips["n_flip"], 1)
            diags.append({"dataset": cfg["dataset"], "model": model, "subset": subset,
                          "anchor_gamma": anchor_gamma, "w": bp_row["w"],
                          "test_trajs": flips["n_traj"], "flips": flips["n_flip"],
                          "flip_base_was_final": flips["base_was_final"] / n,
                          "flip_moved_earlier": flips["moved_earlier"] / n,
                          "flip_mean_hops_earlier": flips["hops_earlier"] / n,
                          "flips_fixed": flips["fixed"], "flips_broke": flips["broke"],
                          "first_turn_mass": mass, "first_turn_mass_uniform": mass_uniform})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)
    pd.DataFrame(diags).to_csv(diag_out, sep="\t", index=False)
    print(f"wrote {args.out}  ({len(df)} rows) and {diag_out}")

    sel = df[df.selected].copy()
    sel["cell"] = sel["dataset"] + "/" + sel["subset"]
    sel["show"] = (sel["step_acc_test"] * 100).round(2).astype(str) + " (" + sel["gamma"].astype(str) + ")"
    for model, g in sel.groupby("model"):
        pivot = g.pivot(index="row", columns="cell", values="show")
        print(f"\n=== {model} (selected gamma in brackets; step acc %, test) ===")
        print(pivot.reindex(["base", "exclude-final"] + CLOSED + ATTN).to_string())
    print("\n" + pd.DataFrame(diags).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
