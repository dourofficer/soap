"""E1 / fig:transfer — cross-distribution transfer with per-pair re-selection.

For each source--target pair over {WW-AG, WW-HC, TE-Cap, TE-Mag}: fit R on the
SOURCE's train split, then RE-SELECT the full configuration from scratch — the
dense base grid (position x band, ens-mid3 included), then the backprop rescore
grid (layer_range x gamma x w) on the winning base config — by the standard rule
(mean TEST step accuracy over the triple, tiebreak agent accuracy) on the
TARGET's test split, exactly as every other experiment selects. Each cell is
therefore "the best a source-fitted reference can do on the target", the same
optimistic protocol as the in-domain numbers it is compared against.

Dependency weights always come from the target trajectories' own attention.
Seeds pair positionally: source seed i's train split with target seed i's
val/test splits. The diagonal repeats the main sweep's protocol verbatim and
must reproduce Table 1's base AND SOAP rows exactly.

Both a base row (the selected base config, no rescoring) and a soap row are
recorded per pair. Shardable by backbone:

    python scripts/ablations/e1_transfer.py --models qwen3.5-9b \
        --out results-ablations/e1_parts/qwen.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    cell_paths, iter_cells, load_selection, select_config)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import (ENSEMBLE_POSITION, band_bounds, base_positions,  # noqa: E402
                        ens_score_steps, fit_svd, member_positions, score_steps)
from main.stores import load_representations, split_files             # noqa: E402

OUT = RESULTS_DIR / "e1_transfer.tsv"
SHORT = {("ww", "algorithm-generated"): "WW-AG", ("ww", "hand-crafted"): "WW-HC",
         ("traceelephant", "captain"): "TE-Cap", ("traceelephant", "magentic"): "TE-Mag"}
BASE_SWEPT = ["position", "c_begin", "c_end"]
RESCORE_SWEPT = ["layer_range", "gamma", "w"]


def collect_cells(model: str) -> list[dict]:
    cells = []
    for cfg, m, subset in iter_cells():
        if m != model:
            continue
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        cells.append({"cfg": cfg, "subset": subset, "name": SHORT[(cfg["dataset"], subset)],
                      "svd": svd_row, "bp": bp_row, "rep_dir": rep_dir,
                      "data_dir": data_dir, "files": files,
                      "seeds": C.seeds_for(cfg, subset)})
    return cells


def load_split(cell, split_name, seed, device, names="all"):
    parts = split_files(cell["files"], cell["cfg"]["splits"], seed)
    return load_representations(cell["rep_dir"], cell["data_dir"], poolings=[POOLING],
                                weight_names=names, files=parts[split_name],
                                device=device)


def metrics_rows(vm, tm, n) -> list[dict]:
    return [{"step_acc_test@1": float(tm["step@1"][i]),
             "agent_acc_test@1": float(tm["agent@1"][i]),
             "step_acc_val@1": float(vm["step@1"][i]),
             "agent_acc_val@1": float(vm["agent@1"][i])} for i in range(n)]


def base_grid(model, cells, device) -> dict[tuple, list[dict]]:
    """Per-seed dense (position x band) grid for every (source, target) pair."""
    n_comp = cells[0]["cfg"]["n_components"]
    bands = band_bounds(n_comp)
    rows: dict[tuple, list[dict]] = {(s["name"], t["name"]): [] for s in cells
                                     for t in cells}
    n_seeds = len(cells[0]["seeds"])
    for i in range(n_seeds):
        trains, fits, members, positions = {}, {}, {}, {}
        for src in cells:
            tr = load_split(src, "train", src["seeds"][i], device)
            trains[src["name"]] = tr
            available = tr.positions()
            positions[src["name"]] = base_positions(available, "all", True)
            members[src["name"]] = member_positions(available)
            fits[src["name"]] = {p: fit_svd(tr.stores[(POOLING, p)].R, n_comp)
                                 for p in available}
        for tgt in cells:
            seed = tgt["seeds"][i]
            val = load_split(tgt, "val", seed, device)
            test = load_split(tgt, "test", seed, device)
            val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
            for src in cells:
                sf, tr = fits[src["name"]], trains[src["name"]]
                for position in tqdm(positions[src["name"]],
                                     desc=f"{model} s{i} {src['name']}->{tgt['name']}",
                                     leave=False):
                    if position == ENSEMBLE_POSITION:
                        mem = members[src["name"]]
                        trR = {p: tr.stores[(POOLING, p)].R for p in mem}
                        vR = {p: val.stores[(POOLING, p)].R for p in mem}
                        tR = {p: test.stores[(POOLING, p)].R for p in mem}
                        vs = [ens_score_steps(cb, ce, mem, sf, trR, vR) for cb, ce in bands]
                        ts = [ens_score_steps(cb, ce, mem, sf, trR, tR) for cb, ce in bands]
                    else:
                        V = sf[position]
                        Rv = val.stores[(POOLING, position)].R
                        Rt = test.stores[(POOLING, position)].R
                        vs = [score_steps(Rv, V, cb, ce) for cb, ce in bands]
                        ts = [score_steps(Rt, V, cb, ce) for cb, ce in bands]
                    vm = compute_metrics_batch(torch.stack(vs), None, [1], ctx=val_ctx)
                    tm = compute_metrics_batch(torch.stack(ts), None, [1], ctx=test_ctx)
                    for m, (cb, ce) in zip(metrics_rows(vm, tm, len(bands)), bands):
                        rows[(src["name"], tgt["name"])].append(
                            {"seed": i, "position": position,
                             "c_begin": cb, "c_end": ce, **m})
            del val, test
            if device == "cuda":
                torch.cuda.empty_cache()
        del trains, fits
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def rescore_grid(model, cells, base_sel, device) -> dict[tuple, list[dict]]:
    """Per-seed backprop rescore grid on each pair's selected base config."""
    n_comp = cells[0]["cfg"]["n_components"]
    rows: dict[tuple, list[dict]] = {k: [] for k in base_sel}
    attn = {}
    for tgt in cells:
        weightings, bounds = aggregate_attn(C.attn_root(tgt["cfg"]), model,
                                            tgt["subset"],
                                            n_ranges=tgt["cfg"]["n_ranges"],
                                            device=device)
        attn[tgt["name"]] = (weightings, [f"{lo}-{hi}" for lo, hi in bounds])

    n_seeds = len(cells[0]["seeds"])
    for i in range(n_seeds):
        # Source train loads restricted to the positions any pair selected.
        needed: dict[str, set] = {src["name"]: set() for src in cells}
        for (s, t), sel in base_sel.items():
            pos = sel["config"]["position"]
            needed[s].update(member_positions_of(cells, s) if pos == ENSEMBLE_POSITION
                             else [pos])
        trains = {src["name"]: load_split(src, "train", src["seeds"][i], device,
                                          sorted(needed[src["name"]]))
                  for src in cells}
        for tgt in cells:
            seed = tgt["seeds"][i]
            tgt_needed = sorted({p for (s, t), sel in base_sel.items()
                                 if t == tgt["name"]
                                 for p in (member_positions_of(cells, s)
                                           if sel["config"]["position"] == ENSEMBLE_POSITION
                                           else [sel["config"]["position"]])})
            val = load_split(tgt, "val", seed, device, tgt_needed)
            test = load_split(tgt, "test", seed, device, tgt_needed)
            val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
            weightings, labels = attn[tgt["name"]]
            cfg = tgt["cfg"]
            gammas = list(cfg["gammas"])
            ws = list(cfg["ws"])
            # W matrices depend only on the target split — shared across sources.
            mats = {(r, str(w), sp): {"backprop": build_W(kp, weightings[r], w, device)}
                    for r in range(len(labels)) for w in ws
                    for sp, kp in (("val", val.keeper), ("test", test.keeper))}
            for src in cells:
                sel = base_sel[(src["name"], tgt["name"])]["config"]
                position = sel["position"]
                cb, ce = int(sel["c_begin"]), int(sel["c_end"])
                tr = trains[src["name"]]
                if position == ENSEMBLE_POSITION:
                    mem = member_positions_of(cells, src["name"])
                    sf = {p: fit_svd(tr.stores[(POOLING, p)].R, n_comp) for p in mem}
                    trR = {p: tr.stores[(POOLING, p)].R for p in mem}
                    s_val = ens_score_steps(cb, ce, mem, sf, trR,
                                            {p: val.stores[(POOLING, p)].R for p in mem})
                    s_test = ens_score_steps(cb, ce, mem, sf, trR,
                                             {p: test.stores[(POOLING, p)].R for p in mem})
                else:
                    V = fit_svd(tr.stores[(POOLING, position)].R, n_comp)
                    s_val = score_steps(val.stores[(POOLING, position)].R, V, cb, ce)
                    s_test = score_steps(test.stores[(POOLING, position)].R, V, cb, ce)
                for r_idx, label in enumerate(labels):
                    for w in ws:
                        Sv = apply_strategy(s_val, val.keeper, mats[(r_idx, str(w), "val")],
                                            "backprop", gammas).T.contiguous()
                        St = apply_strategy(s_test, test.keeper, mats[(r_idx, str(w), "test")],
                                            "backprop", gammas).T.contiguous()
                        vm = compute_metrics_batch(Sv, None, [1], ctx=val_ctx)
                        tm = compute_metrics_batch(St, None, [1], ctx=test_ctx)
                        for gi, m in enumerate(metrics_rows(vm, tm, len(gammas))):
                            rows[(src["name"], tgt["name"])].append(
                                {"seed": i, "layer_range": label,
                                 "gamma": gammas[gi], "w": str(w), **m})
            del val, test, mats
            if device == "cuda":
                torch.cuda.empty_cache()
        del trains
    return rows


def member_positions_of(cells, name: str) -> list[str]:
    from main.stores import rep_names
    cell = next(c for c in cells if c["name"] == name)
    return member_positions(rep_names(cell["rep_dir"] / cell["files"][0]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    models = sorted({m for _, m, _ in iter_cells()})
    if args.models:
        models = [m for m in models if m in args.models]

    out_rows = []
    for model in models:
        cells = collect_cells(model)
        seed_ids = list(range(len(cells[0]["seeds"])))
        grids = base_grid(model, cells, args.device)
        base_sel = {}
        for pair, rows in grids.items():
            base_sel[pair] = select_config(pd.DataFrame(rows), BASE_SWEPT, seed_ids,
                                           "step_acc_test@1", "agent_acc_test@1")
            assert base_sel[pair] is not None, f"no complete base config for {pair}"
            print(f"[{model}] {pair[0]}->{pair[1]} base={base_sel[pair]['config']} "
                  f"{base_sel[pair]['step']:.4f}")
        resc = rescore_grid(model, cells, base_sel, args.device)
        for src in cells:
            for tgt in cells:
                pair = (src["name"], tgt["name"])
                soap = select_config(pd.DataFrame(resc[pair]), RESCORE_SWEPT, seed_ids,
                                     "step_acc_test@1", "agent_acc_test@1")
                assert soap is not None
                base = base_sel[pair]
                print(f"[{model}] {pair[0]}->{pair[1]} soap={soap['config']} "
                      f"{soap['step']:.4f}")
                if src["name"] == tgt["name"]:
                    assert_close(base["step"], float(src["svd"]["step_acc_test"]),
                                 f"{model} {src['name']} diagonal base vs Table 1")
                    assert_close(soap["step"], float(src["bp"]["step_acc_test"]),
                                 f"{model} {src['name']} diagonal soap vs Table 1")
                common = {"model": model, "source": src["name"], "target": tgt["name"],
                          "src_seeds": ",".join(map(str, src["seeds"])),
                          "tgt_seeds": ",".join(map(str, tgt["seeds"])),
                          **{k: base["config"][k] for k in BASE_SWEPT}}
                out_rows.append({**common, "row": "base", "layer_range": "",
                                 "gamma": 0.0, "w": "",
                                 "step_acc_test": base["step"],
                                 "agent_acc_test": base["agent"],
                                 "step_acc_val": base["step_val"],
                                 "agent_acc_val": base["agent_val"]})
                out_rows.append({**common, "row": "soap", **soap["config"],
                                 "step_acc_test": soap["step"],
                                 "agent_acc_test": soap["agent"],
                                 "step_acc_val": soap["step_val"],
                                 "agent_acc_val": soap["agent_val"]})
        print(f"[{model}] diagonals verified")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    order = ["WW-AG", "WW-HC", "TE-Cap", "TE-Mag"]
    for (model, row), g in df.groupby(["model", "row"]):
        pivot = g.pivot_table(index="source", columns="target",
                              values="step_acc_test") * 100
        print(f"\n=== {model} {row} (step acc %, test; rows=source) ===")
        print(pivot.reindex(order)[order].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
