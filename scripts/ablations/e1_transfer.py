"""E1 / fig:transfer — is a fitted SOAP specific to the distribution it was tuned on?

A 4x4 source->target grid over {WW-AG, WW-HC, TE-Cap, TE-Mag}, one grid per backbone.
For each pair: fit R on the SOURCE's train split, freeze the source's full anchor
config (position, band, attention band, gamma, w), and evaluate on the TARGET's test
split. Dependency weights always come from the target trajectories' own attention.
Seeds pair positionally: source seed i's train split with target seed i's test split;
the report is the 3-seed mean. Every diagonal cell must reproduce Table 1's SOAP row.

Both a base row (no rescoring) and a soap row are recorded per pair.

    python scripts/ablations/e1_transfer.py [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    cell_paths, iter_cells, load_selection)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import fit_svd, score_steps                           # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

OUT = RESULTS_DIR / "e1_transfer.tsv"
SHORT = {("ww", "algorithm-generated"): "WW-AG", ("ww", "hand-crafted"): "WW-HC",
         ("traceelephant", "captain"): "TE-Cap", ("traceelephant", "magentic"): "TE-Mag"}


def collect_cells(model: str) -> list[dict]:
    """The four cells of one backbone, with anchors and paths resolved."""
    cells = []
    for cfg, m, subset in iter_cells():
        if m != model:
            continue
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        assert not str(svd_row["position"]).startswith("ens"), \
            "ensemble anchors are not supported by the transfer runner"
        cells.append({"cfg": cfg, "subset": subset, "name": SHORT[(cfg["dataset"], subset)],
                      "svd": svd_row, "bp": bp_row, "rep_dir": rep_dir,
                      "data_dir": data_dir, "files": files,
                      "seeds": C.seeds_for(cfg, subset)})
    return cells


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device

    models = sorted({m for _, m, _ in iter_cells()})
    rows = []
    for model in models:
        cells = collect_cells(model)
        positions = sorted({c["svd"]["position"] for c in cells})

        # Attention aggregation is seed-independent: once per target cell.
        attn = {}
        for c in cells:
            weightings, bounds = aggregate_attn(C.attn_root(c["cfg"]), model, c["subset"],
                                                n_ranges=c["cfg"]["n_ranges"], device=device)
            labels = [f"{lo}-{hi}" for lo, hi in bounds]
            attn[c["name"]] = (weightings, labels)

        acc: dict[tuple, dict] = {}
        for i in range(len(cells[0]["seeds"])):
            # Source fits: SVD at the source's anchor position on its train split.
            fits = {}
            for src in cells:
                parts = split_files(src["files"], src["cfg"]["splits"], src["seeds"][i])
                train = load_representations(src["rep_dir"], src["data_dir"],
                                             poolings=[POOLING],
                                             weight_names=[src["svd"]["position"]],
                                             files=parts["train"], device=device)
                fits[src["name"]] = fit_svd(
                    train.stores[(POOLING, src["svd"]["position"])].R,
                    src["cfg"]["n_components"])
                del train

            for tgt in cells:
                parts = split_files(tgt["files"], tgt["cfg"]["splits"], tgt["seeds"][i])
                test = load_representations(tgt["rep_dir"], tgt["data_dir"],
                                            poolings=[POOLING], weight_names=positions,
                                            files=parts["test"], device=device)
                ctx = KeeperContext(test.keeper)
                weightings, labels = attn[tgt["name"]]
                for src in cells:
                    bp = src["bp"]
                    s = score_steps(test.stores[(POOLING, src["svd"]["position"])].R,
                                    fits[src["name"]],
                                    int(src["svd"]["c_begin"]), int(src["svd"]["c_end"]))
                    weighting = weightings[labels.index(str(bp["layer_range"]))]
                    mats = {"backprop": build_W(test.keeper, weighting, bp["w"], device)}
                    St = apply_strategy(s, test.keeper, mats, "backprop",
                                        [0.0, float(bp["gamma"])]).T.contiguous()
                    m = compute_metrics_batch(St, None, [1], ctx=ctx)
                    for gi, row in ((0, "base"), (1, "soap")):
                        d = acc.setdefault((src["name"], tgt["name"], row),
                                           {"step": 0.0, "agent": 0.0})
                        d["step"] += float(m["step@1"][gi]) / len(src["seeds"])
                        d["agent"] += float(m["agent@1"][gi]) / len(src["seeds"])
                del test
                if device == "cuda":
                    torch.cuda.empty_cache()

        for src in cells:
            for tgt in cells:
                for row in ("base", "soap"):
                    d = acc[(src["name"], tgt["name"], row)]
                    rows.append({"model": model, "source": src["name"],
                                 "target": tgt["name"], "row": row,
                                 "position": src["svd"]["position"],
                                 "c_begin": int(src["svd"]["c_begin"]),
                                 "c_end": int(src["svd"]["c_end"]),
                                 "layer_range": src["bp"]["layer_range"],
                                 "gamma": float(src["bp"]["gamma"]), "w": src["bp"]["w"],
                                 "src_seeds": ",".join(map(str, src["seeds"])),
                                 "tgt_seeds": ",".join(map(str, tgt["seeds"])),
                                 "step_acc_test": d["step"], "agent_acc_test": d["agent"]})
            # Diagonal sanity: must reproduce Table 1's rows.
            assert_close(acc[(src["name"], src["name"], "soap")]["step"],
                         float(src["bp"]["step_acc_test"]),
                         f"{model} {src['name']} diagonal soap")
            assert_close(acc[(src["name"], src["name"], "base")]["step"],
                         float(src["svd"]["step_acc_test"]),
                         f"{model} {src['name']} diagonal base")
        print(f"[{model}] diagonals verified")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    order = ["WW-AG", "WW-HC", "TE-Cap", "TE-Mag"]
    for (model, row), g in df.groupby(["model", "row"]):
        pivot = g.pivot_table(index="source", columns="target",
                              values="step_acc_test") * 100
        print(f"\n=== {model} {row} (step acc %, test; rows=source) ===")
        print(pivot.reindex(order)[order].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
