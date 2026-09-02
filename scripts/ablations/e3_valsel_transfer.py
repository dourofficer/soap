"""E3 Part B — cross-distribution transfer under validation selection.

E1's 4x4 source->target grid over {WW-AG, WW-HC, TE-Cap, TE-Mag}, one grid per
backbone, without-GT — with the selection rule of E3 Part A. For each pair the
reference R is fit on the SOURCE's train split; the config is RE-SELECTED per pair on
the TARGET's validation split (dense base grid, then the backprop rescore grid on the
val-winning base config) and its target-test accuracy is reported.

Two differences from `e1_transfer.py`:

  * selection is val-only (no test convention);
  * the target's val is its MAIN-EXPERIMENT val split (20%), not train+val (40%),
    so every cell — diagonal included — selects on the same data as Part A. The
    diagonal therefore repeats Part A's without-GT selection problem exactly and is
    asserted against results-nogt-valsel/<ds>/select/selection.tsv.

Dependency weights always come from the target trajectories' own attention. Seeds
pair positionally: source seed i's train split with target seed i's val/test splits.
Shardable: --models / --sources narrow the pairs; merge the partial TSVs afterwards.

    python scripts/ablations/e3_valsel_transfer.py --models qwen3.5-9b \
        --sources WW-AG WW-HC --out results-ablations/e3_parts/qwen-ww.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_transfer as E1                                          # noqa: E402
from common import RESULTS_DIR, anchor_rows, assert_close, cell_paths, iter_cells, select_config  # noqa: E402
from main import config as C                                      # noqa: E402
from main.rescore import aggregate_attn                           # noqa: E402
from main.stores import split_files                               # noqa: E402
from main.sweep import BASE_SWEPT, RESCORE_SWEPT                  # noqa: E402

OUT = RESULTS_DIR / "e3_valsel_transfer.tsv"
MODELS = ["qwen3.5-9b", "deepseek-8b"]


def pair_splits_main_val(src, tgt, i):
    """(source train, target val, target test) — the target's MAIN val split."""
    sparts = split_files(src["files"], src["cfg"]["splits"], src["seeds"][i])
    tparts = split_files(tgt["files"], tgt["cfg"]["splits"], tgt["seeds"][i])
    return sparts["train"], tparts["val"], tparts["test"]


# E1's grid builders call `pair_splits` by module-global name; rebind it here so the
# scoring code stays byte-identical and only the partition changes.
E1.pair_splits = pair_splits_main_val


def collect_cells(model: str) -> list[dict]:
    """E1's cells, with the anchors read from the val-rule tree."""
    cells = []
    for cfg, m, subset in iter_cells():
        if m != model:
            continue
        cfg_val = C.load_config(C.REPO_ROOT / f"configs-main/{cfg['dataset']}.yaml",
                                ["select_rule=val"])
        sel = pd.read_csv(C.select_dir(cfg_val) / "selection.tsv", sep="\t")
        svd_row, bp_row = anchor_rows(sel, model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        cells.append({"cfg": cfg, "subset": subset,
                      "name": E1.SHORT[(cfg["dataset"], subset)],
                      "svd": svd_row, "bp": bp_row, "rep_dir": rep_dir,
                      "data_dir": data_dir, "files": files,
                      "seeds": C.seeds_for(cfg, subset)})
    return cells


def run_pair(src, tgt, model, attn, device) -> list[dict]:
    n_seeds = len(src["seeds"])
    weightings, labels = attn[tgt["name"]]
    idx = list(range(n_seeds))
    sc, ac = "step_acc_val@1", "agent_acc_val@1"
    base_df = E1.base_grid(src, tgt, model, device)
    base_sel = select_config(base_df, BASE_SWEPT, idx, sc, ac)
    bcfg = base_sel["config"]
    resc_df = E1.rescore_grid(src, tgt, bcfg, weightings, labels, device)
    soap_sel = select_config(resc_df, RESCORE_SWEPT, idx, sc, ac)
    bm = E1.config_metrics(base_df, bcfg, n_seeds)
    sm = E1.config_metrics(resc_df, soap_sel["config"], n_seeds)
    rows = [E1.out_row(model, src, tgt, "val", "base", bcfg, None, bm),
            E1.out_row(model, src, tgt, "val", "soap", bcfg, soap_sel["config"], sm)]
    print(f"  [val] base={bcfg} test {bm['step_acc_test@1']:.4f} "
          f"soap={soap_sel['config']} test {sm['step_acc_test@1']:.4f}")
    if src["name"] == tgt["name"]:
        assert_close(bm["step_acc_test@1"], float(src["svd"]["step_acc_test"]),
                     f"{model} {src['name']} diagonal base vs Part A")
        assert_close(sm["step_acc_test@1"], float(src["bp"]["step_acc_test"]),
                     f"{model} {src['name']} diagonal soap vs Part A")
        print(f"  diagonal {src['name']} verified against the val-rule selection table")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--sources", nargs="+", default=None,
                   help="short names (WW-AG WW-HC TE-Cap TE-Mag); default all")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    rows = []
    for model in args.models:
        cells = collect_cells(model)
        attn = {}
        for c in cells:
            weightings, bounds = aggregate_attn(C.attn_root(c["cfg"]), model, c["subset"],
                                                n_ranges=c["cfg"]["n_ranges"], device=args.device)
            attn[c["name"]] = (weightings, [f"{lo}-{hi}" for lo, hi in bounds])
        for src in cells:
            if args.sources and src["name"] not in args.sources:
                continue
            for tgt in cells:
                print(f"[{model}] {src['name']} -> {tgt['name']}")
                rows.extend(run_pair(src, tgt, model, attn, args.device))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    order = ["WW-AG", "WW-HC", "TE-Cap", "TE-Mag"]
    for model, g in df[df.row == "soap"].groupby("model"):
        pivot = (g.pivot_table(index="source", columns="target", values="step_acc_test") * 100)
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        print(f"\n=== {model} soap, val-selected (step acc %; rows=source) ===")
        print(pivot[[s for s in order if s in pivot.columns]].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
