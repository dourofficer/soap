"""E3 Part A — validation-selected configs, in-distribution.

The protocol the manuscript claims: within a frozen triple, per (backbone, subset), the
reported config maximizes mean VALIDATION step accuracy over the three seeds and the
number reported is its TEST accuracy. Same two stages as Table 1 — base grid picks
(position, c_begin, c_end); the rescore grid, expanded ONLY for the val-winning base
config, picks (layer_range, gamma, w) per strategy.

The frozen sweep tables (results-{nogt,gt}/<ds>/sweep/*/sweep.tsv) already hold val
metrics for the dense base grid, so the base stage is a free re-read. Their rescore
grid, however, was expanded for the TEST-winning base config; wherever the val winner
differs, this runner recomputes the rescore grid for it (`main.sweep._rescore_pass`,
bit-identical plumbing). It writes the resulting tables into the val-rule tree
(results-{nogt,gt}-valsel/<ds>/) and lets `main.sweep.run_select` under
`select_rule=val` do the selection, so `python -m main select --config ... --set
select_rule=val` reproduces every number here from the tree alone.

Output: results-ablations/e3_valsel_indist.tsv — one row per (tree, backbone, subset,
row) with the val-selected config and its val/test metrics, and beside it the
test-selected row's metrics (from the frozen selection tables) so the optimism gap
is one subtraction.

    python scripts/ablations/e3_valsel_indist.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, norm_val, select_config                       # noqa: E402
from main import config as C                                                 # noqa: E402
from main.rescore import aggregate_attn                                      # noqa: E402
from main.sweep import (BASE_STRATEGY, BASE_SWEPT, RESCORE_SWEPT, _assert_gamma0_is_base,  # noqa: E402
                        _rescore_pass, rule_cols, run_select, sweep_cols)

OUT = RESULTS_DIR / "e3_valsel_indist.tsv"
CONFIGS = ["ww", "traceelephant", "correct-error"]
MODELS = ["qwen3.5-9b", "deepseek-8b"]
ROWS = ["svd", "backprop", "succ-strong", "succ-near"]


def same_config(a: dict, b: pd.Series | dict) -> bool:
    return all(norm_val(a[ax]) == norm_val(b[ax]) for ax in BASE_SWEPT)


def build_val_tree(cfg_val: dict, cfg_test: dict, device: str, force: bool) -> None:
    """Write the val-rule sweep table for every cell of ``cfg_val``."""
    ks = cfg_val["ks"]
    step_col, agent_col = rule_cols(cfg_val)
    for model in cfg_val["models"]:
        for subset in cfg_val["subsets"]:
            seeds = C.seeds_for(cfg_val, subset)
            out_dir = C.sweep_dir(cfg_val, model, subset)
            out = out_dir / "sweep.tsv"
            if out.exists() and not force:
                print(f"[skip] {out}")
                continue
            frozen = pd.read_csv(C.sweep_dir(cfg_test, model, subset) / "sweep.tsv", sep="\t")
            base_df = frozen[frozen.strategy == BASE_STRATEGY]
            best = select_config(base_df, BASE_SWEPT, seeds, step_col, agent_col)
            assert best is not None
            test_best = select_config(base_df, BASE_SWEPT, seeds,
                                      *rule_cols(cfg_test))
            print(f"[base] {model}/{subset}: val-sel {best['config']} "
                  f"(test {best['step']:.4f} val {best['step_val']:.4f}); "
                  f"test-sel {test_best['config']} (test {test_best['step']:.4f})")
            resc = frozen[frozen.strategy != BASE_STRATEGY]
            if same_config(best["config"], test_best["config"]):
                print("       same base winner -> rescore rows reused from the frozen table")
                resc_rows = resc.to_dict("records")
            else:
                rep_dir = C.reps_root(cfg_val) / model / subset
                data_dir = C.data_root(cfg_val) / subset
                weightings, bounds = aggregate_attn(C.attn_root(cfg_val), model, subset,
                                                    n_ranges=cfg_val["n_ranges"], device=device)
                labels = [f"{lo}-{hi}" for lo, hi in bounds]
                resc_rows = _rescore_pass(cfg_val, model, subset, seeds, best["config"],
                                          rep_dir, data_dir, weightings, labels, device)
            df = pd.DataFrame(base_df.to_dict("records") + resc_rows)[sweep_cols(ks)]
            _assert_gamma0_is_base(df, ks)
            df = df.sort_values(step_col, ascending=False, kind="mergesort")
            C.check_stamp(cfg_val, out_dir, force=True)
            df.to_csv(out, sep="\t", index=False)
            print(f"  wrote {out}  ({len(df)} rows)")


def join_selections(cfg_val: dict, cfg_test: dict, tree: str) -> list[dict]:
    val = pd.read_csv(C.select_dir(cfg_val) / "selection.tsv", sep="\t")
    tst = pd.read_csv(C.select_dir(cfg_test) / "selection.tsv", sep="\t")
    rows = []
    for model in cfg_val["models"]:
        for subset in cfg_val["subsets"]:
            for row in ROWS:
                v = val[(val.model == model) & (val.subset == subset) & (val.row == row)]
                t = tst[(tst.model == model) & (tst.subset == subset) & (tst.row == row)]
                assert len(v) == 1 and len(t) == 1, (tree, model, subset, row)
                v, t = v.iloc[0], t.iloc[0]
                # Sanity: on the dense base grid, val-selection cannot lose on val nor
                # win on test. Strategy rows carry no such guarantee — their grids sit
                # on DIFFERENT base configs (staging), so they are recorded, not asserted.
                if row == "svd":
                    assert v.step_acc_val >= t.step_acc_val - 1e-12, (tree, model, subset, row)
                    assert v.step_acc_test <= t.step_acc_test + 1e-12, (tree, model, subset, row)
                rows.append({"tree": tree, "model": model, "subset": subset, "row": row,
                             "seeds": v.seeds,
                             **{ax: v[ax] for ax in BASE_SWEPT + RESCORE_SWEPT},
                             "step_acc_test": v.step_acc_test, "agent_acc_test": v.agent_acc_test,
                             "step_acc_val": v.step_acc_val, "agent_acc_val": v.agent_acc_val,
                             **{f"tsel_{ax}": t[ax] for ax in BASE_SWEPT + RESCORE_SWEPT},
                             "tsel_step_acc_test": t.step_acc_test,
                             "tsel_agent_acc_test": t.agent_acc_test,
                             "tsel_step_acc_val": t.step_acc_val,
                             "tsel_agent_acc_val": t.agent_acc_val,
                             "gap_step_test": t.step_acc_test - v.step_acc_test})
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    rows = []
    for name in CONFIGS:
        for tree, path in (("nogt", f"configs-main/{name}.yaml"),
                           ("gt", f"configs-main/{name}-gt.yaml")):
            cfg_test = C.load_config(C.REPO_ROOT / path)
            cfg_val = C.load_config(C.REPO_ROOT / path, ["select_rule=val"])
            for cfg in (cfg_test, cfg_val):
                cfg["models"] = [m for m in cfg["models"] if m in MODELS]
                cfg["device"] = args.device
            print(f"=== {name} [{tree}] ===")
            build_val_tree(cfg_val, cfg_test, args.device, args.force)
            run_select(cfg_val)
            rows.extend(join_selections(cfg_val, cfg_test, tree))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    for tree in ("nogt", "gt"):
        g = df[(df.tree == tree) & (df.row.isin(["svd", "backprop"]))]
        piv = g.pivot_table(index=["model", "subset"], columns="row",
                            values=["step_acc_test", "tsel_step_acc_test"]) * 100
        print(f"\n=== {tree}: step acc % (val-selected | test-selected) ===")
        print(piv.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
