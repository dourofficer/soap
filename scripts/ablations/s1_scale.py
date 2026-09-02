"""S1 / fig:scale — scalability of SOAP and the representation-based baselines.

Four backbones of the Qwen family, two WW subsets, without GT, on the frozen triples.
Nothing is trained or extracted here: SOAP's rows are read from the per-seed sweep
tables that `main sweep` wrote for each backbone (`results-nogt/ww/sweep/`), at the
configuration `main select` chose (`select/selection.tsv`); the baselines' rows are
scored from the predictions in `../attrib-prompting/outputs-rb-nogt/ww/` through the
B1 path (`scripts/prompting/evaluate.py`'s rules), one run per training seed.

Stages (run in order; `merge` needs the other two):
  soap       -> results-ablations/s1_parts/soap.tsv       (per seed)
  baselines  -> results-ablations/s1_parts/baselines.tsv  (per training seed, triple mean)
  merge      -> results-ablations/s1_scale.tsv            (one row per method x backbone x subset)
                results-ablations/s1_scale_summary.tsv    ("mean (std)" strings, ready to paste)

Sanity checks, asserted in `merge`:
  (i)   the 9B SOAP / base cells reproduce Table 1 (tables/table1_without_gt.tsv);
  (ii)  the 9B OAT / StepFinder cells reproduce B1 (results-ablations/b1_rb_baselines/);
  (iii) no baseline cell is missing a prediction.

    python scripts/ablations/s1_scale.py [--stage soap|baselines|merge]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from main import config as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", REPO / "scripts/prompting/evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CONFIG = REPO / "configs-main/ww.yaml"
PARTS = REPO / "results-ablations/s1_parts"
OUT = REPO / "results-ablations/s1_scale.tsv"
SUMMARY = REPO / "results-ablations/s1_scale_summary.tsv"

PARAMS_B = {"qwen3.5-4b": 4, "qwen3.5-9b": 9, "qwen3-14b": 14, "qwen3.5-27b": 27}
SOAP_BACKBONES = ["qwen3.5-9b", "qwen3-14b", "qwen3.5-27b"]      # extracted for S1
BASELINE_BACKBONES = list(PARAMS_B)                              # predicted by attrib-prompting
FAMILIES = ["oat", "stepfinder", "stepfinder-tsel", "stepfinder-pca", "stepfinder-pca-tsel"]
TRAIN_SEEDS = range(42, 47)
COLUMNS = ["method", "backbone", "params_b", "subset", "step", "step_sd", "agent", "step_val",
           "position", "band", "layer_range", "gamma", "w", "n"]


# ------------------------------------------------------------------------------ soap
def soap_rows(cfg) -> pd.DataFrame:
    """Per-seed test/val accuracy of the selected SOAP config and of its gamma=0 base."""
    sel = pd.read_csv(C.select_dir(cfg) / "selection.tsv", sep="\t", dtype={"w": str})
    rows = []
    for model in SOAP_BACKBONES:
        for subset in cfg["subsets"]:
            bp = sel[(sel.model == model) & (sel.subset == subset) & (sel.row == "backprop")]
            assert len(bp) == 1, f"no backprop selection for {model}/{subset}"
            bp = bp.iloc[0]
            sw = pd.read_csv(C.sweep_dir(cfg, model, subset) / "sweep.tsv", sep="\t",
                             dtype={"w": str, "layer_range": str})
            at_cfg = sw[(sw.position == bp.position) & (sw.c_begin == bp.c_begin)
                        & (sw.c_end == bp.c_end)]
            base = at_cfg[at_cfg.strategy == "base"]
            soap = at_cfg[(at_cfg.strategy == "backprop") & (at_cfg.layer_range == str(bp.layer_range))
                          & np.isclose(at_cfg.gamma, bp.gamma) & (at_cfg.w == str(bp.w))]
            seeds = [int(s) for s in str(bp.seeds).split(",")]
            for method, part in (("soap-base", base), ("soap", soap)):
                part = part[part.seed.isin(seeds)].drop_duplicates("seed")
                assert sorted(part.seed) == sorted(seeds), f"{method} {model}/{subset}: seeds {list(part.seed)}"
                for _, r in part.iterrows():
                    rows.append(dict(method=method, backbone=model, subset=subset, seed=int(r.seed),
                                     step=r["step_acc_test@1"], agent=r["agent_acc_test@1"],
                                     step_val=r["step_acc_val@1"], position=bp.position,
                                     band=f"[{bp.c_begin},{bp.c_end})",
                                     layer_range="" if method == "soap-base" else bp.layer_range,
                                     gamma=0.0 if method == "soap-base" else bp.gamma,
                                     w="" if method == "soap-base" else bp.w))
            # the selection row itself must be the mean of what we just read
            got = soap[soap.seed.isin(seeds)]["step_acc_test@1"].mean()
            assert abs(got - bp.step_acc_test) < 1e-9, f"{model}/{subset}: {got} vs {bp.step_acc_test}"
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- baselines
def baseline_rows(cfg) -> pd.DataFrame:
    """Triple-mean accuracy of every baseline run (one row per training seed)."""
    rows = []
    for subset in cfg["subsets"]:
        seeds = C.seeds_for(cfg, subset)
        ids = {s: ev.test_ids("ww", subset, "results-nogt", cfg["splits"], s) for s in seeds}
        for model in BASELINE_BACKBONES:
            for fam in FAMILIES:
                for ts in TRAIN_SEEDS:
                    preds = ev.read_cell("outputs-rb-nogt", "ww", subset, model, f"{fam}.s{ts}")
                    assert preds, f"no predictions: {subset}/{model}/{fam}.s{ts}"
                    per = [ev.score(preds, ids[s]) for s in seeds]
                    rows.append(dict(method=fam, backbone=model, subset=subset, train_seed=ts,
                                     n_missing=sum(p["n_missing"] for p in per),
                                     step=np.mean([p["step_acc"] for p in per]),
                                     agent=np.mean([p["agent_acc"] for p in per])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ merge
def merge() -> pd.DataFrame:
    soap = pd.read_csv(PARTS / "soap.tsv", sep="\t", dtype={"w": str, "layer_range": str, "band": str},
                       keep_default_na=False)
    base = pd.read_csv(PARTS / "baselines.tsv", sep="\t")
    assert int(base.n_missing.sum()) == 0, "baseline cells with missing predictions"        # (iii)

    keys = ["method", "backbone", "subset"]
    s = soap.groupby(keys, sort=False).agg(
        step=("step", "mean"), step_sd=("step", "std"), agent=("agent", "mean"),
        step_val=("step_val", "mean"), position=("position", "first"), band=("band", "first"),
        layer_range=("layer_range", "first"), gamma=("gamma", "first"), w=("w", "first"),
        n=("seed", "count")).reset_index()
    b = base.groupby(keys, sort=False).agg(
        step=("step", "mean"), step_sd=("step", "std"), agent=("agent", "mean"),
        n=("train_seed", "count")).reset_index()
    b["method"] = pd.Categorical(b.method, FAMILIES)
    b = b.sort_values(["method", "backbone", "subset"]).astype({"method": str})
    df = pd.concat([s, b], ignore_index=True)
    df["params_b"] = df.backbone.map(PARAMS_B)
    for c in ("step", "step_sd", "agent", "step_val"):
        df[c] = 100 * df[c]
    df = df[COLUMNS]

    # (i) Table 1's 9B cells
    t1 = pd.read_csv(REPO / "tables/table1_without_gt.tsv", sep="\t")
    qwen = t1.index[t1.Method == "Backbone: Qwen3.5-9B"][0]
    want = {"soap": t1[(t1.index > qwen) & (t1.Method == "SOAP")].iloc[0],
            "soap-base": t1[(t1.index > qwen) & (t1.Method == "SOAP (w/o rescoring)")].iloc[0]}
    for method, row in want.items():
        for subset, col in (("algorithm-generated", "WW-AG"), ("hand-crafted", "WW-HC")):
            got = df[(df.method == method) & (df.backbone == "qwen3.5-9b") & (df.subset == subset)].step.iloc[0]
            assert abs(got - float(row[col])) < 5e-3, f"Table 1 {method} {col}: {got:.2f} vs {row[col]}"
    # (ii) B1's 9B cells
    b1 = pd.read_csv(REPO / "results-ablations/b1_rb_baselines/by_column_mean_over_train_seeds.tsv", sep="\t")
    b1 = b1[(~b1.with_gt) & (b1.judge == "qwen3.5-9b")]
    for fam in ("oat", "stepfinder"):
        for subset, col in (("algorithm-generated", "WW-AG"), ("hand-crafted", "WW-HC")):
            got = df[(df.method == fam) & (df.backbone == "qwen3.5-9b") & (df.subset == subset)].step.iloc[0]
            ref = 100 * b1[(b1.family == fam) & (b1.column == col)].step_mean.iloc[0]
            assert abs(got - ref) < 1e-4, f"B1 {fam} {col}: {got} vs {ref}"
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One 'mean (std)' string per cell; columns ordered by backbone size."""
    df = df.copy()
    df["cell"] = df.apply(lambda r: f"{r.step:.2f} ({r.step_sd:.1f})", axis=1)
    df["agent_cell"] = df.agent.map(lambda v: f"{v:.2f}")
    order = ["soap", "soap-base"] + FAMILIES
    cols = [(bb, PARAMS_B[bb]) for bb in BASELINE_BACKBONES]
    out = []
    for metric, key in (("step", "cell"), ("agent", "agent_cell")):
        for subset in df.subset.unique():
            for m in order:
                part = df[(df.method == m) & (df.subset == subset)].set_index("backbone")
                out.append({"metric": metric, "subset": subset, "method": m,
                            **{f"{p}B": (part[key].get(bb, "—")) for bb, p in cols},
                            "std_over": "seeds" if m.startswith("soap") else "training seeds"})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["soap", "baselines", "merge"], action="append")
    a = ap.parse_args()
    stages = a.stage or ["soap", "baselines", "merge"]
    cfg = C.load_config(CONFIG)
    PARTS.mkdir(parents=True, exist_ok=True)
    if "soap" in stages:
        soap_rows(cfg).to_csv(PARTS / "soap.tsv", sep="\t", index=False)
        print(f"  {PARTS / 'soap.tsv'}")
    if "baselines" in stages:
        baseline_rows(cfg).to_csv(PARTS / "baselines.tsv", sep="\t", index=False)
        print(f"  {PARTS / 'baselines.tsv'}")
    if "merge" in stages:
        df = merge()
        df.to_csv(OUT, sep="\t", index=False, float_format="%.4f")
        summary(df).to_csv(SUMMARY, sep="\t", index=False)
        print(f"  {OUT}  {len(df)} rows\n  {SUMMARY}")


if __name__ == "__main__":
    main()
