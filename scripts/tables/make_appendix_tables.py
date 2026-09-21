"""Build the appendix tables as TSV, so the LaTeX copies numbers from files, never from memory.

Every table joins result files that already exist; nothing here runs a model. The
five manuscript columns and the CE macro-average follow `make_main_tables.py`. All
accuracies are percentages with two decimals; a blank cell means no result exists.

Outputs, under tables/:
  appendix_agent_without_gt.tsv   agent-level counterpart of Table 1 (all judges/backbones)
  appendix_std_without_gt.tsv     mean ± std over the frozen triple for every Table-1 row
  appendix_with_gt_full.tsv       the with-GT grid: GPT-4o prompting, open-judge partial
                                  cells, OAT/StepFinder, SOAP — both backbones, five subsets
  appendix_valsel.tsv             test-selected vs validation-selected SOAP/base, all cells
  appendix_anchors.tsv            the selected configuration per cell (both GT trees, 14B/27B)
  appendix_ranked.tsv             step@k / agent@k / MRR for SOAP, base, OAT, StepFinder
                                  (needs results-ablations/a9_ranked.tsv)
  appendix_pooling.tsv            mean vs last-token pooling (needs results-ablations/a8_pooling.tsv)

    python scripts/tables/make_appendix_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tables"
ABL = REPO / "results-ablations"

COLUMNS = [("WW-AG", "ww", "algorithm-generated"), ("WW-HC", "ww", "hand-crafted"),
           ("CE", "correct-error", None), ("TE-Cap", "traceelephant", "captain"),
           ("TE-Mag", "traceelephant", "magentic")]
COLS = [c for c, _, _ in COLUMNS]
DATASETS = ["ww", "traceelephant", "correct-error"]
BACKBONES = [("Qwen3.5-9B", "qwen3.5-9b"), ("DeepSeek-R1-Distill-Llama-8B", "deepseek-8b")]
JUDGES = [("GPT-4o", "gpt-4o"), ("GPT-5", "gpt-5")]
# ErrorProbe = the paper's tag -> trace -> team pipeline (`errorprobe_paper`). On the
# Qwen3.5-9B judge that directory holds the handicapped-decoding run (512 new tokens,
# temperature 1.0, top-p 0.95, 16k window), the manuscript's row since 2026-09-06.
PROMPT_ROWS = [("All-at-Once", "all_at_once"), ("Step-by-Step", "step_by_step"),
               ("Binary Search", "binary_search"), ("CORRECT", "correct"),
               ("CHIEF", "chief"), ("RAFFLES", "raffles"), ("ErrorProbe", "errorprobe_paper")]
RB_ROWS = [("StepFinder", "stepfinder"), ("OAT", "oat")]


def pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100.0 * x:.2f}"


def ms(mean, std) -> str:
    return "" if mean is None or np.isnan(mean) else f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def tree(gt: bool) -> str:
    return "results-gt" if gt else "results-nogt"


def load_cfg(ds: str, gt: bool) -> dict:
    return yaml.safe_load((REPO / "configs-main" / f"{ds}{'-gt' if gt else ''}.yaml").read_text())


def selection(gt: bool) -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        df = pd.read_csv(REPO / tree(gt) / ds / "select" / "selection.tsv", sep="\t")
        df.insert(0, "dataset", ds)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def macro(df: pd.DataFrame, ds: str, subset: str | None, value: str, **flt) -> float | None:
    """One manuscript cell: the value itself, or the macro-average over CE's subsets."""
    sel = df[df["dataset"] == ds]
    for k, v in flt.items():
        sel = sel[sel[k] == v]
    if subset is not None:
        sel = sel[sel["subset"] == subset]
    if sel.empty:
        return None
    if subset is None and len(sel["subset"].unique()) != 7:
        return None
    return float(sel.groupby("subset")[value].mean().mean())


def per_seed_soap(gt: bool, model: str, ds: str, subset: str, row: str) -> pd.Series:
    """The per-seed test step accuracy of a selected config, from the sweep table."""
    sel = selection(gt)
    r = sel[(sel.dataset == ds) & (sel.model == model) & (sel.subset == subset) & (sel.row == row)]
    assert len(r) == 1, (gt, model, ds, subset, row)
    r = r.iloc[0]
    sw = pd.read_csv(REPO / tree(gt) / ds / "sweep" / model / subset / "sweep.tsv", sep="\t",
                     dtype={"w": str, "layer_range": str})
    sw = sw[(sw.position == r.position) & (sw.c_begin == r.c_begin) & (sw.c_end == r.c_end)]
    if row == "svd":
        sw = sw[sw.strategy == "base"]
    else:
        sw = sw[(sw.strategy == row) & (sw.layer_range == str(r.layer_range))
                & (np.isclose(sw.gamma.astype(float), float(r.gamma)))
                & (sw.w.map(lambda x: x[:-2] if x.endswith(".0") else x) == str(r.w).replace(".0", ""))]
    seeds = [int(s) for s in str(r.seeds).split(",")]
    sw = sw[sw.seed.isin(seeds)]
    assert len(sw) == len(seeds), (gt, model, ds, subset, row, len(sw))
    out = sw.set_index("seed")["step_acc_test@1"].astype(float)
    assert abs(out.mean() - float(r.step_acc_test)) < 1e-9
    return out


# ── agent-level and std tables for Table 1 ───────────────────────────────────
def table1_like(metric: str, with_std: bool) -> list[list[str]]:
    pr_seed = pd.read_csv(REPO / "results-prompting" / "by_seed.tsv", sep="\t")
    pr_col = pd.read_csv(REPO / "results-prompting" / "by_column.tsv", sep="\t")
    b1_seed = pd.read_csv(ABL / "b1_rb_baselines" / "by_seed.tsv", sep="\t")
    sel = selection(False)
    rows = [["Backbone", "Method"] + COLS]

    def prompt_cell(judge, stem, ds, subset):
        s = pr_seed[(~pr_seed.with_gt) & (pr_seed.judge == judge) & (pr_seed.method == stem)
                    & (pr_seed.dataset == ds)]
        if subset is not None:
            s = s[s.subset == subset]
        if s.empty or (subset is None and s.subset.nunique() != 7):
            return ""
        per_seed = s.groupby("seed")[metric].mean()        # CE: macro over subsets per seed
        return ms(per_seed.mean(), per_seed.std(ddof=0)) if with_std else pct(per_seed.mean())

    def rb_cell(judge, family, ds, subset):
        s = b1_seed[(~b1_seed.with_gt) & (b1_seed.judge == judge) & (b1_seed.family == family)
                    & (b1_seed.dataset == ds)]
        if subset is not None:
            s = s[s.subset == subset]
        if s.empty:
            return ""
        per_seed = s.groupby(["seed", "train_seed"])[metric].mean().groupby("seed").mean()
        return ms(per_seed.mean(), per_seed.std(ddof=0)) if with_std else pct(per_seed.mean())

    def soap_cell(model, ds, subset, row):
        subs = [subset] if subset else sorted(sel[sel.dataset == ds].subset.unique())
        per_seed = []
        for sub in subs:
            if with_std or metric == "step_acc":
                per_seed.append(per_seed_soap(False, model, ds, sub, row))
            else:
                r = sel[(sel.dataset == ds) & (sel.model == model) & (sel.subset == sub)
                        & (sel.row == row)].iloc[0]
                per_seed.append(pd.Series([float(r.agent_acc_test)] * 3))
        v = pd.concat(per_seed, axis=1).mean(axis=1) if len(per_seed) > 1 else per_seed[0]
        return ms(v.mean(), v.std(ddof=0)) if with_std else pct(v.mean())

    for disp, judge in JUDGES:
        for name, stem in PROMPT_ROWS:
            rows.append([disp, name] + [prompt_cell(judge, stem, ds, sub) for _, ds, sub in COLUMNS])
    for disp, model in BACKBONES:
        for name, stem in PROMPT_ROWS:
            rows.append([disp, name] + [prompt_cell(model, stem, ds, sub) for _, ds, sub in COLUMNS])
        for name, fam in RB_ROWS:
            rows.append([disp, name] + [rb_cell(model, fam, ds, sub) for _, ds, sub in COLUMNS])
        rows.append([disp, "SOAP (w/o rescoring)"]
                    + [soap_cell(model, ds, sub, "svd") for _, ds, sub in COLUMNS])
        rows.append([disp, "SOAP"] + [soap_cell(model, ds, sub, "backprop") for _, ds, sub in COLUMNS])
    return rows


# ── the with-GT grid ─────────────────────────────────────────────────────────
def with_gt_full() -> list[list[str]]:
    pr = pd.read_csv(REPO / "results-prompting" / "by_column.tsv", sep="\t")
    b1 = pd.read_csv(ABL / "b1_rb_baselines" / "by_column_mean_over_train_seeds.tsv", sep="\t")
    sel = selection(True)
    rows = [["Backbone", "Method"] + COLS]

    def pr_cell(judge, stem, col):
        h = pr[(pr.with_gt) & (pr.judge == judge) & (pr.method == stem) & (pr.column == col)]
        return "" if h.empty else pct(float(h.step_acc.iloc[0]))

    for disp, judge in [("GPT-4o", "gpt-4o")] + [(d, m) for d, m in BACKBONES]:
        for name, stem in PROMPT_ROWS:
            rows.append([disp, name] + [pr_cell(judge, stem, c) for c in COLS])
        if judge == "gpt-4o":
            continue
        for name, fam in RB_ROWS:
            cells = []
            for c in COLS:
                h = b1[(b1.with_gt) & (b1.judge == judge) & (b1.family == fam) & (b1.column == c)]
                cells.append("" if h.empty else pct(float(h.step_mean.iloc[0])))
            rows.append([disp, name] + cells)
        for name, row in (("SOAP (w/o rescoring)", "svd"), ("SOAP", "backprop")):
            rows.append([disp, name] + [pct(macro(sel, ds, sub, "step_acc_test", model=judge, row=row))
                                        for _, ds, sub in COLUMNS])
    return rows


# ── validation-selected vs test-selected ─────────────────────────────────────
def valsel() -> list[list[str]]:
    e3 = pd.read_csv(ABL / "e3_valsel_indist.tsv", sep="\t")
    e3 = e3[e3.row.isin(["svd", "backprop"])].copy()
    ds_of = {"algorithm-generated": "ww", "hand-crafted": "ww",
             "captain": "traceelephant", "magentic": "traceelephant"}
    e3["dataset"] = e3.subset.map(lambda s: ds_of.get(s, "correct-error"))
    rows = [["Setting", "Backbone", "Method", "Selection"] + COLS]
    for gt_tree, setting in (("nogt", "without GT"), ("gt", "with GT")):
        for disp, model in BACKBONES:
            for row, name in (("svd", "SOAP (w/o rescoring)"), ("backprop", "SOAP")):
                for col_val, label in (("tsel_step_acc_test", "test"), ("step_acc_test", "validation")):
                    cells = [pct(macro(e3, ds, sub, col_val, tree=gt_tree, model=model, row=row))
                             for _, ds, sub in COLUMNS]
                    rows.append([setting, disp, name, label] + cells)
    return rows


# ── the selected configurations ──────────────────────────────────────────────
def anchors() -> list[list[str]]:
    rows = [["setting", "backbone", "dataset", "subset", "seeds", "position", "band",
             "attention band", "gamma", "w", "step_acc_test", "agent_acc_test"]]
    for gt in (False, True):
        sel = selection(gt)
        for _, r in sel[sel.row == "backprop"].sort_values(["dataset", "model", "subset"]).iterrows():
            w = "" if pd.isna(r.w) else str(r.w).replace(".0", "")
            lr = "" if pd.isna(r.layer_range) else str(r.layer_range)
            rows.append(["with GT" if gt else "without GT", r.model, r.dataset, r.subset, r.seeds,
                         r.position, f"[{int(r.c_begin)}, {int(r.c_end)})", lr,
                         f"{float(r.gamma):.1f}", w, pct(float(r.step_acc_test)),
                         pct(float(r.agent_acc_test))])
    return rows


# ── ranked metrics (A9) and pooling (A8) ─────────────────────────────────────
def ranked() -> list[list[str]] | None:
    path = ABL / "a9_ranked.tsv"
    if not path.exists():
        return None
    a9 = pd.read_csv(path, sep="\t")
    metrics = ["step@1", "step@3", "step@5", "agent@1", "agent@3", "agent@5", "mrr"]
    rows = [["Setting", "Backbone", "Method", "Metric"] + COLS]
    names = {"oat": "OAT", "stepfinder": "StepFinder", "base": "SOAP (w/o rescoring)", "soap": "SOAP"}
    for gt, setting in ((False, "without GT"), (True, "with GT")):
        for disp, model in BACKBONES:
            for method in ("oat", "stepfinder", "base", "soap"):
                for m in metrics:
                    cells = [pct(macro(a9, ds, sub, m, with_gt=gt, backbone=model, method=method))
                             for _, ds, sub in COLUMNS]
                    rows.append([setting, disp, names[method], m] + cells)
    return rows


def pooling() -> list[list[str]] | None:
    path = ABL / "a8_pooling.tsv"
    if not path.exists():
        return None
    a8 = pd.read_csv(path, sep="\t", dtype={"w": str, "layer_range": str, "position": str})
    rows = [["Backbone", "Pooling", "Method", "WW-AG", "WW-HC", "TE-Cap", "TE-Mag",
             "config WW-AG", "config WW-HC", "config TE-Cap", "config TE-Mag"]]
    cols = [("WW-AG", "algorithm-generated"), ("WW-HC", "hand-crafted"),
            ("TE-Cap", "captain"), ("TE-Mag", "magentic")]
    for disp, model in BACKBONES:
        for pool in ("mean", "last"):
            for row, name in (("base", "SOAP (w/o rescoring)"), ("soap", "SOAP")):
                cells, cfgs = [], []
                for _, sub in cols:
                    h = a8[(a8.model == model) & (a8.pooling == pool) & (a8.row == row) & (a8.subset == sub)]
                    if h.empty:
                        cells.append(""); cfgs.append(""); continue
                    h = h.iloc[0]
                    cells.append(pct(float(h.step_acc_test)))
                    cfg = f"{h.position} [{int(h.c_begin)},{int(h.c_end)})"
                    if row == "soap":
                        cfg += f" {h.layer_range} γ={float(h.gamma):.1f} w={str(h.w).replace('.0', '')}"
                    cfgs.append(cfg)
                rows.append([disp, pool, name] + cells + cfgs)
    return rows


def write(name: str, rows: list[list[str]] | None) -> None:
    if rows is None:
        print(f"  [skip] {name}: source missing")
        return
    path = OUT / name
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, sep="\t", index=False)
    print(f"  {path}  ({len(rows) - 1} rows)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write("appendix_agent_without_gt.tsv", table1_like("agent_acc", with_std=False))
    write("appendix_std_without_gt.tsv", table1_like("step_acc", with_std=True))
    write("appendix_with_gt_full.tsv", with_gt_full())
    write("appendix_valsel.tsv", valsel())
    write("appendix_anchors.tsv", anchors())
    write("appendix_ranked.tsv", ranked())
    write("appendix_pooling.tsv", pooling())
    return 0


if __name__ == "__main__":
    sys.exit(main())
