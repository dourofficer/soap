"""Build the manuscript's two main tables as TSV, with the SOAP rows filled.

The LaTeX in `manuscript/` is never touched. This writes `tables/table1_without_gt.tsv`
and `tables/table2_with_gt.tsv`, which carry EVERY row and column of the manuscript
tables so the .tex can read its numbers straight out of them.

Two deliberate departures from the .tex, both requested:

  * Both tables share ONE layout. The manuscript's Table 2 omits the DeepSeek-8B block
    and most prompting rows; here Table 2 mirrors Table 1 exactly, so a with-GT number
    is never missing just because the paper chose to move it to an appendix.
  * The `SOAP (w/o rescoring)` row is included. The manuscript commented it out, but it
    is the base score behind every SOAP number and is worth keeping alongside.

The SOAP rows and the prompting rows are filled. The prompting numbers come from
`results-prompting/by_column.tsv` (see `scripts/prompting/evaluate.py`), scored on the
SAME test splits as SOAP so the two are comparable. RAFFLES and CHIEF stay blank:
`../attrib-prompting` has no results for them. The trained attributors and StepFinder/OAT
stay blank too.

SEEDS. Each cell uses the triple frozen in `configs-main/`, which came from the
48/18-triple sweep under the `sum-diff` rule. A number is the mean over that triple's
three seeds. CE is the macro-average over the seven correct-error subsets, matching how
the manuscript forms that column.

    python scripts/tables/make_main_tables.py               # gpt-4o + gpt-5 tables
    python scripts/tables/make_main_tables.py --strategy succ-near
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
SELECTIONS = REPO / "results-sweep" / "selections_all.tsv"

COLUMNS = [("WW-AG", "ww", "algorithm-generated"),
           ("WW-HC", "ww", "hand-crafted"),
           ("CE", "correct-error", None),          # macro-average over its 7 subsets
           ("TE-Cap", "traceelephant", "captain"),
           ("TE-Mag", "traceelephant", "magentic")]

MODELS = [("Qwen3.5-9B", "qwen3.5-9b"), ("DeepSeek-R1-Distill-Llama-8B", "deepseek-8b")]

# (method, actual_data, supervised, fill) — marks copied from the .tex verbatim.
# The fourth field names the method stem in results-prompting, or None to leave blank.
PROMPT_ROWS = [("All-at-Once", "", "", "all_at_once"),
               ("Step-by-Step", "", "", "step_by_step"),
               ("Binary Search", "", "", "binary_search"),
               ("CORRECT", "cmark", "cmark", "correct"),
               ("RAFFLES", "", "", "raffles"), ("CHIEF", "", "", "chief")]

PROMPTING = REPO / "results-prompting" / "by_column.tsv"
JUDGE_DISPLAY = {"gpt-4o": "GPT-4o", "gpt-5": "GPT-5"}
TRAINED_ROWS = [("AgenTracer", "xmark", "cmark", None),
                ("GraphTracer", "xmark", "cmark", None)]
REPR_ROWS = [("StepFinder", "xmark", "xmark", None), ("OAT", "xmark", "xmark", None),
             ("SOAP (w/o rescoring)", "cmark", "xmark", "svd"),
             ("SOAP", "cmark", "xmark", "strategy")]

HEADER = ["Method", "Actual data", "Supervised"] + [c for c, _, _ in COLUMNS]


def load_seeds(dataset: str, gt: bool) -> dict[str, int]:
    """The frozen triple per subset, from configs-main/ — the single source of truth."""
    name = f"{dataset}-gt.yaml" if gt else f"{dataset}.yaml"
    cfg = yaml.safe_load((REPO / "configs-main" / name).read_text())
    return {sub: int(s[0]) for sub, s in cfg["seeds"].items()}


def cell_value(sel: pd.DataFrame, gt: bool, dataset: str, subset: str | None,
               model: str, row: str, seeds: dict[str, int]) -> float | None:
    """Step accuracy for one table cell, as a percentage.

    `subset=None` means the CE column: macro-average the seven subsets, each read at the
    triple frozen for it. All seven share a triple today, but averaging per-subset keeps
    this correct if they ever diverge.
    """
    subs = [subset] if subset else sorted(seeds)
    vals = []
    for sub in subs:
        hit = sel[(sel["with_gt"] == gt) & (sel["dataset"] == dataset)
                  & (sel["subset"] == sub) & (sel["model"] == model)
                  & (sel["row"] == row) & (sel["triple"] == seeds[sub])]
        if hit.empty:
            return None
        vals.append(float(hit.iloc[0]["step_acc_test"]))
    return 100.0 * sum(vals) / len(vals)


def prompting_value(pr: pd.DataFrame | None, gt: bool, judge: str, method: str,
                    column: str) -> str:
    """Step accuracy for one prompting cell, as a percentage."""
    if pr is None:
        return ""
    hit = pr[(pr["with_gt"] == gt) & (pr["judge"] == judge)
             & (pr["method"] == method) & (pr["column"] == column)]
    return "" if hit.empty else f"{100.0 * float(hit.iloc[0]['step_acc']):.2f}"


def build(sel: pd.DataFrame, gt: bool, strategy: str, judge: str,
          pr: pd.DataFrame | None) -> list[list[str]]:
    seeds = {ds: load_seeds(ds, gt) for ds in ("ww", "traceelephant", "correct-error")}
    out = [HEADER, [f"Backbone: {JUDGE_DISPLAY[judge]}", "", "", "", "", "", "", ""],
           ["Prompt-based methods", "", "", "", "", "", "", ""]]
    for method, ad, sup, stem in PROMPT_ROWS:
        cells = ([""] * len(COLUMNS) if stem is None
                 else [prompting_value(pr, gt, judge, stem, c) for c, _, _ in COLUMNS])
        out.append([method, ad, sup] + cells)

    for disp, model in MODELS:
        out.append([f"Backbone: {disp}", "", "", "", "", "", "", ""])
        out.append(["Trained LLM attributors", "", "", "", "", "", "", ""])
        for method, ad, sup, _ in TRAINED_ROWS:
            out.append([method, ad, sup] + [""] * len(COLUMNS))
        out.append(["Representation-based methods", "", "", "", "", "", "", ""])
        for method, ad, sup, fill in REPR_ROWS:
            cells = []
            for _, ds, sub in COLUMNS:
                if fill is None:
                    cells.append("")
                    continue
                row = "svd" if fill == "svd" else strategy
                v = cell_value(sel, gt, ds, sub, model, row, seeds[ds])
                cells.append("" if v is None else f"{v:.2f}")
            out.append([method, ad, sup] + cells)
    return out


def seed_provenance(gt: bool) -> list[list[str]]:
    rows = [["column", "dataset", "subset", "seeds"]]
    for name, ds, sub in COLUMNS:
        seeds = load_seeds(ds, gt)
        for s in ([sub] if sub else sorted(seeds)):
            t = seeds[s]
            rows.append([name, ds, s, f"{t},{t+1},{t+2}"])
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="backprop",
                   choices=["backprop", "succ-strong", "succ-near"])
    p.add_argument("--out-dir", default=str(REPO / "tables"))
    args = p.parse_args()

    if not SELECTIONS.exists():
        raise SystemExit(f"no {SELECTIONS}; run scripts/main/collect.py first")
    sel = pd.read_csv(SELECTIONS, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pr = pd.read_csv(PROMPTING, sep="\t") if PROMPTING.exists() else None
    if pr is None:
        print(f"  [warn] no {PROMPTING}; prompting rows will be blank "
              f"(run scripts/prompting/evaluate.py)")

    for gt, stem in ((False, "table1_without_gt"), (True, "table2_with_gt")):
        for judge in JUDGE_DISPLAY:
            rows = build(sel, gt, args.strategy, judge, pr)
            suffix = "" if judge == "gpt-4o" else f"_{judge.replace('-', '')}"
            path = out_dir / f"{stem}{suffix}.tsv"
            with open(path, "w", newline="") as f:
                csv.writer(f, delimiter="\t").writerows(rows)
            n = sum(1 for r in rows if any(r[3:]) and r[0] != "Method")
            print(f"  {path}   {n} data rows filled")

        pv = out_dir / f"{stem}_seeds.tsv"
        with open(pv, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(seed_provenance(gt))
        print(f"  {pv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
