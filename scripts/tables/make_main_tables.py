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

Only the two SOAP rows are filled. Every other method is left blank for you to complete.

SEEDS. Each cell uses the triple frozen in `configs-main/`, which came from the
48/18-triple sweep under the `sum-diff` rule. A number is the mean over that triple's
three seeds. CE is the macro-average over the seven correct-error subsets, matching how
the manuscript forms that column.

    python scripts/tables/make_main_tables.py
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
PROMPT_ROWS = [("All-at-Once", "", "", None), ("Step-by-Step", "", "", None),
               ("Binary Search", "", "", None), ("CORRECT", "cmark", "cmark", None),
               ("RAFFLES", "", "", None), ("CHIEF", "", "", None)]
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


def build(sel: pd.DataFrame, gt: bool, strategy: str) -> list[list[str]]:
    seeds = {ds: load_seeds(ds, gt) for ds in ("ww", "traceelephant", "correct-error")}
    out = [HEADER, ["Backbone: GPT-4o", "", "", "", "", "", "", ""],
           ["Prompt-based methods", "", "", "", "", "", "", ""]]
    for method, ad, sup, _ in PROMPT_ROWS:
        out.append([method, ad, sup] + [""] * len(COLUMNS))

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

    for gt, stem in ((False, "table1_without_gt"), (True, "table2_with_gt")):
        rows = build(sel, gt, args.strategy)
        path = out_dir / f"{stem}.tsv"
        with open(path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
        filled = sum(1 for r in rows if r[0].startswith("SOAP") and any(r[3:]))
        blank = sum(1 for r in rows[1:] if not r[0].startswith(("Backbone:", "Prompt-",
                                                               "Trained ", "Representation"))
                    and not r[0].startswith("SOAP"))
        print(f"  {path}   {filled} SOAP rows filled, {blank} rows left for you")

        pv = out_dir / f"{stem}_seeds.tsv"
        with open(pv, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(seed_provenance(gt))
        print(f"  {pv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
