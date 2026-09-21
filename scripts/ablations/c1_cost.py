#!/usr/bin/env python
"""C1 — inference cost per trajectory on Who&When: SOAP versus the prompt-based baselines.

Joins three sources into one long table, `results-ablations/c1_cost.tsv`:

  * the baselines' judge calls and token counts, replayed offline by
    `../attrib-prompting/scripts/cost_report.py` into `../attrib-prompting/reports/cost_ww.tsv`;
  * their wall-clock, harvested from the `wrote <dir> (N/N files, Xs)` lines that every
    predictor prints, kept in `../attrib-prompting/logs/open/*.log` and `logs/cost/*.log`
    (batched vLLM runs on one H200, so seconds per trajectory is throughput, not latency);
  * SOAP's own cost, computed here: one forward pass per scoreable step, so the tokens it
    reads are the sum over steps of the step's context length under the 8,192 budget
    (`main.data.build_step_input`, the extractor's exact input), and its time is the
    extraction pass, read from `logs/c1_timing_<subset>.log` when a timed run exists
    (`START <epoch>` / `END <epoch>` lines) and otherwise from the July run's file mtimes.

Scope: backbone qwen3.5-9b, judges qwen3.5-9b and gpt-4o, WW-AG and WW-HC, both GT
settings. `--print-table` emits the body of the manuscript's cost table (without GT).

    python scripts/ablations/c1_cost.py
    python scripts/ablations/c1_cost.py --print-table
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PROMPT = REPO.parent / "attrib-prompting"
OUT = REPO / "results-ablations" / "c1_cost.tsv"

SUBSETS = [("WW-AG", "algorithm-generated"), ("WW-HC", "hand-crafted")]
MODEL = "qwen3.5-9b"
# The July 2026 extraction, reconstructed from results-nogt/ww file mtimes
# (experiments/todo.md, S1 pre-flight): ~10 min for WW-AG, ~2 h 15 min for WW-HC.
MTIME_SECONDS = {"algorithm-generated": 600.0, "hand-crafted": 8100.0}
WROTE = re.compile(r"^\s*wrote (\S+)\s+\((\d+)/(\d+) files, ([\d.]+)s\)")


# ── SOAP tokens ──────────────────────────────────────────────────────────────
def soap_rows() -> list[dict]:
    from transformers import AutoTokenizer
    from main import config as C
    from main.data import build_step_input, iter_scoreable_steps, load_dataset

    rows = []
    for gt in (False, True):
        cfg = yaml.safe_load((REPO / "configs-main" / f"ww{'-gt' if gt else ''}.yaml").read_text())
        tok = AutoTokenizer.from_pretrained(str(REPO / cfg["model_paths"][MODEL]))
        for _, subset in SUBSETS:
            trajs = load_dataset(C.data_root(cfg), subset)
            toks, steps, hard = 0, 0, 0
            for traj in trajs:
                for t in iter_scoreable_steps(traj):
                    enc = build_step_input(traj, t, tok, max_tokens=cfg["max_tokens"],
                                           template_kwargs={"enable_thinking": False}, with_gt=gt)
                    toks += enc["input_ids"].shape[1]
                    steps += 1
                    hard += int(enc["hard_truncated"])
            n = len(trajs)
            secs, src = soap_seconds(subset)
            rows.append({"with_gt": gt, "judge": MODEL, "subset": subset, "method": "soap",
                         "n_traj": n, "n_costed": n, "n_mismatch": 0,
                         "calls_per_traj": 0, "prompt_tok_per_traj": round(toks / n, 1),
                         "out_tok_per_traj": 0, "forward_passes_per_traj": round(steps / n, 2),
                         "time_s_per_traj": round(secs / n, 2) if secs else "",
                         "time_source": src, "hard_truncated_steps": hard})
            print(f"  soap {'gt  ' if gt else 'nogt'} {subset:19s} n={n:3d} steps/traj={steps / n:6.2f}"
                  f" tokens/traj={toks / n:9.1f} time/traj={secs / n if secs else float('nan'):7.2f}s ({src})")
    return rows


def soap_seconds(subset: str) -> tuple[float | None, str]:
    log = REPO / "logs" / f"c1_timing_{subset}.log"
    if log.exists():
        text = log.read_text()
        m0, m1 = re.search(r"^START (\d+)", text, re.M), re.search(r"^END (\d+)", text, re.M)
        if m0 and m1:
            return float(int(m1.group(1)) - int(m0.group(1))), f"timed:{log.name}"
    return MTIME_SECONDS.get(subset), "mtime-2026-07"


# ── baseline wall-clock from the predictors' own summary lines ───────────────
def harvest_times() -> dict[tuple[bool, str, str, str], tuple[float, str]]:
    """(with_gt, judge, subset, method) -> (seconds per trajectory, log name)."""
    found: dict[tuple, tuple[float, float, str]] = {}
    logs = sorted(list((PROMPT / "logs" / "open").glob("*.log"))
                  + list((PROMPT / "logs" / "cost").glob("*.log")), key=lambda p: p.stat().st_mtime)
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            m = WROTE.match(line)
            if not m:
                continue
            path, done, total, secs = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
            parts = Path(path).parts
            if len(parts) != 5 or parts[1] != "ww" or done != total:
                continue
            root, _, subset, judge, method = parts
            with_gt = not root.endswith("-nogt")
            key = (with_gt, judge, subset, method)
            found[key] = (secs / total, log.stat().st_mtime, log.name)   # last complete run wins
    return {k: (v[0], v[2]) for k, v in found.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-tsv", default=str(PROMPT / "reports" / "cost_ww.tsv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()

    cost = pd.read_csv(args.cost_tsv, sep="\t")
    cost = cost[cost.judge.isin([MODEL, "gpt-4o"])].copy()
    times = harvest_times()
    cost["forward_passes_per_traj"] = ""
    cost["time_s_per_traj"] = ""
    cost["time_source"] = ""
    cost["hard_truncated_steps"] = ""
    for i, r in cost.iterrows():
        hit = times.get((bool(r.with_gt), r.judge, r.subset, r.method))
        if hit and r.judge == MODEL:
            cost.at[i, "time_s_per_traj"] = round(hit[0], 2)
            cost.at[i, "time_source"] = hit[1]
    df = pd.concat([cost, pd.DataFrame(soap_rows())], ignore_index=True)
    cols = ["with_gt", "judge", "subset", "method", "n_traj", "n_costed", "n_mismatch",
            "calls_per_traj", "prompt_tok_per_traj", "out_tok_per_traj", "forward_passes_per_traj",
            "time_s_per_traj", "time_source", "prompt_tok_per_call", "truncate_prompt_tokens",
            "step_mode", "hard_truncated_steps"]
    df = df.reindex(columns=cols).fillna("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"  wrote {args.out}  ({len(df)} rows)")
    if args.print_table:
        print_table(df)
    return 0


NAMES = [("all_at_once", "All-at-Once"), ("step_by_step", "Step-by-Step"),
         ("binary_search", "Binary Search"), ("correct", "CORRECT"), ("chief", "CHIEF"),
         ("errorprobe_paper", "ErrorProbe"), ("soap", r"\soap{}")]


def print_table(df: pd.DataFrame) -> None:
    """LaTeX body of tab:cost, without GT: Calls / Prompt tok. / Out. tok. / Time per subset."""
    d = df[~df.with_gt.astype(bool)]

    def cells(judge, method, subset):
        h = d[(d.judge == judge) & (d.method == method) & (d.subset == subset)]
        if h.empty or h.iloc[0]["n_costed"] in ("", 0):
            return [r"\PH"] * 4
        r = h.iloc[0]
        fmt = lambda v, nd=0: r"\PH" if v == "" else (f"{float(v):,.{nd}f}")
        t = r"--" if judge != MODEL else fmt(r["time_s_per_traj"], 1)
        return [fmt(r["calls_per_traj"], 1), fmt(r["prompt_tok_per_traj"]), fmt(r["out_tok_per_traj"]), t]

    for judge, title in (("gpt-4o", "GPT-4o"), (MODEL, "Qwen3.5-9B")):
        print(rf"\rowcolor{{Gray}}\multicolumn{{9}}{{c}}{{\textit{{\textbf{{{title}}}}}}} \\")
        for stem, name in NAMES:
            if stem == "soap" and judge != MODEL:
                continue
            row = cells(judge, stem, "algorithm-generated") + cells(judge, stem, "hand-crafted")
            print(f"{name:14s} & " + " & ".join(row) + r" \\")


if __name__ == "__main__":
    sys.exit(main())
