#!/usr/bin/env python
"""C4 — join the cost measurements into the three tables of app:compute.

Inputs (all produced by the c4 timers and the attrib-prompting cost runs):
  results-ablations/c4_extract_cost/soap_<model>_<subset>.tsv   SOAP extraction, per trajectory
  results-ablations/c4_extract_cost/oat_<model>.tsv              OAT extraction, inference + setup
  results-ablations/c4_extract_cost/stepfinder_<model>.tsv       StepFinder extraction, both phases
  results-ablations/c4_extract_cost/soap_onepass_<model>.tsv     SOAP one pass over the trajectory
  results-ablations/c4_extract_cost/soap_stage2_<model>.tsv      SOAP SVD fit / score / rescore
  results-ablations/c4_extract_cost/rb_stage2_<model>.tsv        OAT / StepFinder scorer forward
  results-ablations/c4_extract_cost/soap_reference_stems_<model>.json
  ../attrib-prompting/logs/cost/{oat,stepfinder}-train-timing.log   training wall-clock, 5 seeds
  ../attrib-prompting/reports/latency/{<method>_<subset>.tsv, cost_sample_<subset>.tsv, ids_<subset>.json}
                                                                  one-trajectory-at-a-time judge runs

Outputs:
  results-ablations/c4_cost_setup.tsv       once per subset: data, extraction, operation
  results-ablations/c4_cost_inference.tsv   per trajectory: extraction, operation
  results-ablations/c4_cost_judges.tsv      per trajectory, the judges vs SOAP on the same sample
  --print-table                             the LaTeX bodies of tab:cost-rb and tab:cost-judges

    python scripts/ablations/c4_cost_tables.py --print-table
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROMPT = REPO.parent / "attrib-prompting"
IN = REPO / "results-ablations" / "c4_extract_cost"
OUT = REPO / "results-ablations"
MODEL = "qwen3.5-9b"
SUBSETS = [("WW-AG", "algorithm-generated"), ("WW-HC", "hand-crafted")]
JUDGE_METHODS = [("All-at-Once", "all_at_once", "all_at_once"),
                 ("Step-by-Step", "step_by_step", "step_by_step_batch"),
                 ("Binary Search", "binary_search", "binary_search"),
                 ("CORRECT", "correct", "correct"),
                 ("CHIEF", "chief", "chief"),
                 ("ErrorProbe", "errorprobe_paper", "errorprobe_paper")]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def train_seconds() -> dict:
    """Training wall-clock of the five seeds, from the timed re-runs."""
    out = {}
    oat = (PROMPT / "logs" / "cost" / "oat-train-timing.log")
    sf = (PROMPT / "logs" / "cost" / "stepfinder-train-timing.log")
    if oat.exists():
        m = re.search(r"OAT_TRAIN_TOTAL_SECONDS ([\d.]+)", oat.read_text())
        if m:
            out[("OAT", "algorithm-generated")] = out[("OAT", "hand-crafted")] = float(m.group(1))
    if sf.exists():
        for m in re.finditer(r"SF_TRAIN_TOTAL_SECONDS (\S+) ([\d.]+)", sf.read_text()):
            out[("StepFinder", m.group(1))] = float(m.group(2))
    return out


def soap_rows() -> pd.DataFrame:
    frames = [read(IN / f"soap_{MODEL}_{s}.tsv") for _, s in SUBSETS]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["passes"] = df.passes_act + df.passes_attn
    df["tokens"] = df.tokens_act + df.tokens_attn
    df["seconds"] = df.total_s
    df["peak_gb"] = df[["peak_act_gb", "peak_attn_gb"]].max(axis=1)
    df["traj"] = df.traj.astype(str)
    return df


def per_pass(df, sec="seconds", passes="passes"):
    """Seconds per forward pass, pooled over the frame (total seconds / total passes)."""
    return float(df[sec].sum() / df[passes].sum()) if len(df) and df[passes].sum() else None


def setup_table(soap, oat, sf, stage2_soap, trains) -> pd.DataFrame:
    stems = json.loads((IN / f"soap_reference_stems_{MODEL}.json").read_text()) \
        if (IN / f"soap_reference_stems_{MODEL}.json").exists() else {}
    rows = []
    for col, subset in SUBSETS:
        # OAT: the 103 MCP-Atlas successes, shared by every subset
        o = oat[oat.phase == "setup"]
        rows.append(dict(subset=col, method="OAT", corpus="MCP-Atlas successes", n_traj=len(o),
                         labels="task success", passes=int(o.passes.sum()), tokens=int(o.tokens.sum()),
                         extract_s=round(o.seconds.sum(), 1), peak_gb=round(o.peak_gb.max(), 2),
                         n_extracted=len(o),
                         stage2="PCA + neural CDE, 5 seeds", stage2_s=trains.get(("OAT", subset))))
        # StepFinder: the vendored regenerated failures of the subset's agent system
        s = sf[(sf.phase == "setup") & (sf.corpus == subset)]
        rows.append(dict(subset=col, method="StepFinder", corpus=f"regenerated failures ({subset})",
                         n_traj=len(s), labels="decisive step", passes=int(s.passes.sum()),
                         tokens=int(s.tokens.sum()), extract_s=round(s.seconds.sum(), 1),
                         peak_gb=round(s.peak_gb.max(), 2) if len(s) else None, n_extracted=len(s),
                         stage2="BiLSTM scorer, 5 seeds", stage2_s=trains.get(("StepFinder", subset))))
        # SOAP: the reference split (first seed of the frozen triple)
        ref = set(str(x) for x in stems.get(subset, []))
        p = soap[(soap.subset == subset) & soap.traj.isin(ref)]
        st = stage2_soap[(stage2_soap.subset == subset) & (stage2_soap.device == "cpu")]
        rows.append(dict(subset=col, method="SOAP", corpus="reference split (failures)",
                         n_traj=len(ref), labels="none", passes=int(p.passes.sum()),
                         tokens=int(p.tokens.sum()), extract_s=round(p.seconds.sum(), 1),
                         peak_gb=round(p.peak_gb.max(), 2) if len(p) else None, n_extracted=len(p),
                         stage2="SVD of the reference matrix",
                         stage2_s=float(st.svd_fit_s.iloc[0]) if len(st) else None))
    return pd.DataFrame(rows)


def onepass_rows() -> pd.DataFrame:
    """SOAP's one pass over the whole trajectory (c4_onepass_cost.py): the last step's
    activation pass, whose input is the trajectory in its context under the budget."""
    df = read(IN / f"soap_onepass_{MODEL}.tsv")
    if not df.empty:
        df["traj"] = df.traj.astype(str)
    return df


def inference_table(soap, oat, sf, stage2_soap, stage2_rb, onepass) -> pd.DataFrame:
    rows = []
    for col, subset in SUBSETS:
        for method, df in (("OAT", oat[(oat.phase == "inference") & (oat.corpus == subset)]),
                           ("StepFinder", sf[(sf.phase == "inference") & (sf.corpus == subset)]),
                           ("SOAP", soap[soap.subset == subset])):
            if df.empty:
                continue
            if method == "SOAP":
                st = stage2_soap[stage2_soap.subset == subset]
                ms = {d: float(st[st.device == d].score_ms_per_traj.iloc[0]
                               + st[st.device == d].rescore_ms_per_traj.iloc[0])
                      for d in ("cpu", "cuda") if len(st[st.device == d])}
                op = "band projection + rescoring"
            else:
                st = (stage2_rb[(stage2_rb.method == method) & (stage2_rb.subset == subset)]
                      if not stage2_rb.empty else stage2_rb)
                ms = {d: float(st[st.device == d].score_ms_per_traj.iloc[0])
                      for d in ("cpu", "cuda") if len(st) and len(st[st.device == d])}
                op = "neural CDE + conformal decoder" if method == "OAT" else "BiLSTM scorer"
            # The one-pass convention (manuscript, 2026-09-17): SOAP's extraction is charged
            # as ONE forward pass over the trajectory, since attention is causal and one pass
            # yields every step's states. `extract_s_per_pass` pools both stages of the
            # per-step extractor; `one_pass_s` is the timed whole-trajectory pass
            # (soap_onepass_*.tsv), or the per-pass time of the baselines' own extractors.
            if method == "SOAP":
                op_ = onepass[onepass.subset == subset]
                one_pass = float(op_.pass_s.mean()) if len(op_) else per_pass(df, "act_s", "passes_act")
            else:
                one_pass = per_pass(df)
            rows.append(dict(subset=col, method=method, n_traj=len(df),
                             passes=round(df.passes.mean(), 1), tokens=round(df.tokens.mean()),
                             extract_s=round(df.seconds.mean(), 2),
                             extract_s_per_pass=round(per_pass(df), 3),
                             one_pass_s=round(one_pass, 3),
                             peak_gb_mean=round(df.peak_gb.mean(), 2),
                             peak_gb_max=round(df.peak_gb.max(), 2),
                             weights_gb=round(df.weights_gb.iloc[0], 2),
                             stage2=op, stage2_ms_cpu=ms.get("cpu"), stage2_ms_cuda=ms.get("cuda")))
    return pd.DataFrame(rows)


def judges_table(soap, stage2_soap, onepass) -> pd.DataFrame:
    rows = []
    for col, subset in SUBSETS:
        st = stage2_soap[(stage2_soap.subset == subset) & (stage2_soap.device == "cpu")]
        scoring_ms = float(st.score_ms_per_traj.iloc[0] + st.rescore_ms_per_traj.iloc[0]) if len(st) else None
        ids_path = PROMPT / "reports" / "latency" / f"ids_{subset}.json"
        sample = read(PROMPT / "reports" / "latency" / f"cost_sample_{subset}.tsv")
        if not ids_path.exists():
            continue
        ids = [str(i) for i in json.loads(ids_path.read_text())]
        for name, stem, label in JUDGE_METHODS:
            lat = read(PROMPT / "reports" / "latency" / f"{label}_{subset}.tsv")
            tok = sample[sample.method == stem] if not sample.empty else pd.DataFrame()
            rows.append(dict(subset=col, method=name, n=len(lat),
                             seconds=round(lat.seconds.mean(), 2) if len(lat) else None,
                             calls=round(float(tok.calls_per_traj.iloc[0]), 1) if len(tok) else None,
                             prompt_tokens=round(float(tok.prompt_tok_per_traj.iloc[0])) if len(tok) else None,
                             generated_tokens=round(float(tok.out_tok_per_traj.iloc[0])) if len(tok) else None))
        p = soap[(soap.subset == subset) & soap.traj.isin(ids)]
        op_ = onepass[(onepass.subset == subset) & onepass.traj.isin(ids)]
        one_pass = float(op_.pass_s.mean()) if len(op_) else per_pass(p, "act_s", "passes_act")
        rows.append(dict(subset=col, method="SOAP", n=len(p),
                         seconds=round(p.seconds.mean(), 2) if len(p) else None,
                         calls=round(p.passes.mean(), 1) if len(p) else None,
                         prompt_tokens=round(p.tokens.mean()) if len(p) else None,
                         generated_tokens=0,
                         # one-pass convention: one activation pass of one step + CPU scoring
                         one_pass_s=round(one_pass, 3) if one_pass else None,
                         scoring_ms=round(scoring_ms, 2) if scoring_ms else None,
                         one_pass_plus_scoring_s=round(one_pass + scoring_ms / 1e3, 3)
                         if one_pass and scoring_ms else None))
    return pd.DataFrame(rows)


def fmt(v, nd=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    return f"{v:,.{nd}f}"


def print_tables(setup, infer, judges):
    print("% ---- tab:cost-rb, setup block (once per subset) ----")
    for _, r in setup.iterrows():
        print(f"{r.subset} & {r.method} & {r.n_traj} & {r.labels} & {fmt(r.passes)} & {fmt(r.tokens)} "
              f"& {fmt(r.extract_s)} & {fmt(r.peak_gb, 1)} & {r.stage2} & {fmt(r.stage2_s, 1)} \\\\")
    print("% ---- tab:cost-rb, inference block (per trajectory) ----")
    for _, r in infer.iterrows():
        print(f"{r.subset} & {r.method} & {fmt(r.passes, 1)} & {fmt(r.tokens)} & {fmt(r.extract_s, 2)} "
              f"& {fmt(r.peak_gb_max, 1)} & {r.stage2} & {fmt(r.stage2_ms_cpu, 1)} \\\\")
    print("% ---- tab:cost-inference (manuscript layout): fitting s/seed | s per forward pass | CPU scoring ms ----")
    for name in ["OAT", "StepFinder", "SOAP"]:
        fit, pp, sc = [], [], []
        for col, _ in SUBSETS:
            s_ = setup[(setup.method == name) & (setup.subset == col)]
            i_ = infer[(infer.method == name) & (infer.subset == col)]
            if s_.empty or i_.empty:
                fit.append("--"); pp.append("--"); sc.append("--"); continue
            s_, i_ = s_.iloc[0], i_.iloc[0]
            fit.append(fmt(s_.stage2_s / (5 if name != "SOAP" else 1), 2))
            pp.append(fmt(i_.one_pass_s, 2))
            sc.append(fmt(i_.stage2_ms_cpu, 1))
        print(f"{name:11s} & " + " & ".join(fit) + " & " + " & ".join(pp) + " & " + " & ".join(sc) + r" \\")
    print("% ---- tab:cost-judges ----")
    for name in [m[0] for m in JUDGE_METHODS] + ["SOAP"]:
        cells = []
        for col, _ in SUBSETS:
            r = judges[(judges.method == name) & (judges.subset == col)]
            if r.empty:
                cells += ["--"] * 4
                continue
            r = r.iloc[0]
            cells += [fmt(r.calls, 1), fmt(r.prompt_tokens), fmt(r.generated_tokens), fmt(r.seconds, 1)]
            if name == "SOAP":
                cells[-1] = f"{fmt(r.one_pass_plus_scoring_s, 2)} (one pass {fmt(r.one_pass_s, 2)} + scoring {fmt(r.scoring_ms, 1)} ms; all passes {fmt(r.seconds, 1)})"
        print(f"{name:14s} & " + " & ".join(cells) + r" \\")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()
    soap = soap_rows()
    oat = read(IN / f"oat_{MODEL}.tsv")
    # StepFinder's setup corpora may be timed in shards (--tag); one row per trajectory.
    parts = [read(p) for p in sorted(IN.glob(f"stepfinder_{MODEL}*.tsv"))]
    sf = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if not sf.empty:
        sf["traj"] = sf.traj.astype(str)
        sf = sf.drop_duplicates(subset=["phase", "corpus", "traj"], keep="first")
    if not oat.empty:
        oat["traj"] = oat.traj.astype(str)
    stage2_soap = read(IN / f"soap_stage2_{MODEL}.tsv")
    stage2_rb = read(IN / f"rb_stage2_{MODEL}.tsv")
    setup = setup_table(soap, oat, sf, stage2_soap, train_seconds())
    onepass = onepass_rows()
    infer = inference_table(soap, oat, sf, stage2_soap, stage2_rb, onepass)
    judges = judges_table(soap, stage2_soap, onepass)
    for name, df in (("c4_cost_setup", setup), ("c4_cost_inference", infer), ("c4_cost_judges", judges)):
        df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
        print(f"  wrote {OUT / name}.tsv ({len(df)} rows)")
        print(df.to_string(index=False))
    if args.print_table:
        print_tables(setup, infer, judges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
