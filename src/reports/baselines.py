"""Shared reporting support: display maps, recorded-score loading, baseline scoring.

This is the support library under the triples protocol (``src.reports.triples``):
dataset display names, the recorded-score loader (``load_scores``), and the scoring
of external baseline predictions (prompting / CHIEF / CORRECT JSONLs) on the same
test splits the proxy pipeline uses. The scoring here was carried over unchanged in
behaviour from the pre-protocol report modules — the baseline numbers must stay
byte-identical to the recorded ones, so resist "tidying" ``_agent_hit`` / ``_acc``.

    from src.reports.baselines import (MODEL_DISPLAY, SUBSET_DISPLAY, ROW_TO_PRED,
                                       SPLIT_MODEL, load_scores, baseline_cell, fmt)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pandas as pd

from ..common import paths
from ..stores import split_data
from ..metrics import standardize_role

SPLIT_MODEL = "qwen3.5-9b"          # canonical id-source for the split (both models share it)

# Display names, per dataset.
MODEL_DISPLAY = [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")]
SUBSET_DISPLAY = {
    "ww": [("Algorithm-Generated", "algorithm-generated"), ("Hand-Crafted", "hand-crafted")],
    "correct-error": [("ARC", "arc"), ("GAIA", "gaia"), ("Hotpot", "hotpot"),
                      ("MATH500", "math500"), ("MMLU-Pro", "mmlu_pro"),
                      ("Musique", "musique"), ("WikiMQA", "wikimqa")],
    "traceelephant": [("magentic", "magentic"), ("captain", "captain")],
}

# baseline row label -> (baseline dir, method stem on disk)
ROW_TO_PRED = {
    "All-at-once":   ("prompting", "all_at_once"),
    "Step-by-step":  ("prompting", "step_by_step"),
    "Binary search": ("prompting", "binary_search"),
    "CHIEF":         ("chief", "chief"),
    "CORRECT":       ("correct", "correct"),
}


def fmt(v) -> str:
    return f"{v:.4f}" if v is not None else ""


# ── recorded score files ─────────────────────────────────────────────────────
def load_scores(cfg, model, subset) -> pd.DataFrame | None:
    """Concat scores/<tag>/<model>/<subset>/seed-*.tsv, restricted to the configured
    seed set (score dirs may hold extra seeds scored later), k==1 rows only."""
    d = paths.scores_root(cfg) / model / subset
    files = sorted(d.glob("seed-*.tsv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    seeds = cfg.get("seeds")
    if seeds:
        df = df[df["seed"].isin(seeds)]
    return df[df["k"] == 1].reset_index(drop=True)


# ── baseline scoring (ported verbatim from the legacy prompting report) ──────
def _norm_agent(x):
    return None if x is None else standardize_role(str(x)).strip().lower()


def _agent_hit(pred, gold) -> bool:
    p, g = _norm_agent(pred), _norm_agent(gold)
    if p is None or g is None or g == "":
        return False
    return p == g or g in p


def _step_hit(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return int(pred) == int(gold)
    except (TypeError, ValueError):
        return False


def load_predictions(pred_file: Path) -> dict[str, dict]:
    preds = {}
    with pred_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                preds[str(row["id"])] = row
    return preds


def val_test_ids(files: list[str], train: float, val: float, seed: int):
    """(val_ids, test_ids) as stems — reproduces the pipeline's split exactly."""
    trval, test = split_data(files, train + val, seed)
    _train, va = split_data(trval, train / (train + val), seed)
    return [Path(f).stem for f in va], [Path(f).stem for f in test]


def _acc(ids: list[str], preds: dict) -> tuple[float, float]:
    """(agent_frac, step_frac) over `ids`; a missing prediction counts as wrong."""
    agent_c = step_c = 0
    for i in ids:
        row = preds.get(str(i))
        if row is None:
            continue
        agent_c += _agent_hit(row.get("predicted_agent"), row.get("gold_agent"))
        step_c += _step_hit(row.get("predicted_step"), row.get("gold_step"))
    n = len(ids)
    return (agent_c / n if n else 0.0), (step_c / n if n else 0.0)


def baseline_cell(cfg, model, subset, root, method, seeds, reps_files):
    """Mean (step, agent) test accuracy of one recorded baseline over ``seeds``.

    Baseline predictions always live under the PLAIN outputs tree passed via cfg —
    their GT-ness is a corpus property, not a knob of the run. Returns None when the
    predictions are missing or cover fewer trajectories than the rep file list."""
    pf = (paths.outputs_base(cfg) / "baselines" / root / model / subset
          / f"predictions_method-{method}.jsonl")
    if not pf.exists():
        return None
    preds = load_predictions(pf)
    if len(preds) < len(reps_files):
        return None
    sp = cfg["splits"]
    steps, agents = [], []
    for seed in seeds:
        _v, test_ids = val_test_ids(reps_files, sp["train"], sp["val"], seed)
        a, s = _acc(test_ids, preds)
        steps.append(s)
        agents.append(a)
    return (mean(steps), mean(agents)) if steps else None
