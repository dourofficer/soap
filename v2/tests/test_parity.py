"""OPTIONAL migration check: reproduce a legacy implementation's numbers exactly.

This suite is NOT part of the repo's own correctness story — `tests/test_invariants.py`
is, and it runs standalone. This one exists only for adopting results produced by an
earlier implementation: it re-scores the same inputs here and asserts the numbers match
bit-for-bit, so historical tables can be trusted (or the discrepancy located).

It is skipped unless a legacy tree is pointed at explicitly:

    LEGACY_REPO=/path/to/legacy pytest tests/test_parity.py -q

The legacy tree must contain
    <LEGACY_REPO>/outputs-<ds>/weighted-projections/325/<model>/<subset>/svd_pooling-*_seed-*.tsv
    <LEGACY_REPO>/outputs-<ds>/discounted-splits/sweep/325/<model>/<subset>/svd.tsv
and this repo must hold a base-score run under scores/parity/ (configs/parity/<ds>.yaml).

Gate A — base scores: our proj rows (asc, k=1) == legacy weighted-projections, joined on
(position, c_begin, c_end, centered, weighted).
Gate B — rescoring: our reference discount loop reproduces legacy sweep rows exactly.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
_LEGACY = os.environ.get("LEGACY_REPO")
V1 = Path(_LEGACY) if _LEGACY else None
pytestmark = pytest.mark.skipif(
    not _LEGACY, reason="set LEGACY_REPO=<path> to run the optional migration check")

JOIN = ["c_begin", "c_end", "centered", "weighted"]
MCOLS = ["step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test"]
CASES = [("correct-full", m, s) for m in ("qwen3.5-9b", "deepseek-8b") for s in (1, 2, 3)]


@pytest.mark.parametrize("dataset,model,seed", CASES)
def test_gate_a_proj_parity(dataset, model, seed):
    v2_path = V2 / f"outputs/{dataset}/scores/parity/{model}/magentic/seed-{seed}.tsv"
    assert v2_path.exists(), f"missing v2 parity scores: {v2_path} (run the parity score config)"
    v2 = pd.read_csv(v2_path, sep="\t")
    v2 = v2[(v2.method == "proj") & (v2.direction == "asc") & (v2.k == 1)]

    for pool in ("mean", "last"):
        v1_path = (V1 / f"outputs-{dataset}/weighted-projections/325/{model}/magentic/"
                   f"svd_pooling-{pool}_seed-{seed}.tsv")
        v1 = pd.read_csv(v1_path, sep="\t")
        v1 = v1[v1.method == "proj"]
        a = v2[v2.pooling == pool]
        m = a.merge(v1, on=["position"] + JOIN, suffixes=("_v2", "_v1"))
        assert len(m) == len(v1), f"{model} s{seed} {pool}: merged {len(m)} != v1 {len(v1)}"
        for c in MCOLS:
            err = (m[f"{c}_v2"] - m[f"{c}_v1"]).abs().max()
            assert err < 1e-9, f"{model} s{seed} {pool} {c}: max|d|={err}"


# ── Gate B: discount math == v1 sweep, via the faithful loop path ───────────
GATE_B_COLS = ["undisc_step_acc_val", "undisc_agent_acc_val", "undisc_step_acc_test",
               "undisc_agent_acc_test", "disc_step_acc_val", "disc_agent_acc_val",
               "disc_step_acc_test", "disc_agent_acc_test"]


def _gate_b_mismatches(model, dataset="correct-full", n=48):
    """Reproduce n stratified v1 sweep rows via v2's loop path; return list of mismatches.

    Loads reps ONCE per seed and attention ONCE per model (reused across rows).
    """
    import torch
    from src.stores import load_representations, split_files, list_rep_files
    from src.score.svd import fit_one, score_config
    from src.rescore.weights import aggregate_attn
    from src.rescore.strategies import orient, discount_loop
    from src.metrics import compute_metrics
    from src.score.scorers import native_direction

    v1 = pd.read_csv(V1 / f"outputs-{dataset}/discounted-splits/sweep/325/{model}/magentic/svd.tsv", sep="\t")
    sample = (v1[v1.gamma.isin([0.1, 1.0]) & v1.w.astype(str).isin(["1", "all"])]
              .groupby(["seed", "orient", "layer_range"], group_keys=False)
              .apply(lambda g: g.head(1)).head(n))

    base = V2 / f"outputs/{dataset}"
    rep_dir = base / "activations" / model / "magentic"
    data_dir = V2 / f"data/{dataset}/magentic"
    files = list_rep_files(rep_dir)
    weightings, bounds = aggregate_attn(base / "attention", model, "magentic", n_ranges=4, device="cuda")
    labels = [f"{lo}-{hi}" for lo, hi in bounds]

    mismatches, cache = [], {}
    for _, row in sample.iterrows():
        seed, pooling, position = int(row["seed"]), row["pooling"], row["position"]
        if seed not in cache:
            parts = split_files(files, {"train": 0.3, "val": 0.2, "test": 0.5}, seed)
            load = lambda fl: load_representations(rep_dir, data_dir, ["mean", "last"], files=fl, device="cuda")
            cache[seed] = {"train": load(parts["train"]), "val": load(parts["val"]), "test": load(parts["test"]), "fit": {}}
        c = cache[seed]
        key = (pooling, position)
        if key not in c["fit"]:
            c["fit"][key] = fit_one(c["train"].stores[key].R)
        entry = c["fit"][key]
        scfg = (row["method"], int(row["c_begin"]), int(row["c_end"]), bool(row["centered"]), bool(row["weighted"]))
        weighting = weightings[labels.index(str(row["layer_range"]))]
        nd = native_direction(row["method"])       # v1 undisc = base score's native metric
        got = {}
        for split in ("val", "test"):
            s = score_config(c[split].stores[key].R, entry, *scfg)
            kp = c[split].keeper
            um = compute_metrics(s, kp, [1], nd)
            o = orient(s, row["orient"])
            dm = compute_metrics(discount_loop(o, kp, weighting, float(row["gamma"]), row["w"]), kp, [1], "desc")
            got[f"undisc_step_acc_{split}"] = um[f"step@1_{nd}"]
            got[f"undisc_agent_acc_{split}"] = um[f"agent@1_{nd}"]
            got[f"disc_step_acc_{split}"] = dm["step@1_desc"]
            got[f"disc_agent_acc_{split}"] = dm["agent@1_desc"]
        for col in GATE_B_COLS:
            if abs(got[col] - row[col]) >= 1e-9:
                mismatches.append((seed, row["orient"], row["layer_range"], row["gamma"], row["w"], col, got[col], row[col]))
    return mismatches, len(sample)


@pytest.mark.parametrize("model", ["qwen3.5-9b", "deepseek-8b"])
def test_gate_b_discount_parity(model):
    mm, n = _gate_b_mismatches(model)
    assert not mm, f"{model}: {len(mm)}/{n*8} metric mismatches; first: {mm[0]}"
