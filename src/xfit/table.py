"""Build ``results_synthfit.tsv`` — the synthetic cross-fit comparison table.

A SEPARATE file (never touches ``results_extended.tsv``) in the SAME schema, so cells sit
side by side with the main table. It keeps the baseline block verbatim, then for EACH base
scorer (proj / angres / resid) x EACH rescoring (SVD base / CRR / Backprop) reports THREE
rows: the in-distribution fit and the two synthetic generators (``synth-q9`` /
``synth-q35``). Each subset column is filled by that subset's matched-harness source
(algorithm-generated/captain <- captain fit; hand-crafted/magentic/correct-* <- magentic
fit). Cells are test-selected step/agent, meaned over the SAME chosen seeds the extended
main table used (so the in-dist rows reproduce ``results_extended`` exactly).

    # from repo root (xfit.score + xfit.rescore must have run)
    python -m src.xfit.table                     # all target datasets
    python -m src.xfit.table --dataset ww
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from ..common import paths
from ..common.config import load_manifest
from ..stores import list_rep_files
from ..reports.main_table import (
    MODEL_DISPLAY, SUBSET_DISPLAY, ROW_TO_PRED, SPLIT_MODEL,
    _read_tsv, _by_seed, _mean_over, _fmt, _baseline_cell,
    _selection_cells, _choose_seeds,
)
from .common import load_config, source_tag, iter_sources, targets_for

STRATS = [("SVD", "base"), ("CRR", "crr"), ("Backprop", "backprop")]
SCORERS = ["proj", "angres", "resid"]
GENERATORS = [("indist", "in-dist"), ("q9", "synth-q9"), ("q35", "synth-q35")]


# ── reduced-root resolution ──────────────────────────────────────────────────
def _reduced_root(dataset: str, split_tag: str | None):
    m = dict(load_manifest(dataset))
    m["dataset"] = dataset
    if split_tag:
        m["split_tag"] = split_tag
    return paths.reduced_root(m)


def _harness_of(cfg) -> dict[tuple[str, str], str]:
    """(dataset, subset) -> generating harness, from the matched-target map."""
    out = {}
    for harness in cfg["targets"]:
        for tgt in targets_for(cfg, harness):
            out[(tgt["dataset"], tgt["subset"])] = harness
    return out


def _source_for(cfg, dataset, subset, gen) -> str | None:
    """Synthetic source name for a subset column and generator ('q9'/'q35')."""
    harness = _harness_of(cfg).get((dataset, subset))
    if harness is None:
        return None
    return cfg["sources"][harness].get(gen)


# ── cell readers (one subset, one reduced root) ──────────────────────────────
def _svd_cell(root: Path, model, subset, scorer, seeds):
    rows = _read_tsv(root / model / subset / "base_by_method_test.tsv") or []
    cell = _by_seed([r for r in rows if r["method"] == scorer])
    return _mean_over(cell, seeds, "step_acc_test"), _mean_over(cell, seeds, "agent_acc_test")


def _disc_cell(root: Path, model, subset, stem, seeds):
    rows = _read_tsv(root / model / subset / f"{stem}_test.tsv") or []
    cell = _by_seed(rows)
    return (_mean_over(cell, seeds, "disc_step_acc_test"),
            _mean_over(cell, seeds, "disc_agent_acc_test"))


def _ours_cell(cfg, dataset, model, subset, disk_strat, scorer, gen, seeds):
    """(step, agent) for one Ours cell: strategy x scorer x generator, matched-harness root."""
    if gen == "indist":
        root = _reduced_root(dataset, None)
    else:
        source = _source_for(cfg, dataset, subset, gen)
        root = _reduced_root(dataset, source_tag(source)) if source else None
    if root is None:
        return None, None
    if disk_strat == "base":
        return _svd_cell(root, model, subset, scorer, seeds)
    return _disc_cell(root, model, subset, f"{disk_strat}_ext_{scorer}", seeds)


# ── skeleton ─────────────────────────────────────────────────────────────────
def _skeleton():
    rows = [("section", "Prompt-based methods")]
    rows += [("baseline", lbl) for lbl in
             ["All-at-once", "Step-by-step", "Binary search", "ECHO", "CORRECT", "FALAT", "CHIEF"]]
    rows += [("section", "Trained / fine-tuned attributors"),
             ("blank", "GraphTracer"), ("blank", "AgenTracer"),
             ("section", "Ours (in-dist vs synthetic fit)")]
    for scorer in SCORERS:
        for disp_strat, disk_strat in STRATS:
            for gen, gen_disp in GENERATORS:
                label = f"{disp_strat} ({scorer})" if gen == "indist" \
                    else f"{disp_strat} ({scorer} · {gen_disp})"
                rows.append((f"ours:{disk_strat}:{scorer}:{gen}", label))
    return rows


CAVEAT = [
    ["# SYNTHETIC CROSS-FIT (ceiling test) — results_synthfit.tsv"],
    ["# SVD fit on synthetic gaia+assistantbench trajectories (datagen/); scored on the "
     "IDENTICAL 325 test split (5/5 val/test target split; test half == 325 test)."],
    ["# LEAKAGE: the gaia+assistantbench fit pool shares exact TEST questions with the "
     "targets, so synth-q9/synth-q35 columns are OPTIMISTICALLY biased (ceiling, accepted)."],
    ["# The qwen9b-vs-qwen35b (synth-q9 vs synth-q35) comparison is meaningful (shared leakage)."],
]


def build_dataset(cfg, dataset: str) -> Path:
    subsets = SUBSET_DISPLAY[dataset]
    n_select = cfg.get("select_seeds") or 3

    cfg325 = dict(load_manifest(dataset))
    cfg325["dataset"] = dataset
    # Same seed selection the extended main table uses (in-dist crr_ext_proj basis).
    selected, have = _selection_cells(cfg325, subsets, "crr_ext_proj_test.tsv")
    chosen = _choose_seeds(selected, n_select) if have else list(range(1, n_select + 1))
    reps_files = {sk: list_rep_files(paths.reps_root(cfg325) / SPLIT_MODEL / sk)
                  for _, sk in subsets}

    header_a = ["Backbone", "Method", "TF", "AF"]
    header_b = ["", "", "", ""]
    for disp, _ in subsets:
        header_a += [disp, ""]
        header_b += ["Step-level", "Agent-level"]
    lines = list(CAVEAT)
    lines += [header_a, header_b, ["N/A", "Random", "", ""] + [""] * (2 * len(subsets))]

    for disp_model, mk in MODEL_DISPLAY:
        first = True
        for kind, label in _skeleton():
            backbone = disp_model if first else ""
            first = False
            cells = []
            if kind == "section":
                cells = [""] * (2 * len(subsets))
            elif kind == "baseline" and label in ROW_TO_PRED:
                root, method = ROW_TO_PRED[label]
                for _, sk in subsets:
                    c = _baseline_cell(cfg325, dataset, mk, sk, root, method, chosen, reps_files[sk])
                    cells += [_fmt(c[0]), _fmt(c[1])] if c else ["", ""]
            elif kind.startswith("ours:"):
                _, disk_strat, scorer, gen = kind.split(":")
                for _, sk in subsets:
                    step, agent = _ours_cell(cfg, dataset, mk, sk, disk_strat, scorer, gen, chosen)
                    cells += [_fmt(step), _fmt(agent)]
            else:
                cells = [""] * (2 * len(subsets))
            lines.append([backbone, label, "", ""] + cells)

    out = paths.tables_root(cfg325) / "results_synthfit.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(lines)
    print(f"[synthfit] {dataset} seeds={sorted(chosen)} -> {out}")
    return out


def run(cfg, only_dataset=None) -> None:
    datasets = []
    for harness in cfg["targets"]:
        for tgt in targets_for(cfg, harness):
            if tgt["dataset"] not in datasets:
                datasets.append(tgt["dataset"])
    for ds in datasets:
        if only_dataset in (None, ds):
            build_dataset(cfg, ds)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    run(load_config(args.overrides), only_dataset=args.dataset)


if __name__ == "__main__":
    main()
