"""Build the per-dataset results table WITH the prompting-baseline rows filled.

Superset of ``experiments.reports.build_summary_tables``: it reuses that module's
data loading / seed selection / SVD+CRR cell logic verbatim (imported, not copied),
and additionally fills the three prompt-based rows — ``All-at-once``,
``Step-by-step``, ``Binary search`` — from the VLLM prompting predictions
(``outputs-<ds>/prompting/...``).

Fairness is automatic: the baseline cells are averaged over the SAME
``chosen_seeds`` and evaluated on the SAME per-seed **test** split that the SVD/CRR
rows use. The split is reproduced with ``baselines.prompting.report.val_test_ids``
(byte-identical to ``src/svd/reproduce.py``), over the canonical ``qwen3.5-9b``
reps id-list, splits 0.3/0.2/0.5. Cells are ``%.4f`` fractions (0-1), test-only —
matching the SVD/CRR cells exactly.

A prompting cell is filled only when its predictions file is **complete**
(row count == #trajectories); otherwise it is left blank (with a warning), so an
in-progress run never reports a misleadingly low number.

    python -m experiments.reports.build_summary_tables_with_baselines [--dataset NAME] [--select-seeds K]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from experiments.reports.build_summary_tables import (
    DATASETS,
    METHOD_SKELETON,
    build_dataset,
    fmt,
    write_selected_rows,
    write_seed_ranking,
)
from baselines.prompting.report import (
    _acc,
    load_predictions,
    reps_file_list,
    val_test_ids,
)

# Row label (in METHOD_SKELETON) -> prompting method key (predictions filename).
ROW_TO_METHOD = {
    "All-at-once": "all_at_once",
    "Step-by-step": "step_by_step",
    "Binary search": "binary_search",
}

SPLIT_MODEL = "qwen3.5-9b"          # canonical id-source for the split (see report.py)
SPLITS = {"train": 0.3, "val": 0.2, "test": 0.5}

# Where the prompting predictions / reps live, per dataset.
PROMPTING_ROOTS = {
    "correct-error": {"pred_root": "outputs-correct-error/prompting",
                      "reps_root": "outputs-correct-error/activations"},
    "traceelephant": {"pred_root": "outputs-traceelephant/prompting",
                      "reps_root": "outputs-traceelephant/activations"},
}


def baseline_cells(dataset: str, model_key: str, subset_key: str, method: str,
                   chosen_seeds: set[int]) -> tuple[float, float] | None:
    """(step_mean, agent_mean) over chosen_seeds on the test split, or None if unavailable.

    Returns None (→ blank cell) when the reps or predictions are missing, or when
    the predictions file is incomplete (fewer rows than trajectories).
    """
    roots = PROMPTING_ROOTS[dataset]
    reps_dir = Path(roots["reps_root"]) / SPLIT_MODEL / subset_key
    pred_file = Path(roots["pred_root"]) / model_key / subset_key / f"predictions_method-{method}.jsonl"
    if not reps_dir.exists() or not pred_file.exists():
        return None

    files = reps_file_list(reps_dir)
    preds = load_predictions(pred_file)
    if len(preds) < len(files):
        # In-progress / partial run — don't report a misleadingly low number.
        print(f"    [skip] {model_key}/{subset_key}/{method}: "
              f"partial predictions ({len(preds)}/{len(files)})")
        return None

    steps, agents = [], []
    for seed in sorted(chosen_seeds):
        _val_ids, test_ids = val_test_ids(files, SPLITS["train"], SPLITS["val"], seed)
        _n, agent_frac, step_frac = _acc(test_ids, preds)   # note: (_n, agent, step)
        steps.append(step_frac)
        agents.append(agent_frac)
    if not steps:
        return None
    return mean(steps), mean(agents)


def write_main_table_with_baselines(path: Path, dataset: str, spec: dict,
                                    cell_avg, chosen_seeds: set[int]) -> None:
    """Same layout as build_summary_tables.write_main_table, but fill the 3 baseline rows."""
    subsets = spec["subsets"]

    header_a = ["Backbone", "Method", "TF", "AF"]
    header_b = ["", "", "", ""]
    for disp, _ in subsets:
        header_a += [disp, ""]
        header_b += ["Step-level", "Agent-level"]

    def blank_metrics():
        return [""] * (2 * len(subsets))

    lines: list[list[str]] = [header_a, header_b]
    lines.append(["N/A", "Random", "", ""] + blank_metrics())

    for disp_model, mk in spec["models"]:
        first_in_block = True
        for kind, label in METHOD_SKELETON:
            backbone = disp_model if first_in_block else ""
            first_in_block = False
            metrics: list[str] = []
            if kind in ("SVD", "CRR"):
                sp = "undisc_step_acc_test" if kind == "SVD" else "disc_step_acc_test"
                ag = "undisc_agent_acc_test" if kind == "SVD" else "disc_agent_acc_test"
                for _, sk in subsets:
                    metrics += [fmt(cell_avg(mk, sk, sp)), fmt(cell_avg(mk, sk, ag))]
            elif label in ROW_TO_METHOD:
                method = ROW_TO_METHOD[label]
                for _, sk in subsets:
                    cells = baseline_cells(dataset, mk, sk, method, chosen_seeds)
                    if cells is None:
                        metrics += ["", ""]
                    else:
                        metrics += [fmt(cells[0]), fmt(cells[1])]
            else:
                metrics = blank_metrics()
            lines.append([backbone, label, "", ""] + metrics)

    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(lines)


def run(name: str, n_select: int) -> None:
    spec = DATASETS[name]
    selected, seeds, ranking, chosen_seeds, cell_avg = build_dataset(name, spec, n_select)

    # seed-means, replicated from build_summary_tables.run (kept local there).
    def seed_means(seed: int) -> tuple[float, float]:
        disc, diff = [], []
        for cell in selected.values():
            r = cell.get(seed)
            if r is None:
                continue
            disc.append(float(r["disc_step_acc_test"]))
            diff.append(float(r["diff_step_acc_test"]))
        return (mean(disc) if disc else float("-inf"),
                mean(diff) if diff else float("-inf"))

    out_root = Path(spec["out_root"])
    write_main_table_with_baselines(out_root / "results_table.tsv", name, spec,
                                    cell_avg, chosen_seeds)
    write_selected_rows(out_root / "results_table.selected_rows.tsv",
                        spec, selected, chosen_seeds)
    write_seed_ranking(out_root / "results_table.seed_ranking.tsv",
                       ranking, chosen_seeds, seed_means)

    print(f"[{name}] selected seeds (top {n_select}): {sorted(chosen_seeds)}")
    for disp_model, mk in spec["models"]:
        for disp_sub, sk in spec["subsets"]:
            parts = []
            for label, method in ROW_TO_METHOD.items():
                c = baseline_cells(name, mk, sk, method, chosen_seeds)
                parts.append(f"{label}={fmt(c[0])}/{fmt(c[1])}" if c else f"{label}=--")
            print(f"    {disp_model:12s} {disp_sub:9s} | " + " | ".join(parts))
    print(f"[{name}] wrote results_table.tsv (+ audit files) under {out_root}/\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(PROMPTING_ROOTS), default=None,
                    help="which dataset (default: all)")
    ap.add_argument("--select-seeds", type=int, default=3)
    args = ap.parse_args()

    names = [args.dataset] if args.dataset else list(PROMPTING_ROOTS)
    for name in names:
        run(name, args.select_seeds)


if __name__ == "__main__":
    main()
