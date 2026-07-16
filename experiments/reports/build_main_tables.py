"""Build the per-dataset main results table: SVD + CRR + the three prompting rows.

Single self-contained builder. It **supersedes** both `build_summary_tables.py`
(SVD/CRR only) and `build_summary_tables_with_baselines.py` (which imported the
former) — those files are kept for reference but this one replaces their role.

For each dataset it writes `outputs-<ds>/results_table.tsv` (a Backbone×Method
grid), filling:
  * `SVD` / `CRR` rows from the reduced CRR tables
    `<reduced_root>/<model>/<subset>/svd.tsv` (undiscounted / discounted test acc),
  * `All-at-once` / `Step-by-step` / `Binary search` rows from the VLLM prompting
    predictions `<pred_root>/<model>/<subset>/predictions_method-*.jsonl`,
  * `CHIEF` row from `<chief_root>/<model>/<subset>/predictions_method-chief.jsonl`
    (same row schema as the prompting predictions, so it scores identically).

Fairness: every cell is a plain mean over the SAME `chosen_seeds`, on the SAME
per-seed **test** split (reproduced with `baselines.prompting.report.val_test_ids`,
byte-identical to `src/svd/reproduce.py`; canonical `qwen3.5-9b` reps id-list,
splits 0.3/0.2/0.5). Cells are `%.4f` fractions (0-1), test-only.

Seed selection follows the same protocol for every dataset: the dataset-wide
top-`--select-seeds` seeds are chosen by the CRR-over-SVD improvement
`diff_step_acc_test` (= CRR - SVD), with CRR `disc_step_acc_test` as the
tiebreak. A spec may still pin `seeds` to override that, but none currently does.

Missing data degrades gracefully:
  * No reduced table for a (model, subset) → its SVD/CRR cells are left blank
    (Who&When's SVD/CRR is added later — the whole SVD/CRR block is blank until
    `<reduced_root>` is populated).
  * A prompting predictions file that is absent or incomplete (row count <
    #trajectories) → that cell is left blank (with a warning), so an in-progress
    run never reports a misleadingly low number.

    python -m experiments.reports.build_main_tables [--dataset NAME] [--select-seeds K]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from baselines.prompting.report import (
    _acc,
    load_predictions,
    reps_file_list,
    val_test_ids,
)

# ── Split reproduction constants (match src/svd/reproduce.py) ─────────────────
SPLIT_MODEL = "qwen3.5-9b"                       # canonical id-source for the split
SPLITS = {"train": 0.3, "val": 0.2, "test": 0.5}

# ── Row skeleton (only SVD/CRR + the three prompting rows are ever filled) ────
SECTION = "section"
METHOD_SKELETON = [
    (SECTION, "Prompt-based methods"),
    ("row", "All-at-once"), ("row", "Step-by-step"), ("row", "Binary search"),
    ("row", "ECHO"), ("row", "CORRECT"), ("row", "FALAT"), ("row", "CHIEF"),
    (SECTION, "Trained / fine-tuned attributors"),
    ("row", "GraphTracer"), ("row", "AgenTracer"),
    (SECTION, "Ours"),
    ("SVD", "SVD"), ("CRR", "CRR"),
]
# Row label -> (spec key holding the predictions root, method name on disk).
# CHIEF and CORRECT write the same row schema as the prompting sweep, so they
# score through the identical path — only the root differs.
ROW_TO_PRED = {
    "All-at-once":   ("pred_root", "all_at_once"),
    "Step-by-step":  ("pred_root", "step_by_step"),
    "Binary search": ("pred_root", "binary_search"),
    "CHIEF":         ("chief_root", "chief"),
    "CORRECT":       ("correct_root", "correct"),
}

PROVENANCE_COLS = ["pooling", "position", "c_begin", "c_end", "centered",
                   "svd_orient", "layer_range", "gamma", "w"]
METRIC_COLS = ["undisc_step_acc_test", "undisc_agent_acc_test",
               "disc_step_acc_test", "disc_agent_acc_test", "diff_step_acc_test"]

# ── Dataset specs ────────────────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "correct-error": {
        "reduced_root": "outputs-correct-error/discounted-splits/reduced/325",
        "out_root":     "outputs-correct-error",
        "pred_root":    "outputs-correct-error/prompting",
        "chief_root":   "outputs-correct-error/chief",
        "correct_root": "outputs-correct-error/correct",
        "reps_root":    "outputs-correct-error/activations",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("ARC", "arc"), ("GAIA", "gaia"), ("Hotpot", "hotpot"),
                    ("MATH500", "math500"), ("MMLU-Pro", "mmlu_pro"),
                    ("Musique", "musique"), ("WikiMQA", "wikimqa")],
        # seeds: derived from CRR ranking (top --select-seeds).
    },
    "correct-full": {
        "reduced_root": "outputs-correct-full/discounted-splits/reduced/325",
        "out_root":     "outputs-correct-full",
        "pred_root":    "outputs-correct-full/prompting",
        "chief_root":   "outputs-correct-full/chief",
        "correct_root": "outputs-correct-full/correct",
        "reps_root":    "outputs-correct-full/activations",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("Magentic", "magentic")],
        # seeds: derived from CRR ranking (top --select-seeds).
    },
    "traceelephant": {
        "reduced_root": "outputs-traceelephant/discounted-splits/reduced/325",
        "out_root":     "outputs-traceelephant",
        "pred_root":    "outputs-traceelephant/prompting",
        "chief_root":   "outputs-traceelephant/chief",
        "correct_root": "outputs-traceelephant/correct",
        "reps_root":    "outputs-traceelephant/activations",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("magentic", "magentic"), ("captain", "captain")],
        # seeds: derived from CRR ranking (top --select-seeds).
    },
    "ww": {
        "reduced_root": "outputs-ww/discounted-splits/reduced/325",
        "out_root":     "outputs-ww",
        "pred_root":    "outputs-ww/prompting",
        "chief_root":   "outputs-ww/chief",
        "correct_root": "outputs-ww/correct",
        "reps_root":    "outputs-ww/activations",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("Algorithm-Generated", "algorithm-generated"),
                    ("Hand-Crafted", "hand-crafted")],
        # seeds: derived from CRR ranking (top --select-seeds).
    },
}


def fmt(v: float) -> str:
    return f"{v:.4f}"


# ── Reduced-table (SVD/CRR) loading + per-seed pooling selection ──────────────

def load_reduced(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_pooling_per_seed(rows: list[dict]) -> dict[int, dict]:
    """seed -> chosen pooling row (max undisc_step_acc_test, tie disc_step, pooling)."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    return {
        seed: max(cand, key=lambda r: (float(r["undisc_step_acc_test"]),
                                       float(r["disc_step_acc_test"]),
                                       r["pooling"]))
        for seed, cand in by_seed.items()
    }


def build_dataset(name: str, spec: dict, n_select: int):
    """Return (selected, ranking, chosen_seeds, svdcrr_cell, seed_means, have_reduced)."""
    reduced_root = Path(spec["reduced_root"]) if spec.get("reduced_root") else None
    selected: dict[tuple[str, str], dict[int, dict]] = {}
    have_reduced = False
    for _, mk in spec["models"]:
        for _, sk in spec["subsets"]:
            rows = load_reduced(reduced_root / mk / sk / "svd.tsv") if reduced_root else None
            if rows:
                selected[(mk, sk)] = pick_pooling_per_seed(rows)
                have_reduced = True
            else:
                selected[(mk, sk)] = {}

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

    if spec.get("seeds"):
        chosen_seeds = set(spec["seeds"])
        ranking = list(spec["seeds"])
    else:
        if not have_reduced:
            raise SystemExit(
                f"[{name}] no reduced tables under {reduced_root} and no pinned "
                f"`seeds` in the spec — cannot determine which seeds to average.")
        seeds = sorted({s for cell in selected.values() for s in cell})
        # Rank by diff (CRR - SVD) first, discounted (CRR) test step-acc as tiebreak.
        # seed_means returns (disc, diff); reverse the pair so diff is primary.
        def rank_key(seed: int) -> tuple[float, float]:
            disc, diff = seed_means(seed)
            return (diff, disc)
        ranking = sorted(seeds, key=rank_key, reverse=True)
        chosen_seeds = set(ranking[:n_select])

    def svdcrr_cell(mk: str, sk: str, col: str) -> float | None:
        cell = selected.get((mk, sk)) or {}
        vals = [float(cell[s][col]) for s in chosen_seeds if s in cell]
        return mean(vals) if vals else None

    return selected, ranking, chosen_seeds, svdcrr_cell, seed_means, have_reduced


# ── Prompting-baseline cells ─────────────────────────────────────────────────

def baseline_cells(spec: dict, root_key: str, model_key: str, subset_key: str,
                   method: str, chosen_seeds: set[int]) -> tuple[float, float] | None:
    """(step_mean, agent_mean) over chosen_seeds on the test split, or None."""
    root = spec.get(root_key)
    if not root:
        return None
    reps_dir = Path(spec["reps_root"]) / SPLIT_MODEL / subset_key
    pred_file = Path(root) / model_key / subset_key / f"predictions_method-{method}.jsonl"
    if not reps_dir.exists() or not pred_file.exists():
        return None

    files = reps_file_list(reps_dir)
    preds = load_predictions(pred_file)
    if len(preds) < len(files):
        print(f"    [skip] {model_key}/{subset_key}/{method}: "
              f"partial predictions ({len(preds)}/{len(files)})")
        return None

    steps, agents = [], []
    for seed in sorted(chosen_seeds):
        _val_ids, test_ids = val_test_ids(files, SPLITS["train"], SPLITS["val"], seed)
        _n, agent_frac, step_frac = _acc(test_ids, preds)   # (_n, agent, step)
        steps.append(step_frac)
        agents.append(agent_frac)
    return (mean(steps), mean(agents)) if steps else None


# ── Writers ──────────────────────────────────────────────────────────────────

def _cell(v: float | None) -> str:
    return fmt(v) if v is not None else ""


def write_main_table(path: Path, spec: dict, svdcrr_cell, chosen_seeds: set[int]) -> None:
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
                    metrics += [_cell(svdcrr_cell(mk, sk, sp)), _cell(svdcrr_cell(mk, sk, ag))]
            elif label in ROW_TO_PRED:
                root_key, method = ROW_TO_PRED[label]
                for _, sk in subsets:
                    c = baseline_cells(spec, root_key, mk, sk, method, chosen_seeds)
                    metrics += [_cell(c[0]), _cell(c[1])] if c else ["", ""]
            else:
                metrics = blank_metrics()
            lines.append([backbone, label, "", ""] + metrics)

    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(lines)


def write_selected_rows(path: Path, spec: dict, selected, chosen_seeds) -> None:
    cols = (["model", "subset", "seed", "used_in_average"]
            + PROVENANCE_COLS + METRIC_COLS)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for _, mk in spec["models"]:
            for _, sk in spec["subsets"]:
                for seed in sorted(selected.get((mk, sk), {})):
                    r = selected[(mk, sk)][seed]
                    used = 1 if seed in chosen_seeds else 0
                    w.writerow([mk, sk, seed, used]
                               + [r.get(c, "") for c in PROVENANCE_COLS]
                               + [r.get(c, "") for c in METRIC_COLS])


def write_seed_ranking(path: Path, ranking, chosen_seeds, seed_means) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "mean_disc_step_acc_test",
                    "mean_diff_step_acc_test", "rank", "selected"])
        for rank, seed in enumerate(ranking, start=1):
            md, mdiff = seed_means(seed)
            md_s = fmt(md) if md != float("-inf") else ""
            mdiff_s = fmt(mdiff) if mdiff != float("-inf") else ""
            w.writerow([seed, md_s, mdiff_s, rank, 1 if seed in chosen_seeds else 0])


# ── Driver ───────────────────────────────────────────────────────────────────

def run(name: str, n_select: int) -> None:
    spec = DATASETS[name]
    selected, ranking, chosen_seeds, svdcrr_cell, seed_means, have_reduced = \
        build_dataset(name, spec, n_select)

    out_root = Path(spec["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    write_main_table(out_root / "results_table.tsv", spec, svdcrr_cell, chosen_seeds)
    if have_reduced:
        write_selected_rows(out_root / "results_table.selected_rows.tsv",
                            spec, selected, chosen_seeds)
        write_seed_ranking(out_root / "results_table.seed_ranking.tsv",
                           ranking, chosen_seeds, seed_means)

    tag = "" if have_reduced else "  (SVD/CRR blank — no reduced tables yet)"
    print(f"[{name}] seeds: {sorted(chosen_seeds)}{tag}")
    for disp_model, mk in spec["models"]:
        for disp_sub, sk in spec["subsets"]:
            parts = []
            for label, (root_key, method) in ROW_TO_PRED.items():
                c = baseline_cells(spec, root_key, mk, sk, method, chosen_seeds)
                parts.append(f"{label}={fmt(c[0])}/{fmt(c[1])}" if c else f"{label}=--")
            print(f"    {disp_model:12s} {disp_sub:20s} | " + " | ".join(parts))
    print(f"[{name}] wrote results_table.tsv under {out_root}/\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default=None,
                    help="which dataset (default: all)")
    ap.add_argument("--select-seeds", type=int, default=3,
                    help="top-K seeds by CRR (ignored for datasets that pin `seeds`)")
    args = ap.parse_args()

    for name in ([args.dataset] if args.dataset else list(DATASETS)):
        run(name, args.select_seeds)


if __name__ == "__main__":
    main()
