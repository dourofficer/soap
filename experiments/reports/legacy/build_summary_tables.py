"""Build the per-dataset results table (SVD & CRR rows only) plus audit logs.

Reads the reduced CRR tables at
    outputs-<ds>/discounted-splits/reduced/325/<model>/<subset>/svd.tsv
(each file holds one optimal row per (pooling, seed)), and produces, per dataset:

  outputs-<ds>/results_table.tsv               — the Backbone x Method grid
  outputs-<ds>/results_table.selected_rows.tsv — provenance of every chosen row
  outputs-<ds>/results_table.seed_ranking.tsv  — the dataset-wide seed ranking

Selection procedure (per dataset):
  1. Per (model, subset, seed): pick the pooling row with max undisc_step_acc_test
     (tie-break: max disc_step_acc_test, then pooling name).
  2. Dataset-wide: rank seeds by mean over all (model, subset) of
     (disc_step_acc_test, diff_step_acc_test) desc; take the top `n_select`.
     The same seeds are used for both backbones.
  3. Each cell = mean over the selected seeds of the chosen row's metric:
     SVD row -> undisc_{step,agent}_acc_test ; CRR row -> disc_{step,agent}_acc_test.

    python -m experiments.reports.build_summary_tables [--dataset NAME] [--select-seeds K]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

# ── Dataset specs ────────────────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "correct-error": {
        "reduced_root": "outputs-correct-error/discounted-splits/reduced/325",
        "out_root":     "outputs-correct-error",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("ARC", "arc"), ("GAIA", "gaia"), ("Hotpot", "hotpot"),
                    ("MATH500", "math500"), ("MMLU-Pro", "mmlu_pro"),
                    ("Musique", "musique"), ("WikiMQA", "wikimqa")],
    },
    "traceelephant": {
        "reduced_root": "outputs-traceelephant/discounted-splits/reduced/325",
        "out_root":     "outputs-traceelephant",
        "models":  [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")],
        "subsets": [("magentic", "magentic"), ("captain", "captain")],
    },
}

# Method-row skeleton (only SVD/CRR are populated; the rest are template blanks).
SECTION = "section"   # marker: a section-header row (no metric cells)
METHOD_SKELETON = [
    (SECTION, "Prompt-based methods"),
    ("row", "All-at-once"), ("row", "Step-by-step"), ("row", "Binary search"),
    ("row", "ECHO"), ("row", "CORRECT"), ("row", "FALAT"), ("row", "CHIEF"),
    (SECTION, "Trained / fine-tuned attributors"),
    ("row", "GraphTracer"), ("row", "AgenTracer"),
    (SECTION, "Ours"),
    ("SVD", "SVD"), ("CRR", "CRR"),
]

PROVENANCE_COLS = ["pooling", "position", "c_begin", "c_end", "centered",
                   "svd_orient", "layer_range", "gamma", "w"]
METRIC_COLS = ["undisc_step_acc_test", "undisc_agent_acc_test",
               "disc_step_acc_test", "disc_agent_acc_test", "diff_step_acc_test"]


def fmt(v: float) -> str:
    return f"{v:.4f}"


def load_reduced(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing reduced table: {path}")
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_pooling_per_seed(rows: list[dict]) -> dict[int, dict]:
    """For one (model, subset) file: seed -> chosen pooling row (max undisc_step)."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    chosen: dict[int, dict] = {}
    for seed, cand in by_seed.items():
        chosen[seed] = max(
            cand,
            key=lambda r: (float(r["undisc_step_acc_test"]),
                           float(r["disc_step_acc_test"]),
                           r["pooling"]),
        )
    return chosen


def build_dataset(name: str, spec: dict, n_select: int):
    reduced_root = Path(spec["reduced_root"])

    # selected[(model_key, subset_key)][seed] = chosen row
    selected: dict[tuple[str, str], dict[int, dict]] = {}
    for _, mk in spec["models"]:
        for _, sk in spec["subsets"]:
            rows = load_reduced(reduced_root / mk / sk / "svd.tsv")
            selected[(mk, sk)] = pick_pooling_per_seed(rows)

    # All seeds (consistent across cells); use the union, sorted.
    seeds = sorted({s for cell in selected.values() for s in cell})

    # Dataset-wide seed ranking: mean over all (model, subset) pairs.
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

    ranking = sorted(seeds, key=lambda s: seed_means(s), reverse=True)
    chosen_seeds = set(ranking[:n_select])

    # Per-cell averages over the chosen seeds.
    def cell_avg(mk: str, sk: str, col: str) -> float:
        vals = [float(selected[(mk, sk)][s][col]) for s in chosen_seeds
                if s in selected[(mk, sk)]]
        return mean(vals)

    return selected, seeds, ranking, chosen_seeds, cell_avg


def write_main_table(path: Path, spec: dict, cell_avg) -> None:
    subsets = spec["subsets"]
    n_meta = 4  # Backbone, Method, TF, AF

    header_a = ["Backbone", "Method", "TF", "AF"]
    header_b = ["", "", "", ""]
    for disp, _ in subsets:
        header_a += [disp, ""]
        header_b += ["Step-level", "Agent-level"]

    def blank_metrics():
        return [""] * (2 * len(subsets))

    lines: list[list[str]] = [header_a, header_b]
    # Random row (backbone N/A), all metric cells blank.
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
                for seed in sorted(selected[(mk, sk)]):
                    r = selected[(mk, sk)][seed]
                    used = 1 if seed in chosen_seeds else 0
                    w.writerow([mk, sk, seed, used]
                               + [r.get(c, "") for c in PROVENANCE_COLS]
                               + [r.get(c, "") for c in METRIC_COLS])


def write_seed_ranking(path: Path, ranking, chosen_seeds, cell_seed_means) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "mean_disc_step_acc_test",
                    "mean_diff_step_acc_test", "rank", "selected"])
        for rank, seed in enumerate(ranking, start=1):
            md, mdiff = cell_seed_means(seed)
            w.writerow([seed, fmt(md), fmt(mdiff), rank,
                        1 if seed in chosen_seeds else 0])


def run(name: str, n_select: int) -> None:
    spec = DATASETS[name]
    selected, seeds, ranking, chosen_seeds, cell_avg = build_dataset(name, spec, n_select)

    # Reconstruct the seed-means fn for the ranking file (kept local to build_dataset).
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
    write_main_table(out_root / "results_table.tsv", spec, cell_avg)
    write_selected_rows(out_root / "results_table.selected_rows.tsv",
                        spec, selected, chosen_seeds)
    write_seed_ranking(out_root / "results_table.seed_ranking.tsv",
                       ranking, chosen_seeds, seed_means)

    print(f"[{name}] seeds available: {seeds}")
    print(f"[{name}] selected seeds (top {n_select}, dataset-wide): "
          f"{sorted(chosen_seeds)}")
    for disp_model, mk in spec["models"]:
        for disp_sub, sk in spec["subsets"]:
            print(f"    {disp_model:12s} {disp_sub:9s} | "
                  f"SVD step/agent = {fmt(cell_avg(mk, sk, 'undisc_step_acc_test'))}/"
                  f"{fmt(cell_avg(mk, sk, 'undisc_agent_acc_test'))} | "
                  f"CRR step/agent = {fmt(cell_avg(mk, sk, 'disc_step_acc_test'))}/"
                  f"{fmt(cell_avg(mk, sk, 'disc_agent_acc_test'))}")
    print(f"[{name}] wrote 3 files under {out_root}/\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default=None,
                    help="which dataset (default: all)")
    ap.add_argument("--select-seeds", type=int, default=3)
    args = ap.parse_args()

    names = [args.dataset] if args.dataset else list(DATASETS)
    for name in names:
        run(name, args.select_seeds)


if __name__ == "__main__":
    main()
