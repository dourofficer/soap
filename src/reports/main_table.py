"""Backbone x Method main results table — SVD/CRR (ours) beside the text baselines.

Reproduces the legacy ``results_table.tsv`` grid. Two variants (``variant`` cfg):

* ``faithful`` (default): the exact legacy grid. SVD = proj base score, CRR = discount,
  taken from a proj-only, pooling=separate reduction (``crr_proj_*.tsv``) and the legacy
  pick-pooling-per-seed protocol; baseline rows (All-at-once / Step-by-step / Binary
  search / CHIEF / CORRECT) scored from the reused prediction JSONLs. Meant to match the
  archived tables cell-for-cell.
* ``extended``: same baselines, but the "Ours" section reports, FOR EACH base scorer
  (proj / angres / resid — maha excluded), its own SVD / CRR(discount) / Backprop triple.
  SVD comes from ``base_by_method_test.tsv``; CRR and Backprop come from per-scorer
  variant reductions ``crr_ext_<scorer>_test.tsv`` / ``backprop_ext_<scorer>_test.tsv``
  (produced by ``scripts/rescore_scorers.sh``, i.e. both rescoring strategies applied to
  each scorer, not only to the joint winner). All joint-pooling.

Baselines are SCORED here, never re-run: the prediction JSONLs (LLM outputs) are reused
model-output artifacts, and scoring is deterministic given a test-id set. The split is
reproduced from the qwen3.5-9b reps id-list at 0.3/0.2/0.5 — byte-identical to the split
every archived number used, so the comparison is on the same test trajectories.

    # from repo root
    python -m src.reports.main_table --config configs/tables/ww.yaml                 # faithful
    python -m src.reports.main_table --config configs/tables/ww.yaml --set variant=extended
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..stores import split_data, list_rep_files
from ..metrics import standardize_role

SPLIT_MODEL = "qwen3.5-9b"          # canonical id-source for the split (both models share it)

# Display names matching the legacy grid, per dataset.
MODEL_DISPLAY = [("Qwen3.5-9B", "qwen3.5-9b"), ("Deepseek-8B", "deepseek-8b")]
SUBSET_DISPLAY = {
    "ww": [("Algorithm-Generated", "algorithm-generated"), ("Hand-Crafted", "hand-crafted")],
    "correct-error": [("ARC", "arc"), ("GAIA", "gaia"), ("Hotpot", "hotpot"),
                      ("MATH500", "math500"), ("MMLU-Pro", "mmlu_pro"),
                      ("Musique", "musique"), ("WikiMQA", "wikimqa")],
    "correct-full": [("Magentic", "magentic")],
    "traceelephant": [("magentic", "magentic"), ("captain", "captain")],
}

# label -> (baseline dir, method stem on disk)
ROW_TO_PRED = {
    "All-at-once":   ("prompting", "all_at_once"),
    "Step-by-step":  ("prompting", "step_by_step"),
    "Binary search": ("prompting", "binary_search"),
    "CHIEF":         ("chief", "chief"),
    "CORRECT":       ("correct", "correct"),
}

SECTION = "section"
SKELETON_FAITHFUL = [
    (SECTION, "Prompt-based methods"),
    ("row", "All-at-once"), ("row", "Step-by-step"), ("row", "Binary search"),
    ("row", "ECHO"), ("row", "CORRECT"), ("row", "FALAT"), ("row", "CHIEF"),
    (SECTION, "Trained / fine-tuned attributors"),
    ("row", "GraphTracer"), ("row", "AgenTracer"),
    (SECTION, "Ours"),
    ("SVD", "SVD"), ("CRR", "CRR"),
]
# extended: keep the baseline block, then for EACH base scorer report its own
# SVD / CRR(discount) / Backprop triple (both rescoring strategies applied to that
# scorer, not just to the joint winner). Fed by per-scorer variant reductions
# (crr_ext_<scorer>_*.tsv / backprop_ext_<scorer>_*.tsv). maha is intentionally excluded.
EXT_SCORERS = ["proj", "angres", "resid"]


def _extended_skeleton():
    rows = SKELETON_FAITHFUL[:-2]                      # baselines + (SECTION, "Ours")
    for m in EXT_SCORERS:
        rows += [(f"BASE:{m}", f"SVD ({m})"),
                 (f"CRRV:{m}", f"CRR ({m})"),
                 (f"BPV:{m}", f"Backprop ({m})")]
    return rows


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
    """(val_ids, test_ids) as stems — reproduces the legacy/CRR split exactly."""
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


# ── reduced-table loading + legacy pooling pick ──────────────────────────────
def _read_tsv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_pooling_per_seed(rows: list[dict]) -> dict[int, dict]:
    """seed -> chosen (pooling) row: max (undisc_step_acc_test, disc_step_acc_test, pooling)."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    return {s: max(c, key=lambda r: (float(r["undisc_step_acc_test"]),
                                     float(r["disc_step_acc_test"]), r["pooling"]))
            for s, c in by_seed.items()}


def _by_seed(rows: list[dict]) -> dict[int, dict]:
    return {int(r["seed"]): r for r in rows}


# ── selection-basis cells (also the faithful SVD/CRR source) ─────────────────
def _selection_cells(cfg, subsets, crr_file):
    """selected[(model,subset)] = {seed: chosen row}. pick_pooling_per_seed collapses a
    per-(pooling,seed) table (faithful/separate) to one row per seed, and is a no-op on
    an already-per-seed table (extended/joint). Used both for seed selection and, for the
    faithful variant, as the SVD/CRR cell source."""
    reduced = paths.reduced_root(cfg)
    selected, have = {}, False
    for _, mk in MODEL_DISPLAY:
        for _, sk in subsets:
            rows = _read_tsv(reduced / mk / sk / crr_file)
            selected[(mk, sk)] = pick_pooling_per_seed(rows) if rows else {}
            have |= bool(rows)
    return selected, have


def _choose_seeds(selected, n_select) -> list[int]:
    """Top-n seeds by mean disc_step_acc_test across cells, tiebreak mean diff (legacy)."""
    seeds = sorted({s for cell in selected.values() for s in cell})

    def means(seed):
        disc = [float(c[seed]["disc_step_acc_test"]) for c in selected.values() if seed in c]
        diff = [float(c[seed].get("diff_step_acc_test", 0.0)) for c in selected.values() if seed in c]
        return (mean(disc) if disc else -1e9, mean(diff) if diff else -1e9)

    ranked = sorted(seeds, key=means, reverse=True)
    return ranked[:n_select]


# ── grid writer ──────────────────────────────────────────────────────────────
def _fmt(v):
    return f"{v:.4f}" if v is not None else ""


def _baseline_cell(cfg, dataset, model, subset, root, method, seeds, reps_files):
    pf = paths.outputs_base(cfg) / "baselines" / root / model / subset / f"predictions_method-{method}.jsonl"
    if not pf.exists():
        return None
    preds = load_predictions(pf)
    if len(preds) < len(reps_files):
        return None
    steps, agents = [], []
    for seed in seeds:
        _v, test_ids = val_test_ids(reps_files, 0.3, 0.2, seed)
        a, s = _acc(test_ids, preds)
        steps.append(s)
        agents.append(a)
    return (mean(steps), mean(agents)) if steps else None


def run(cfg) -> None:
    dataset = cfg.get("dataset") or cfg.get("name")
    variant = cfg.get("variant", "faithful")
    n_select = cfg.get("select_seeds") or 3
    subsets = SUBSET_DISPLAY[dataset]
    skeleton = SKELETON_FAITHFUL if variant == "faithful" else _extended_skeleton()
    reduced = paths.reduced_root(cfg)
    # Seed selection basis: faithful uses the proj/separate CRR; extended uses the
    # per-scorer proj CRR (crr_ext_proj) so it stands alone without the faithful pass.
    sel_file = "crr_proj_test.tsv" if variant == "faithful" else "crr_ext_proj_test.tsv"

    with RunTimer(cfg, "tables") as rec:
        rec.note(variant=variant, select_seeds=n_select)

        selected, have = _selection_cells(cfg, subsets, sel_file)
        chosen = _choose_seeds(selected, n_select) if have else list(range(1, n_select + 1))

        # reps file list per subset (for the split), from the canonical qwen id-source.
        reps_files = {sk: list_rep_files(paths.reps_root(cfg) / SPLIT_MODEL / sk)
                      for _, sk in subsets}

        # header rows
        header_a = ["Backbone", "Method", "TF", "AF"]
        header_b = ["", "", "", ""]
        for disp, _ in subsets:
            header_a += [disp, ""]
            header_b += ["Step-level", "Agent-level"]
        lines = [header_a, header_b, ["N/A", "Random", "", ""] + [""] * (2 * len(subsets))]

        for disp_model, mk in MODEL_DISPLAY:
            first = True
            for kind, label in skeleton:
                backbone = disp_model if first else ""
                first = False
                cells = []
                if kind == SECTION:
                    cells = [""] * (2 * len(subsets))
                elif kind == "SVD":                          # faithful: proj undisc (pooling picked)
                    cells = _faithful_ours(selected, mk, subsets, chosen, "undisc")
                elif kind == "CRR":                          # faithful: proj discount
                    cells = _faithful_ours(selected, mk, subsets, chosen, "disc")
                elif kind.startswith("BASE:"):               # extended: SVD(<scorer>) from by-method
                    cells = _base_by_method(reduced, mk, subsets, chosen, kind.split(":", 1)[1])
                elif kind.startswith("CRRV:"):               # extended: CRR(<scorer>)
                    cells = _variant_disc(reduced, mk, subsets, chosen, f"crr_ext_{kind.split(':',1)[1]}")
                elif kind.startswith("BPV:"):                # extended: Backprop(<scorer>)
                    cells = _variant_disc(reduced, mk, subsets, chosen, f"backprop_ext_{kind.split(':',1)[1]}")
                elif label in ROW_TO_PRED:
                    root, method = ROW_TO_PRED[label]
                    for _, sk in subsets:
                        c = _baseline_cell(cfg, dataset, mk, sk, root, method, chosen, reps_files[sk])
                        cells += [_fmt(c[0]), _fmt(c[1])] if c else ["", ""]
                else:
                    cells = [""] * (2 * len(subsets))
                lines.append([backbone, label, "", ""] + cells)

        out = paths.tables_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        fname = "results_table.tsv" if variant == "faithful" else "results_extended.tsv"
        with open(out / fname, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(lines)
        rec.add_output(out / fname)
        print(f"[main_table] {dataset} variant={variant} seeds={sorted(chosen)} -> {out/fname}")


def _faithful_ours(selected, model, subsets, seeds, which):
    """Faithful SVD (which='undisc') / CRR (which='disc') from the proj/separate table."""
    step_col, agent_col = f"{which}_step_acc_test", f"{which}_agent_acc_test"
    cells = []
    for _, sk in subsets:
        cell = selected.get((model, sk), {})
        cells += [_fmt(_mean_over(cell, seeds, step_col)), _fmt(_mean_over(cell, seeds, agent_col))]
    return cells


def _base_by_method(reduced, model, subsets, seeds, method):
    """Extended SVD(<scorer>): best config of that scorer per seed (joint), from base_by_method."""
    cells = []
    for _, sk in subsets:
        rows = _read_tsv(reduced / model / sk / "base_by_method_test.tsv") or []
        cell = _by_seed([r for r in rows if r["method"] == method])
        cells += [_fmt(_mean_over(cell, seeds, "step_acc_test")),
                  _fmt(_mean_over(cell, seeds, "agent_acc_test"))]
    return cells


def _variant_disc(reduced, model, subsets, seeds, stem):
    """Extended CRR/Backprop(<scorer>): discounted metric from a per-scorer variant table."""
    cells = []
    for _, sk in subsets:
        rows = _read_tsv(reduced / model / sk / f"{stem}_test.tsv") or []
        cell = _by_seed(rows)
        cells += [_fmt(_mean_over(cell, seeds, "disc_step_acc_test")),
                  _fmt(_mean_over(cell, seeds, "disc_agent_acc_test"))]
    return cells


def _mean_over(cell: dict, seeds, col):
    vals = [float(cell[s][col]) for s in seeds if s in cell and cell[s].get(col) not in (None, "")]
    return mean(vals) if vals else None


def main() -> None:
    run(load_and_narrow(base_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
