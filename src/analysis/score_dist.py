"""Per-step base-score distributions — decisive-error steps vs ordinary steps.

Recomputes the RAW base score (native convention, no orientation / normalization /
rescoring) for the manuscript's anchor configuration of every (model, subset) cell and
writes one row per scored step, tagged with the gold ``is_mistake`` flag. This is the
data behind the method-section claim that decisive-error steps project SMALLER onto the
selected spectral band (``proj`` is "lower = error", i.e. direction ``asc``).

The anchor configuration is READ from the exp-august Protocol-2 ("triples") selection
table, so this stays in sync with the reported numbers instead of hardcoding them:
pooling / method / centered / weighted / position / [c_begin, c_end) / the cell's seeds.
Per seed the SVD is fit on that seed's TRAIN split and the TEST split is scored, exactly
as the score stage does. Each cell's reproduced mean test step@1 is checked against the
``svd_step`` recorded in the selection table.

    # from v2/
    python -m src.analysis.score_dist --config configs/datasets/ww.yaml
    python -m src.analysis.score_dist --config configs/datasets/ww.yaml \
        --model qwen3.5-9b --subset hand-crafted --device cpu
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.config import V2_ROOT
from ..common.provenance import RunTimer
from ..metrics import compute_metrics
from ..score.svd import fit_one, score_config, N_COMPONENTS
from ..stores import load_representations, split_files, list_rep_files

SELECTION_TSV = "exp-august/outputs/manuscript-tables/table1_main_selection.tsv"
ARTIFACTS_ROOT = "artifacts/score-dist"
ANCHOR_COLS = ["pooling", "method", "centered", "weighted",
               "position", "c_begin", "c_end", "direction"]


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def load_anchors(tsv: Path, dataset: str) -> pd.DataFrame:
    """Anchor config per (model, subset) for one dataset.

    Primary source is the manuscript selection table (Protocol 2 / "triples"). Datasets
    it does not cover -- correct-full, which the merge script excludes -- fall back to
    that dataset's own Protocol-1 table, whose ``SVD (proj)`` rows carry the same base
    axes. Rows without a position (correct-error's ``average`` macro-row) are dropped.
    """
    frames = []
    if tsv.exists():
        d = pd.read_csv(tsv, sep="\t")
        d = d[d["dataset"] == dataset]
        if not d.empty:
            frames.append(d.assign(anchor_source="manuscript-table"))
    focused = (V2_ROOT / "exp-august" / "outputs" / dataset
               / "tables" / "325" / "focused_selection.tsv")
    if focused.exists():
        d = pd.read_csv(focused, sep="\t")
        d = d[d["row"].astype(str).str.startswith("SVD")]
        if not d.empty:
            frames.append(d.assign(anchor_source="focused-table"))
    assert frames, f"no anchor rows for dataset {dataset!r}"
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].notna()]
    df = df.drop_duplicates(subset=["model", "subset"], keep="first")
    assert not df.empty, f"no usable anchor rows for dataset {dataset!r}"
    return df.set_index(["model", "subset"])


def _ranks(scores: torch.Tensor, direction: str) -> np.ndarray:
    """1-based within-slice rank; ties resolve to the EARLIEST step (metric convention).

    Closed-form identity with the metrics' stable sort (see src/metrics.py):
    rank(i) = 1 + #{j : s_j > s_i} + #{j < i : s_j == s_i}, on the higher-is-better view.
    """
    s = scores if direction == "desc" else -scores
    gt = (s.unsqueeze(0) > s.unsqueeze(1)).sum(1)
    eq = (s.unsqueeze(0) == s.unsqueeze(1)) & torch.tril(
        torch.ones(len(s), len(s), dtype=torch.bool), diagonal=-1)
    return (1 + gt + eq.sum(1)).cpu().numpy()


def score_cell(cfg, model: str, subset: str, anchor: pd.Series,
               device: str) -> tuple[list[dict], list[float]]:
    """Score every seed of one cell; return (per-step rows, per-seed test step@1)."""
    dataset = paths._ds(cfg)
    pooling, position = anchor["pooling"], anchor["position"]
    method, direction = anchor["method"], anchor["direction"]
    c_begin, c_end = int(anchor["c_begin"]), int(anchor["c_end"])
    centered, weighted = _as_bool(anchor["centered"]), _as_bool(anchor["weighted"])
    seeds = [int(s) for s in str(anchor["seeds"]).split(",")]

    rep_dir = paths.reps_root(cfg) / model / subset
    data_dir = paths.data_root(cfg) / subset
    files = list_rep_files(rep_dir)

    rows, accs = [], []
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        load = lambda fs: load_representations(          # noqa: E731
            rep_dir, data_dir, [pooling], weight_names=[position], files=fs, device=device)
        train, test = load(parts["train"]), load(parts["test"])

        entry = fit_one(train.stores[(pooling, position)].R, N_COMPONENTS)
        scores = score_config(test.stores[(pooling, position)].R, entry,
                              method, c_begin, c_end, centered, weighted)

        keeper = test.keeper
        accs.append(compute_metrics(scores, keeper, ks=[1], direction=direction)
                    [f"step@1_{direction}"])

        for start, end in keeper.traj_ranges:
            seg = scores[start:end]
            rank = _ranks(seg, direction)
            for i, e in enumerate(keeper.index[start:end]):
                rows.append({
                    "dataset": dataset,
                    "model": model, "subset": subset, "seed": seed, "split": "test",
                    "traj_idx": e.traj_idx, "step_idx": e.step_idx,
                    "n_steps": end - start, "role": e.role,
                    "is_mistake": bool(e.is_mistake),
                    "score": float(seg[i]),
                    "rank": int(rank[i]), "is_pred": bool(rank[i] == 1),
                    "pooling": pooling, "position": position, "method": method,
                    "c_begin": c_begin, "c_end": c_end,
                    "centered": centered, "weighted": weighted, "direction": direction,
                })
    return rows, accs


def run(cfg) -> None:
    device = cfg.get("device", "cpu")
    dataset = paths._ds(cfg)
    tsv = Path(cfg.get("selection_tsv", V2_ROOT / SELECTION_TSV))
    out_dir = Path(cfg.get("artifacts_root", V2_ROOT / ARTIFACTS_ROOT)) / dataset

    anchors = load_anchors(tsv, dataset)

    with RunTimer(cfg, "analysis") as rec:
        rec.note(selection_tsv=str(tsv), eval_split="test", device=device)
        rows: list[dict] = []
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                if (model, subset) not in anchors.index:
                    print(f"[score-dist] {model}/{subset}: no anchor row — skipped")
                    continue
                anchor = anchors.loc[(model, subset)]
                cell_rows, accs = score_cell(cfg, model, subset, anchor, device)
                rows.extend(cell_rows)

                df = pd.DataFrame(cell_rows)
                mis, non = df[df.is_mistake].score, df[~df.is_mistake].score
                # Recorded base accuracy: "svd_step" in the manuscript table,
                # "step_acc_test" in the Protocol-1 fallback table.
                key = "svd_step" if "svd_step" in anchor.index else "step_acc_test"
                got, want = float(np.mean(accs)), float(anchor[key])
                flag = "ok" if abs(got - want) < 5e-4 else "*** MISMATCH ***"
                print(f"[score-dist] {model}/{subset} {anchor['position']} "
                      f"band[{int(anchor['c_begin'])},{int(anchor['c_end'])}) "
                      f"seeds={anchor['seeds']}: step@1 {got:.4f} vs recorded "
                      f"{want:.4f} {flag}")
                print(f"    error steps n={len(mis)} mean={mis.mean():.4g} "
                      f"median={mis.median():.4g} | normal steps n={len(non)} "
                      f"mean={non.mean():.4g} median={non.median():.4g}")

        if cfg.get("dry_run"):
            print(f"[score-dist] dry run — {len(rows)} rows not written")
            return
        out = out_dir / "scores.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        rec.add_output(out)
        print(f"  wrote {out}  ({len(rows)} steps)")


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
