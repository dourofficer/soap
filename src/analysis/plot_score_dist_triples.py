"""Per-triple score distributions: each seed on its own, plus the three pooled.

For every (model, subset, seed-window) produced by ``score_dist_triples``, draws one
figure per seed plus one for the window's seeds pooled -- four separate PDFs per window,
so any single one can be dropped straight into the manuscript. All four share the
window's x-range, so they stay comparable when flipped through.

Both x variants of ``plot_score_dist`` are produced (``raw`` and ``znorm``); the panel
drawing, palette and separability statistics are imported from it unchanged.

``ranking.tsv`` is the index into the output: one row per (cell, window) with the
pooled AUC of each variant and the per-seed spread, best first. Use it to pick a
window rather than browsing several hundred figures.

    # from v2/
    python -m src.analysis.plot_score_dist_triples --steps artifacts/score-dist/ww/triples/steps.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

from .plot_score_dist import (       # noqa: E402
    MODEL_DISPLAY, SUBSET_DISPLAY, VARIANTS, SURFACE,
    _panel, _with_x, _save, pooled_auc, rank_auc,
)


def _xrange(cell: pd.DataFrame, variant: str) -> tuple[float, float]:
    """Shared x-limits across a window's panels, so seeds are comparable by eye."""
    x = _with_x(cell.copy(), variant).x
    return float(x.min()), float(x.max())


def run(steps_tsv: Path, out_root: Path, show_hist: bool = False) -> None:
    df = pd.read_csv(steps_tsv, sep="\t")
    cfgs_path = steps_tsv.parent / "configs.tsv"
    cfgs = pd.read_csv(cfgs_path, sep="\t") if cfgs_path.exists() else pd.DataFrame()

    rank_rows, n_files = [], 0
    cells = df.groupby(["model", "subset"], sort=False)

    for (model, subset), cell_all in cells:
        s_disp = SUBSET_DISPLAY.get(subset, subset)
        for triple, win in cell_all.groupby("triple", sort=True):
            seeds = sorted(set(win.seed))
            per_seed_auc = {}
            tri = str(triple).replace(",", "-")
            for variant in VARIANTS:
                # One x-range for the whole window, so its four figures stay
                # comparable to each other when flipped through side by side.
                lo, hi = _xrange(win, variant)
                # Each seed on its own, then the window pooled -- separate files.
                panels = [(win[win.seed == s], f"seed-{s}") for s in seeds]
                panels.append((win, "pooled"))
                for g, tag in panels:
                    fig, ax = plt.subplots(figsize=(5.0, 3.2), facecolor=SURFACE)
                    # Title carries the subset only; the filename carries model,
                    # window and seed (the convention set for the other figures).
                    _panel(ax, g, variant, s_disp, show_hist=show_hist)
                    ax.set_xlim(lo, hi)
                    fig.tight_layout()
                    stem = f"{model}_{subset}_seeds-{tri}_{tag}"
                    n_files += len(_save(fig, out_root / variant, stem))
                for seed in seeds:
                    per_seed_auc[(variant, seed)] = pooled_auc(
                        _with_x(win[win.seed == seed].copy(), variant))

            row = {
                "dataset": win.dataset.iloc[0] if "dataset" in win else "",
                "model": model, "subset": subset, "triple": triple,
                "n_error": int(win.is_mistake.sum()),
                "n_ordinary": int((~win.is_mistake).sum()),
                "within_traj_auc": rank_auc(win),
            }
            for variant in VARIANTS:
                vals = [per_seed_auc[(variant, s)] for s in seeds]
                row[f"pooled_auc_{variant}"] = pooled_auc(_with_x(win.copy(), variant))
                row[f"seed_spread_{variant}"] = float(np.max(vals) - np.min(vals))
            if not cfgs.empty:
                c = cfgs[(cfgs.model == model) & (cfgs.subset == subset)
                         & (cfgs.triple == triple)]
                if not c.empty:
                    row["position"] = c.position.iloc[0]
                    row["band"] = f"[{c.c_begin.iloc[0]},{c.c_end.iloc[0]})"
                    row["step_acc"] = c.step_acc_recorded.iloc[0]
                    row["verified"] = bool(c.verified.iloc[0])
            rank_rows.append(row)

    best_key = f"pooled_auc_{VARIANTS[0]}"
    ranking = pd.DataFrame(rank_rows).sort_values(best_key, ascending=False)
    out_root.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_root / "ranking.tsv", sep="\t", index=False)

    print(f"\n  top windows by {best_key} (0.5 = fully overlapping):")
    for _, r in ranking.head(10).iterrows():
        print(f"    {MODEL_DISPLAY.get(r.model, r.model):12s} "
              f"{SUBSET_DISPLAY.get(r.subset, r.subset):20s} seeds {str(r.triple):9s} "
              f"raw={r.pooled_auc_raw:.3f} znorm={r.pooled_auc_znorm:.3f} "
              f"within={r.within_traj_auc:.3f} spread={r[f'seed_spread_{VARIANTS[0]}']:.3f}")
    print(f"  wrote {n_files} figures + ranking.tsv under {out_root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=Path,
                   default=Path("artifacts/score-dist/ww/triples/steps.tsv"))
    p.add_argument("--out", type=Path, default=None,
                   help="output root (default: <steps dir>/plots)")
    p.add_argument("--hist", action="store_true",
                   help="overlay binned histograms behind the density curves")
    a = p.parse_args()
    run(a.steps, a.out or a.steps.parent / "plots", a.hist)


if __name__ == "__main__":
    main()
