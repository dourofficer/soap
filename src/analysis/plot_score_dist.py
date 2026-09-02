"""Plot base-score distributions: decisive-error steps vs ordinary steps.

Reads the long per-step table written by ``src.analysis.score_dist_triples`` and draws, per
(model, subset) cell, the two groups' score distributions as filled kernel densities.
Two x-axis variants are produced:

  raw    — the score exactly as scored, on a linear axis
  znorm  — standardized within each trajectory. The scorer only ever ranks steps
           INSIDE one trajectory, so pooling raw scores across trajectories adds
           between-trajectory scale variation the ranking never sees; removing it
           shows the separation the argmax actually acts on.

Each variant writes one figure per cell plus a ``combined`` grid (rows = backbone,
cols = subset), as PDF (for the manuscript) and PNG (for browsing). ``summary.tsv``
is the table-view twin and ranks cells by separability:

  pooled_auc_<variant>  what the figure shows — overlap of the two plotted curves
  within_traj_auc       what the method acts on; invariant to any per-trajectory
                        monotone rescaling, so one value covers every variant
  both: 0.5 = the groups coincide, 1.0 = every error step scores below every other

    # from v2/
    python -m src.analysis.plot_score_dist --scores artifacts/score-dist/ww/triples/steps.tsv
    python -m src.analysis.plot_score_dist --scores .../steps.tsv --hist --per-seed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

# House palette (2026-08-31): the manuscript's orangeinplot / dark purpleinplot, as in
# the sensitivity and qualitative figures. The legend labels and summary.tsv remain the
# relief -- identity never rests on hue alone here.
C_ERROR, C_NORMAL = "#E8834F", "#4F4D8A"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

MODEL_DISPLAY = {"qwen3.5-9b": "Qwen3.5-9B", "deepseek-8b": "DeepSeek-8B"}
SUBSET_DISPLAY = {"algorithm-generated": "WW-AG", "hand-crafted": "WW-HC"}
VARIANTS = ("raw", "znorm")
XLABEL = {
    "raw":   r"base score  $\pi_\mathcal{C}(v_t)$",
    # The scorer only ever ranks steps WITHIN a trajectory, so pooling raw scores
    # across trajectories adds between-trajectory scale variation the ranking never
    # sees -- which is what smears the two groups together. Standardizing per
    # trajectory shows the separation the method actually acts on.
    "znorm": r"base score, standardized within trajectory",
}


def _with_x(v: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add the plotted x column for a variant (see XLABEL for why znorm exists)."""
    if variant == "znorm":
        g = v.groupby(["model", "subset", "seed", "traj_idx"])["score"]
        v["x"] = (v.score - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        return v.dropna(subset=["x"])
    v["x"] = v.score
    return v


def pooled_auc(v: pd.DataFrame, col: str = "x") -> float:
    """P[an error step scores below an ordinary step drawn from ANY trajectory].

    This is what the reader's eye measures on the figure -- the overlap of the two
    plotted histograms -- as opposed to rank_auc, which is what the method acts on.
    """
    e, o = v[col][v.is_mistake].to_numpy(), v[col][~v.is_mistake].to_numpy()
    if not len(e) or not len(o):
        return float("nan")
    # Mann-Whitney U with ties averaged, so total overlap scores exactly 0.5.
    ranks = pd.Series(np.concatenate([e, o])).rank(method="average").to_numpy()
    u_greater = (ranks[:len(e)].sum() - len(e) * (len(e) + 1) / 2) / (len(e) * len(o))
    return float(1.0 - u_greater)        # lower score = error, so invert


def rank_auc(df: pd.DataFrame) -> float:
    """P[error step scores BELOW an ordinary step of the same trajectory].

    Within-trajectory because that is the only comparison the ranking ever makes;
    0.5 = chance, 1.0 = the error step is always the lowest-scoring step.
    """
    vals = []
    for _, g in df.groupby(["model", "subset", "seed", "traj_idx"], sort=False):
        err, oth = g.score[g.is_mistake].to_numpy(), g.score[~g.is_mistake].to_numpy()
        if len(err) != 1 or not len(oth):
            continue
        vals.append(float((err[0] < oth).mean() + 0.5 * (err[0] == oth).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))   # sparse horizontal lines
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)


def _panel(ax, cell: pd.DataFrame, variant: str, title: str | None,
           legend: bool = True, show_hist: bool = False) -> None:
    """One cell: overlaid error/ordinary distributions on the chosen x-variant."""
    v = _with_x(cell.copy(), variant)

    err, non = v.x[v.is_mistake].to_numpy(), v.x[~v.is_mistake].to_numpy()
    # Full observed range: the axis carries the true score scale end to end, with no
    # percentile clipping hiding the tails.
    lo, hi = float(v.x.min()), float(v.x.max())
    # Bin count follows the SMALLER group (the error steps): finer bins only add
    # sampling spikes there, which would compete with the KDE lines for attention.
    bins = np.linspace(lo, hi, 30)
    grid = np.linspace(lo, hi, 400)

    for vals, color, label in ((non, C_NORMAL, "Ordinary steps"),
                               (err, C_ERROR, "Decisive-error steps")):
        if show_hist:
            ax.hist(vals, bins=bins, density=True, color=color, alpha=0.38,
                    edgecolor=SURFACE, linewidth=0.5, zorder=2)
        if len(vals) > 2 and np.ptp(vals) > 0:
            kde = gaussian_kde(vals)
            dens = kde(grid)
            # Unfilled since 2026-08-31: the lines alone carry the shape, drawn in the
            # darker house colors and a heavier weight, as in the other figures.
            ax.plot(grid, dens, color=color, linewidth=2.2, zorder=4,
                    label=f"{label}  (n={len(vals):,})", solid_capstyle="round")
    ax.set_ylim(bottom=0)

    ax.set_xlim(lo, hi)
    ax.set_ylabel("density", color=INK_2, fontsize=9)
    ax.set_xlabel(XLABEL[variant], color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=8)
    _style(ax)
    if legend:
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right",
                        handlelength=1.6, borderpad=0.2)
        for t in leg.get_texts():
            t.set_color(INK_2)


def _save(fig, out: Path, stem: str, formats: tuple[str, ...] = ("pdf",)) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        written.append(p)
    plt.close(fig)
    return written


def run(scores_tsv: Path, out_root: Path, per_seed: bool = False,
        show_hist: bool = False) -> None:
    df = pd.read_csv(scores_tsv, sep="\t")
    models = [m for m in MODEL_DISPLAY if m in set(df.model)] or sorted(set(df.model))
    subsets = [s for s in SUBSET_DISPLAY if s in set(df.subset)] or sorted(set(df.subset))
    cells = [(m, s) for m in models for s in subsets
             if not df[(df.model == m) & (df.subset == s)].empty]

    rows, n_files = [], 0
    for variant in VARIANTS:
        out = out_root / variant
        for model, subset in cells:                        # single-cell figures
            cell = df[(df.model == model) & (df.subset == subset)]
            fig, ax = plt.subplots(figsize=(5.0, 3.2), facecolor=SURFACE)
            _panel(ax, cell, variant, SUBSET_DISPLAY.get(subset, subset), show_hist=show_hist)
            n_files += len(_save(fig, out, f"{model}_{subset}"))
            if per_seed:
                for seed, g in cell.groupby("seed"):
                    fig, ax = plt.subplots(figsize=(5.0, 3.2), facecolor=SURFACE)
                    _panel(ax, g, variant, SUBSET_DISPLAY.get(subset, subset), show_hist=show_hist)
                    n_files += len(_save(fig, out / "per-seed",
                                         f"{model}_{subset}_seed-{seed}"))

        fig, axes = plt.subplots(len(models), len(subsets), squeeze=False,
                                 figsize=(4.8 * len(subsets), 3.0 * len(models)),
                                 facecolor=SURFACE)
        for i, model in enumerate(models):                 # 2x2 combined grid
            for j, subset in enumerate(subsets):
                cell = df[(df.model == model) & (df.subset == subset)]
                ax = axes[i][j]
                if cell.empty:
                    ax.axis("off")
                    continue
                _panel(ax, cell, variant, SUBSET_DISPLAY.get(subset, subset),
                       legend=(i == 0 and j == 0), show_hist=show_hist)
            # Panel titles name only the subset, so the backbone identifies its ROW
            # here -- otherwise the two rows would be labelled identically. Anchored to
            # the y-label (not a fixed axes offset) so it clears tick labels of any width.
            ax0 = axes[i][0]
            ax0.annotate(MODEL_DISPLAY.get(model, model), xy=(0, 0.5),
                         xytext=(-ax0.yaxis.labelpad - 14, 0),
                         xycoords=ax0.yaxis.label, textcoords="offset points",
                         rotation=90, ha="right", va="center",
                         color=INK, fontsize=10)
        fig.tight_layout(pad=1.4)
        n_files += len(_save(fig, out, "combined"))

    for model, subset in cells:                            # table-view twin
        cell = df[(df.model == model) & (df.subset == subset)]
        err, ordn = cell[cell.is_mistake], cell[~cell.is_mistake]
        rows.append({
            "dataset": cell.dataset.iloc[0] if "dataset" in cell else "",
            "model": model, "subset": subset,
            "n_error": len(err), "n_ordinary": len(ordn),
            # Separability AS PLOTTED, per variant: 0.5 = the two histograms coincide,
            # 1.0 = every error step sits below every ordinary step.
            **{f"pooled_auc_{v}": pooled_auc(_with_x(cell.copy(), v)) for v in VARIANTS},
            # Invariant to any per-trajectory monotone rescaling, so one value covers
            # every variant -- this is the separation the argmax actually sees.
            "within_traj_auc": rank_auc(cell),
            "median_error": err.score.median(), "median_ordinary": ordn.score.median(),
            "position": cell.position.iloc[0],
            "band": f"[{cell.c_begin.iloc[0]},{cell.c_end.iloc[0]})",
            "seeds": ",".join(str(s) for s in sorted(set(cell.seed))),
        })
    summary = pd.DataFrame(rows).sort_values(f"pooled_auc_{VARIANTS[-1]}", ascending=False)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "summary.tsv", sep="\t", index=False)

    print(f"\n  separability, best first ({VARIANTS[-1]}); 0.5 = fully overlapping:")
    for _, r in summary.iterrows():
        pooled = "  ".join(f"{v}={r[f'pooled_auc_{v}']:.3f}" for v in VARIANTS)
        print(f"    {MODEL_DISPLAY.get(r.model, r.model):12s} "
              f"{SUBSET_DISPLAY.get(r.subset, r.subset):20s} {pooled}  "
              f"within-traj={r.within_traj_auc:.3f}  n_err={r.n_error}")
    print(f"  wrote {n_files} figures under {out_root}  (+ summary.tsv)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, default=Path("artifacts/score-dist/ww/scores.tsv"))
    p.add_argument("--out", type=Path, default=None,
                   help="output root (default: <scores dir>/plots)")
    p.add_argument("--per-seed", action="store_true", help="also draw one panel per seed")
    p.add_argument("--hist", action="store_true",
                   help="overlay binned histograms behind the density curves")
    a = p.parse_args()
    run(a.scores, a.out or a.scores.parent / "plots", a.per_seed, a.hist)


if __name__ == "__main__":
    main()
