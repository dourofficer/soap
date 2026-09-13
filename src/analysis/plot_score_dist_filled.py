"""Base-score distribution in the filled-density style of ``artifacts/distribution_plot.png``.

Same data as ``plot_score_dist_triples`` (the per-step table from ``score_dist_triples``),
different look: each group is a kernel density drawn as a colored line over a translucent
fill of the same hue, on a plain boxed axis with serif type and no grid. That is the
style of the reference figure the manuscript's other density plots follow.

One figure per call, for one (model, subset, window, seed) cell -- the unit the
manuscript actually places. ``--seed pooled`` pools the window's three seeds.

    python -m src.analysis.plot_score_dist_filled \\
        --steps artifacts/score-dist/correct-error/triples/steps.tsv \\
        --model qwen3.5-9b --subset arc --triple 6,7,8 --seed 7 \\
        --out manuscript/assets/qwen3.5-9b_arc_seeds-6-7-8_seed-7.pdf
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

from .plot_score_dist import XLABEL, _with_x  # noqa: E402

# Sampled from the reference figure: salmon for the error steps, dusty purple for
# the ordinary ones. Fills are the same hue at low opacity, so overlaps blend.
C_ERROR, C_NORMAL = "#EE9A6E", "#7B72A6"
FILL_ALPHA, LINE_W = 0.22, 1.6

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False,
}


def draw(ax, cell: pd.DataFrame, variant: str = "raw", title: str | None = None,
         legend: bool = True, counts: bool = False) -> None:
    """One filled-density panel: ordinary steps behind, decisive-error steps in front."""
    v = _with_x(cell.copy(), variant)
    err, non = v.x[v.is_mistake].to_numpy(), v.x[~v.is_mistake].to_numpy()
    lo, hi = float(v.x.min()), float(v.x.max())
    grid = np.linspace(lo, hi, 500)
    peak = 0.0

    for vals, color, label in ((non, C_NORMAL, "Ordinary steps"),
                               (err, C_ERROR, "Decisive-error steps")):
        if len(vals) < 3 or np.ptp(vals) == 0:
            continue
        dens = gaussian_kde(vals)(grid)
        peak = max(peak, float(dens.max()))
        if counts:
            label = f"{label} (n={len(vals):,})"
        ax.fill_between(grid, 0, dens, color=color, alpha=FILL_ALPHA, linewidth=0, zorder=2)
        ax.plot(grid, dens, color=color, linewidth=LINE_W, zorder=3, label=label,
                solid_capstyle="round")

    ax.set_xlim(lo, hi)
    # Headroom above the taller curve keeps the legend off the peaks.
    ax.set_ylim(0, 1.3 * peak if legend else 1.05 * peak)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlabel(XLABEL[variant], fontsize=10)
    ax.tick_params(colors="black")
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    if legend:
        ax.legend(fontsize=9, loc="upper right", handlelength=1.8,
                  borderpad=0.3, labelspacing=0.3)


def select(df: pd.DataFrame, model: str, subset: str, triple: str, seed: str) -> pd.DataFrame:
    cell = df[(df.model == model) & (df.subset == subset) & (df.triple == triple)]
    if cell.empty:
        raise SystemExit(f"no rows for model={model} subset={subset} triple={triple}")
    if seed != "pooled":
        cell = cell[cell.seed == int(seed)]
        if cell.empty:
            raise SystemExit(f"seed {seed} not in window {triple}")
    return cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=Path,
                   default=Path("artifacts/score-dist/correct-error/triples/steps.tsv"))
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--subset", default="arc")
    p.add_argument("--triple", default="6,7,8", help="seed window as in steps.tsv, e.g. 6,7,8")
    p.add_argument("--seed", default="7", help="one seed of the window, or 'pooled'")
    p.add_argument("--variant", choices=("raw", "znorm"), default="raw")
    p.add_argument("--title", default=None)
    p.add_argument("--counts", action="store_true", help="append n= to legend labels")
    p.add_argument("--figsize", type=float, nargs=2, default=(3.3, 2.35),
                   help="inches; small by design so type scales up in a wrapfigure")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: <steps dir>/plots/filled/<stem>.pdf)")
    a = p.parse_args()

    df = pd.read_csv(a.steps, sep="\t")
    cell = select(df, a.model, a.subset, a.triple, a.seed)
    stem = f"{a.model}_{a.subset}_seeds-{a.triple.replace(',', '-')}_seed-{a.seed}"
    out = a.out or a.steps.parent / "plots" / "filled" / f"{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=a.figsize, facecolor="white")
        draw(ax, cell, a.variant, a.title, counts=a.counts)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"wrote {out}  (n_error={int(cell.is_mistake.sum())}, "
          f"n_ordinary={int((~cell.is_mistake).sum())})")


if __name__ == "__main__":
    main()
