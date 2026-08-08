"""Publication figures for the qualitative examples (see :mod:`src.analysis.qualitative`).

Each example is drawn as two stacked panels sharing a step axis:

  top     the per-step score BEFORE rescoring against the score AFTER, so the base
          scorer's argmax can be seen sitting on a downstream step while SOAP's argmax
          lands on the gold decisive step;
  bottom  the promotion itself, ``delta_t = final - oriented``, as stems.

The bottom panel exists because the correction is often small next to the score range
(and, wherever a step is nobody's top-``w`` predecessor, exactly zero -- the curves
then coincide by construction, not by a plotting artifact). On its own axis every
non-zero correction is legible no matter how small, and which steps were promoted at
all becomes readable at a glance.

The "before" curve is the ORIENTED base score, not the raw one. The raw ``proj`` score
is in the native "lower = error" convention, which would need an inverted axis to
read; the oriented score is what the rescoring actually adds to
(``final = oriented + gamma * B`` whenever ``score_norm='none'``), so both curves live
on one axis, are additively comparable, and the shaded band between them IS the
correction. Never renormalize the curves separately -- that comparability is the point.

Chrome follows ``plot_score_dist`` (same grid/axis inks, same save conventions). The
series colours are configurable; see ``COLORS``. This module additionally pins Type-42
fonts and a serif face to match the manuscript body text, scoped to the drawing calls
so they cannot leak into any other figure produced in the same process.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Series colours: the manuscript's own purpleinplot/orangeinplot (main.tex), lightened.
# Override per run with --set colors.base=... / --set colors.soap=...
COLORS = {"base": "#807EAF", "soap": "#F1A484"}

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

MARKER_LIMIT = 25          # per-point markers only while they still read as points
LOG_SCALE_RATIO = 50.0     # 1/proj can spike; switch scales rather than clip
GALLERY_COLS, GALLERY_ROWS = 4, 4

# Type-42 keeps the camera-ready free of Type-3 fonts; serif matches the ICLR body.
RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=3, width=0.8)


def _save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        written.append(p)
    plt.close(fig)
    return written


def _use_log(y: np.ndarray, scale: str) -> bool:
    if scale in ("log", "linear"):
        return scale == "log"
    med = float(np.median(y))
    return med > 0 and float(y.max()) / med > LOG_SCALE_RATIO


def _sci(ax) -> None:
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    ax.yaxis.get_offset_text().set(fontsize=7, color=MUTED)


def _scores(ax, x, base, soap, cand, colors, scale, compact=False) -> bool:
    """Top panel: the before/after curves, the gold step, and each method's argmax."""
    marks = len(x) <= MARKER_LIMIT and not compact
    lw = 1.6 if len(x) <= MARKER_LIMIT else 1.0

    ax.axvline(cand["true_step"], color=INK_2, ls=(0, (4, 3)), lw=1.0, zorder=1)
    # Backprop only ever adds evidence, so the band is the correction itself.
    ax.fill_between(x, base, soap, color=colors["soap"], alpha=0.22, lw=0, zorder=2)
    ax.plot(x, soap, color=colors["soap"], ls="-", lw=lw, zorder=3,
            solid_capstyle="round", marker="^" if marks else None, ms=3.4, mew=0)
    # The dashed base curve goes ON TOP: where the two coincide, the solid SOAP curve
    # shows through the gaps instead of being hidden under it.
    ax.plot(x, base, color=colors["base"], ls=(0, (3.5, 2.5)), lw=lw, zorder=4,
            marker="o" if marks else None, ms=3.1, mew=0)

    for pred, y, colour, shape in ((cand["base_pred_step"], base, colors["base"], "o"),
                                   (cand["soap_pred_step"], soap, colors["soap"], "^")):
        i = int(np.flatnonzero(x == pred)[0])
        ax.plot([pred], [y[i]], marker=shape, ms=6 if compact else 8, mfc=colour,
                mec=SURFACE, mew=1.4 if compact else 2.0, ls="none", zorder=5)

    logged = _use_log(np.concatenate([base, soap]), scale)
    ax.set_yscale("log") if logged else _sci(ax)
    ax.margins(x=0.04)
    _style(ax)
    return logged


def _delta(ax, x, base, soap, cand, colors) -> None:
    """Bottom panel: the promotion each step received, on its own scale."""
    d = soap - base
    ax.axvline(cand["true_step"], color=INK_2, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.vlines(x, 0, d, color=colors["soap"], lw=2.2 if len(x) <= MARKER_LIMIT else 1.2,
              zorder=3)
    gold = int(np.flatnonzero(x == cand["true_step"])[0])
    ax.plot([x[gold]], [d[gold]], marker="^", ms=7, mfc=colors["soap"], mec=SURFACE,
            mew=1.6, ls="none", zorder=4)
    ax.margins(x=0.04)
    ax.set_ylim(0, max(float(d.max()) * 1.25, 1e-12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    _sci(ax)
    _style(ax)
    ax.set_ylabel(r"$\Delta$", color=INK_2)


def _arrays(steps):
    return (steps["step_idx"].to_numpy(),
            steps["oriented"].to_numpy(dtype=float),
            steps["final"].to_numpy(dtype=float))


def _handles(colors) -> list:
    return [
        Line2D([], [], color=colors["base"], ls=(0, (3.5, 2.5)), lw=1.6, marker="o",
               ms=3.6, label="w/o rescoring (SVD)"),
        Line2D([], [], color=colors["soap"], ls="-", lw=1.6, marker="^", ms=3.6,
               label="SOAP (full)"),
        Line2D([], [], color=INK_2, ls=(0, (4, 3)), lw=1.0,
               label=r"decisive error $t^\star$"),
    ]


def _legend(target, colors, **kw):
    leg = target.legend(handles=_handles(colors), frameon=False, handlelength=2.4, **kw)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return leg


def _example_fig(steps, cand: dict, colors: dict, scale: str, title: str | None):
    """One example: scores over a promotion panel, sharing the step axis."""
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(4.6, 3.5), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    x, base, soap = _arrays(steps)
    logged = _scores(top, x, base, soap, cand, colors, scale)
    _delta(bot, x, base, soap, cand, colors)
    top.set_ylabel("attribution score", color=INK_2)
    bot.set_xlabel("step index", color=INK_2)
    bot.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    if title:
        top.set_title(title, color=INK, fontsize=9.5, loc="left", pad=6)
    _legend(top, colors, loc="best", borderaxespad=0.3)
    return fig, logged


def plot_examples(examples: list[dict], colors: dict, scale: str) -> list[Path]:
    """One two-panel figure per example, written beside its own logs."""
    written: list[Path] = []
    with plt.rc_context(RC):
        for e in examples:
            c = e["cand"]
            title = (f"{e['cell']['column']} · seed {c['seed']} · traj {c['traj_idx']}"
                     + ("" if c["manuscript_window"] else "  (non-window seed)"))
            fig, logged = _example_fig(e["steps"], c, colors, scale, title)
            # Explicit margins, not tight_layout: shared x + height_ratios make these
            # axes "incompatible with tight_layout" and it silently mis-sizes them.
            fig.subplots_adjust(left=0.17, right=0.98, top=0.91, bottom=0.13, hspace=0.14)
            written += _save(fig, Path(e["dir"]), "scores")
            e["log_scale"] = logged
    return written


def plot_gallery(cell_label: str, examples: list[dict], out_dir: Path,
                 colors: dict, scale: str) -> list[Path]:
    """Contact sheets of every example for the cell, paginated, for browsing/picking."""
    written: list[Path] = []
    per_page = GALLERY_COLS * GALLERY_ROWS
    pages = math.ceil(len(examples) / per_page)
    with plt.rc_context(RC):
        for p in range(pages):
            chunk = examples[p * per_page:(p + 1) * per_page]
            rows = math.ceil(len(chunk) / GALLERY_COLS)
            fig, axes = plt.subplots(rows, GALLERY_COLS, figsize=(2.6 * GALLERY_COLS,
                                                                 1.9 * rows),
                                     facecolor=SURFACE, squeeze=False)
            for ax, e in zip(axes.ravel(), chunk):
                x, base, soap = _arrays(e["steps"])
                _scores(ax, x, base, soap, e["cand"], colors, scale, compact=True)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                ax.set_title(f"seed {e['cand']['seed']} · traj {e['cand']['traj_idx']}"
                             f"  ({e['cand']['true_step']}$\\leftarrow$"
                             f"{e['cand']['base_pred_step']})",
                             color=INK, fontsize=7.5, loc="left", pad=3)
                ax.tick_params(labelsize=6)
                ax.yaxis.get_offset_text().set(fontsize=5.5)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            fig.tight_layout(pad=0.5, rect=(0, 0.05, 1, 1))
            _legend(fig, colors, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
            stem = "gallery" if pages == 1 else f"gallery_p{p + 1}"
            written += _save(fig, out_dir, stem)
    return written


def plot_combined(headline: list[dict], out_root: Path, colors: dict,
                  scale: str) -> list[Path]:
    """The manuscript-ready figure: the top-ranked example of each cell, side by side."""
    n = len(headline)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, n, figsize=(3.9 * n, 3.6), facecolor=SURFACE,
                                 squeeze=False, sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1]})
        for j, (e, tag) in enumerate(zip(headline, "abcdefgh")):
            top, bot = axes[0][j], axes[1][j]
            x, base, soap = _arrays(e["steps"])
            _scores(top, x, base, soap, e["cand"], colors, scale)
            _delta(bot, x, base, soap, e["cand"], colors)
            top.set_title(f"({tag}) {e['cell']['column']}", color=INK, fontsize=10,
                          loc="left", pad=6)
            bot.set_xlabel("step index", color=INK_2)
            bot.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            if j == 0:
                top.set_ylabel("attribution score", color=INK_2)
            else:
                bot.set_ylabel("")
        # Explicit margins (tight_layout cannot handle shared x + height_ratios), with
        # the bottom band reserved for the shared legend.
        fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.20,
                            hspace=0.14, wspace=0.22)
        _legend(fig, colors, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005))
        return _save(fig, out_root, "combined")
