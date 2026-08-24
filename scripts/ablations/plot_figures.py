"""Draw the manuscript's ablation figures from `results-ablations/`.

Four PDFs (with PNG previews) land in `artifacts/ablations/`:

- `fig_sensitivity`          — main text: rows WW-AG / WW-HC on Qwen3.5-9B, panels
                               (a) gamma, (b) representation layer (line), (c)
                               attention band. Dashed orange line = base score of
                               the selected configuration (no rescoring); darker
                               bar / large marker = the selected configuration.
- `fig_transfer_datasize`    — main text: (a) 4x4 source->target heatmap (Qwen,
                               test-selected), (b) accuracy vs reference-data fraction.
- `fig_sensitivity_appendix` — all eight (backbone, subset) cells, plus the window-w panel.
- `fig_transfer_appendix`    — heatmaps for both backbones and both conventions.

`--print-tables` dumps ready-to-paste tabular bodies for the hand-typed tables (best
per column wrapped in `\\best{}`), so no number is typed from memory. Nothing is
recomputed here: every value is read from a TSV that an ablation runner wrote.

    python scripts/ablations/plot_figures.py [--out artifacts/ablations] [--only NAME]
    python scripts/ablations/plot_figures.py --print-tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402
from matplotlib.patches import Patch, Rectangle    # noqa: E402
from matplotlib.ticker import MaxNLocator         # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results-ablations"
OUT = REPO / "artifacts" / "ablations"

# House palette (src/analysis/qualitative_plot.py): manuscript purpleinplot/orangeinplot.
PURPLE, PURPLE_DARK, ORANGE = "#807EAF", "#4F4D8A", "#E8834F"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
}
CMAP = LinearSegmentedColormap.from_list("soap", ["#ffffff", "#7676a4"])

BACKBONES = {"qwen3.5-9b": "Qwen3.5-9B", "deepseek-8b": "DeepSeek-8B"}
SUBSETS = {"algorithm-generated": "WW-AG", "hand-crafted": "WW-HC",
           "captain": "TE-Cap", "magentic": "TE-Mag"}
CE = ("arc", "gaia", "hotpot", "math500", "mmlu_pro", "musique", "wikimqa")
TARGETS = ["WW-AG", "WW-HC", "TE-Cap", "TE-Mag"]
MAIN_CELLS = [("qwen3.5-9b", "algorithm-generated"), ("qwen3.5-9b", "hand-crafted")]
APPENDIX_CELLS = [("qwen3.5-9b", "captain"), ("qwen3.5-9b", "magentic"),
                  ("deepseek-8b", "algorithm-generated"), ("deepseek-8b", "hand-crafted"),
                  ("deepseek-8b", "captain"), ("deepseek-8b", "magentic")]


# ----------------------------------------------------------------------------- data
def tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name, sep="\t", dtype={"w": str, "fraction": str})


def cell(df: pd.DataFrame, model: str, subset: str) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["subset"] == subset)]


def pos_label(p: str) -> str:
    if p == "embed":
        return "E"
    if p.endswith("_normed"):
        return "N"
    return p.split("/")[1]


def layer_axis(df6a: pd.DataFrame, model: str, subset: str):
    """(labels, soap, base, anchor_idx) over the representation positions, in depth order."""
    c = cell(df6a, model, subset)
    c = c[c["position"] != "ens-mid3"]           # the layer ensemble is not a depth
    soap = c[c["variant"] == "anchor-soap"].reset_index(drop=True)
    base = c[c["variant"] == "anchor-base"].set_index("position")
    labels = [pos_label(p) for p in soap["position"]]
    anchor = int(np.flatnonzero(soap["is_anchor"].values)[0])
    return labels, 100 * soap["step_acc_test"].values, \
        100 * base.loc[soap["position"], "step_acc_test"].values, anchor


# ---------------------------------------------------------------------------- style
def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=2.5, width=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def save(fig, stem: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor=SURFACE)
        written.append(p)
    plt.close(fig)
    return written


def bars(ax, labels, vals, anchor: int, base: float, err=None) -> None:
    x = np.arange(len(vals))
    colors = [PURPLE_DARK if i == anchor else PURPLE for i in x]
    ax.bar(x, vals, 0.72, color=colors, linewidth=0,
           yerr=err, error_kw={"ecolor": MUTED, "elinewidth": 0.6, "capsize": 0})
    ax.axhline(base, color=ORANGE, linestyle="--", linewidth=1.0, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(vals) - 0.4)


def layer_line(ax, labels, soap, base_per_layer, anchor: int, base: float, step: int = 1) -> None:
    """Many positions (DeepSeek): a line over depth, REDE Fig. 14 style."""
    x = np.arange(len(soap))
    ax.plot(x, base_per_layer, color=PURPLE, linewidth=0.9, linestyle=":", zorder=2)
    ax.plot(x, soap, color=PURPLE_DARK, linewidth=1.1, marker="o", markersize=2.5, zorder=3)
    ax.plot([anchor], [soap[anchor]], marker="o", markersize=5, color=PURPLE_DARK, zorder=4)
    ax.axhline(base, color=ORANGE, linestyle="--", linewidth=1.0, zorder=3)
    ticks = list(range(0, len(soap), 4)) if len(soap) > 12 else list(range(0, len(soap), step))
    if len(soap) - 1 - ticks[-1] >= 2:
        ticks.append(len(soap) - 1)
    ax.set_xticks(ticks, [labels[i] for i in ticks])
    ax.set_xlim(-0.6, len(soap) - 0.4)


def ylim(ax, *arrays) -> None:
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    lo, hi = max(0.0, v.min() - 4), v.max() + 4
    ax.set_ylim(lo, hi)


# ------------------------------------------------------------------------ figures
PANELS = {
    "gamma": (r"Propagation strength $\gamma$", 1.15),
    "w": (r"Window $w$", 0.85),
    "layer": (r"Representation layer $l^\star$", 1.3),
    "band": (r"Attention band $L^\star$", 0.8),
}


def sensitivity(cells, stem: str, out: Path, panels=("gamma", "layer", "band")) -> list[Path]:
    """One row per (backbone, subset) cell, one column per varied axis."""
    d5, d4, d6a, d6b = tsv("a5_gamma.tsv"), tsv("a4_window.tsv"), \
        tsv("a6a_rep_layer.tsv"), tsv("a6b_attn_band.tsv")
    n, m = len(cells), len(panels)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(n, m, figsize=(5.5, 1.35 * n + 0.25),
                                 gridspec_kw={"width_ratios": [PANELS[k][1] for k in panels]})
        axes = np.atleast_2d(axes)
        for r, (model, subset) in enumerate(cells):
            tag = SUBSETS[subset] if all(mm == "qwen3.5-9b" for mm, _ in cells) \
                else f"{BACKBONES[model].split('-')[0]} {SUBSETS[subset]}"
            # The reference level of every panel: the base score of the selected
            # configuration (gamma=0 of the per-seed sweep, mean over the triple).
            g = cell(d5, model, subset).groupby("gamma")["step_acc_test"]
            gm, gs = 100 * g.mean(), 100 * g.std(ddof=0)
            base = gm.loc[0.0]
            anchor_gamma = cell(d4, model, subset)["gamma"].iloc[0]

            for ax, key in zip(axes[r], panels):
                if key == "gamma":
                    gammas = gm.index.values
                    bars(ax, [f"{x:g}" if i % 2 == 0 else "" for i, x in enumerate(gammas)],
                         gm.values, int(np.argmin(np.abs(gammas - anchor_gamma))), base, err=gs.values)
                    ax.set_xlabel(r"$\gamma$", labelpad=1)
                    ylim(ax, gm.values + gs.values, gm.values - gs.values, base)
                elif key == "w":
                    order = ["1", "2", "3", "4", "5", "all"]
                    w = cell(d4, model, subset).set_index("w").loc[order]
                    bars(ax, order, 100 * w["step_acc_test"].values,
                         int(np.flatnonzero(w["is_anchor"].values)[0]), base)
                    ax.set_xlabel(r"$w$", labelpad=1)
                    ylim(ax, 100 * w["step_acc_test"].values, base)
                elif key == "layer":
                    labels, soap, base_layer, anchor = layer_axis(d6a, model, subset)
                    layer_line(ax, labels, soap, base_layer, anchor, base, step=1 if m == 3 else 2)
                    ax.set_xlabel("layer", labelpad=1)
                    ylim(ax, soap, base_layer, base)
                elif key == "band":
                    b = cell(d6b, model, subset).copy()
                    b["lo"] = b["layer_range"].str.split("-").str[0].astype(int)
                    b = b.sort_values("lo")
                    bars(ax, [x.replace("-", "–") for x in b["layer_range"]], 100 * b["step_acc_test"].values,
                         int(np.flatnonzero(b["is_anchor"].values)[0]), base)
                    ax.set_xlabel("attn. layers", labelpad=1)
                    if b["layer_range"].str.len().max() > 3:
                        ax.tick_params(axis="x", labelsize=5, rotation=30)
                    ylim(ax, 100 * b["step_acc_test"].values, base)
                style(ax)
                if r == 0:
                    title = PANELS[key][0] if m == 3 else PANELS[key][0].replace("Representation layer", "Layer")
                    ax.set_title(f"({'abcd'[panels.index(key)]}) {title}", pad=4, color=INK)
            axes[r, 0].set_ylabel(f"{tag}\nstep acc. (%)", labelpad=2)

        handles = [Patch(color=PURPLE_DARK, label="selected configuration (Table 1)"),
                   Patch(color=PURPLE, label=r"$+$SOAP at the varied value"),
                   Line2D([], [], color=ORANGE, linestyle="--", linewidth=1,
                          label="base score (no rescoring)"),
                   Line2D([], [], color=PURPLE_DARK, marker="o", markersize=2.5, linewidth=1.1,
                          label=r"$+$SOAP at that layer"),
                   Line2D([], [], color=PURPLE, linestyle=":", linewidth=1,
                          label="base score at that layer")]
        fig.subplots_adjust(wspace=0.34, hspace=0.5)
        y = axes[-1, 0].get_position().y0 - 0.42 / fig.get_figheight()   # just under the x-labels
        fig.legend(handles=handles, ncol=3, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, y), handlelength=1.6, columnspacing=1.2)
        return save(fig, stem, out)


def heatmap(ax, grid: pd.DataFrame, title: str | None = None) -> None:
    v = grid.values
    ax.imshow(v, cmap=CMAP, vmin=0, vmax=max(50, v.max()), aspect="equal")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v[i, j] > 0.8 * max(50, v.max()) else INK)
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.2))
    ax.set_xticks(range(4), TARGETS)
    ax.set_yticks(range(4), TARGETS)
    ax.set_xlabel("Target (evaluated on)", labelpad=2)
    ax.set_ylabel("Source (reference fit on)", labelpad=2)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, pad=4)


def transfer_grid(df: pd.DataFrame, model: str, convention: str) -> pd.DataFrame:
    """Source x target SOAP step accuracy (%). The diagonal always carries the
    main-experiment in-distribution number (the test-convention diagonal, which the
    runner asserted equals Table 1), whichever convention the off-diagonal cells use —
    the reporting rule of experiments/todo.md (E1)."""
    def grid(conv):
        d = df[(df["model"] == model) & (df["convention"] == conv) & (df["row"] == "soap")]
        return (d.pivot(index="source", columns="target", values="step_acc_test") * 100).loc[TARGETS, TARGETS]
    g = grid(convention)
    if convention != "test":
        t = grid("test")
        for k in TARGETS:
            g.loc[k, k] = t.loc[k, k]
    return g


def fig_transfer_datasize(out: Path) -> list[Path]:
    e1, a7 = tsv("e1_transfer.tsv"), tsv("a7_datasize.tsv")
    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.15),
                                       gridspec_kw={"width_ratios": [1, 1.25]})
        heatmap(ax1, transfer_grid(e1, "qwen3.5-9b", "val"), "(a) Cross-distribution transfer")

        x = [10, 20, 30]
        for subset, ls in (("algorithm-generated", "-"), ("hand-crafted", "--")):
            c = cell(a7, "qwen3.5-9b", subset)
            for row, color, m in (("base", PURPLE, "s"), ("soap", ORANGE, "o")):
                y = 100 * c[c["row"] == row].set_index("fraction").loc[["1/3", "2/3", "1"], "step_acc_test"].values
                ax2.plot(x, y, color=color, linestyle=ls, marker=m, markersize=3.5, linewidth=1.1)
        style(ax2)
        ax2.set_xticks(x, ["10%", "20%", "30%"])
        ax2.set_xlim(7, 33)
        ax2.set_ylim(ax2.get_ylim()[0] - 4.5, ax2.get_ylim()[1] + 1)
        ax2.set_xlabel("Reference data (% of corpus)", labelpad=2)
        ax2.set_ylabel("Step acc. (%)", labelpad=2)
        ax2.set_title("(b) Quantity of unlabeled reference data", color=INK, pad=4)
        handles = [Line2D([], [], color=ORANGE, marker="o", markersize=3.5, label=r"$+$SOAP"),
                   Line2D([], [], color=PURPLE, marker="s", markersize=3.5, label="Base score"),
                   Line2D([], [], color=INK_2, linestyle="-", label="WW-AG"),
                   Line2D([], [], color=INK_2, linestyle="--", label="WW-HC")]
        ax2.legend(handles=handles, ncol=2, frameon=False, loc="lower right", handlelength=1.8,
                   columnspacing=1.0, labelcolor=INK_2)
        fig.subplots_adjust(wspace=0.35)
        return save(fig, "fig_transfer_datasize", out)


def fig_transfer_appendix(out: Path) -> list[Path]:
    e1 = tsv("e1_transfer.tsv")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(5.0, 5.0))
        for i, conv in enumerate(("test", "val")):
            for j, model in enumerate(BACKBONES):
                heatmap(axes[i, j], transfer_grid(e1, model, conv),
                        f"{BACKBONES[model]}, {conv}-selected")
        fig.subplots_adjust(wspace=0.45, hspace=0.45)
        return save(fig, "fig_transfer_appendix", out)


FIGURES = {
    "fig_sensitivity": lambda out: sensitivity(MAIN_CELLS, "fig_sensitivity", out),
    "fig_transfer_datasize": fig_transfer_datasize,
    "fig_sensitivity_appendix": lambda out: sensitivity(
        MAIN_CELLS + APPENDIX_CELLS, "fig_sensitivity_appendix", out, panels=("gamma", "w", "layer", "band")),
    "fig_transfer_appendix": fig_transfer_appendix,
}


# ------------------------------------------------------------------------- tables
def _fmt(cols: list[list[float]], bold_max: bool = True) -> list[list[str]]:
    """Format a rows x cols matrix; column maxima wrapped in \\best{}."""
    m = np.array(cols, dtype=float)
    out = [[f"{v:.2f}" for v in row] for row in m]
    if bold_max:
        for j in range(m.shape[1]):
            for i in np.flatnonzero(np.isclose(m[:, j], m[:, j].max())):
                out[i][j] = r"\best{" + out[i][j] + "}"
    return out


def _emit(title: str, row_labels: list[str], matrix, header: list[str]) -> None:
    print(f"% ---- {title}")
    print("% " + " & ".join(header))
    for lab, row in zip(row_labels, _fmt(matrix)):
        print(f"{lab:<34s} & " + " & ".join(row) + r" \\")
    print()


def ce_macro(d: pd.DataFrame) -> float:
    """Macro-average over CE's seven subsets (the house rule for every CE number)."""
    assert set(d["subset"]) == set(CE), sorted(set(d["subset"]))
    return float(d["step_acc_test"].mean())


def print_tables() -> None:
    a1 = tsv("a1_scorefn/scorefn.tsv")
    rows = [("perplexity", "Perplexity"), ("random", "Random subspace"), ("top", "Top subspace"),
            ("tail", "Tail subspace"), ("full", "Full spectrum"), ("norm-l1", r"Norm $\ell_1$"),
            ("norm-l2", r"Norm $\ell_2$"), ("ours", "Spectral band (ours)")]
    for model in BACKBONES:
        cols = list(SUBSETS)
        mat = [[100 * cell(a1, model, s).set_index("row").loc[k, "step_acc_test"] for s in cols]
               for k, _ in rows]
        _emit(f"tab:scorefn {BACKBONES[model]}", [l for _, l in rows], mat, [SUBSETS[s] for s in cols])

    a2 = tsv("a2_weights.tsv"); a2 = a2[a2["with_gt"] == False]          # noqa: E712
    a3 = tsv("a3_position.tsv"); a3 = a3[a3["selected"] == True]          # noqa: E712
    wrows = [("base", "Base score (no rescoring)"), ("uniform-unnorm", "Uniform (unnormalized)"),
             ("uniform-norm", "Uniform (normalized)")]
    prows = [("temporal-z", "Temporal bias, z-scored"), ("temporal-raw", "Temporal bias, raw"),
             ("earliest-top5", "Earliest of top-$5$")]
    cols = ["algorithm-generated", "hand-crafted", "CE", "captain", "magentic"]

    def col(df, model, s, key):
        d = df[(df["model"] == model) & (df["row"] == key)]
        if s == "CE":
            return 100 * ce_macro(d[d["dataset"] == "correct-error"])
        return 100 * d[d["subset"] == s]["step_acc_test"].iloc[0]

    for model in BACKBONES:
        mat = [[col(a2, model, s, k) for s in cols] for k, _ in wrows]
        mat += [[col(a3, model, s, k) for s in cols] for k, _ in prows]
        mat += [[col(a2, model, s, "soap") for s in cols]]
        labels = [l for _, l in wrows] + [l for _, l in prows] + ["Attention-guided (ours)"]
        _emit(f"tab:weights {BACKBONES[model]}", labels, mat,
              ["WW-AG", "WW-HC", "CE", "TE-Cap", "TE-Mag"])
        lam = a3[(a3["model"] == model) & (a3["row"].str.startswith("temporal"))]
        print("% selected lambdas: " + "; ".join(
            f"{r.row}/{r.dataset}/{r.subset}={r['lambda']:g}" for _, r in lam.iterrows()))
        print()

    a7 = tsv("a7_datasize.tsv")
    for model, subset in MAIN_CELLS + APPENDIX_CELLS:
        c = cell(a7, model, subset)
        mat = [[100 * c[(c["fraction"] == f) & (c["row"] == r)]["step_acc_test"].iloc[0]
                for r in ("base", "soap")] for f in ("1/3", "2/3", "1")]
        n = [c[c["fraction"] == f]["n_train"].iloc[0] for f in ("1/3", "2/3", "1")]
        print(f"% ---- tab:datasize {BACKBONES[model]} {SUBSETS[subset]} (n_train per seed: {n})")
        for f, row in zip(("10\\%", "20\\%", "30\\%"), _fmt(mat, bold_max=False)):
            print(f"{f:<10s} & " + " & ".join(row) + r" \\")
        print()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", choices=sorted(FIGURES), action="append")
    ap.add_argument("--print-tables", action="store_true")
    a = ap.parse_args()
    if a.print_tables:
        print_tables()
        return
    for name in a.only or FIGURES:
        for p in FIGURES[name](a.out):
            print(p.relative_to(REPO))


if __name__ == "__main__":
    main()
