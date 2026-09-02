"""The SOAP overview figure (manuscript Figure: method).

Three stages, left to right, in the manuscript's own palette:

1. A frozen proxy model reads the failed trajectory step by step and yields, for
   every step, a pooled representation v_t and the attention mass it places on
   its predecessors.
2. The spectral base score: an SVD of the unlabeled reference steps, a spectral
   band C, the projection pi_C(v_t), and S = 1/(pi_C + eps). Its argmax lands on
   a downstream consequence of the decisive error (contamination).
3. Attention-guided rescoring: dependency weights w_{i,t} route each step's error
   signal back to the predecessors it depends on; S~ = S + gamma*B moves the argmax
   to t*.

Pure matplotlib, no data: the bars and weights are illustrative. Writes
manuscript/assets/soap_overview.pdf and a PNG preview under artifacts/method_figure/.

    python -m src.analysis.method_figure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_PDF = REPO / "manuscript" / "assets" / "soap_overview.pdf"
OUT_PNG = REPO / "artifacts" / "method_figure" / "soap_overview.png"

PURPLE, PURPLE_DARK, ORANGE, ORANGE_DARK = "#807EAF", "#4F4D8A", "#F1A484", "#D9722F"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
PANEL, PANEL_EDGE, AXIS = "#f6f5f1", "#d8d6cc", "#c3c2b7"
AGENTS = ["#b9c7de", "#cfd9c4", "#e6d3b3"]          # muted agent hues for step boxes
RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
      "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
      "mathtext.fontset": "cm", "font.size": 7}

T, T_STAR, T_BASE = 6, 3, 5          # steps, gold step, base-score argmax (downstream)
BASE = np.array([0.30, 0.42, 0.62, 0.55, 0.80, 0.70])           # S(s_t), illustrative
FINAL = np.array([0.34, 0.50, 0.98, 0.62, 0.84, 0.70])          # S + gamma*B


def txt(ax, x, y, s, size=5.5, color=INK_2, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def arrow(ax, p, q, color=INK_2, lw=0.8, rad=0.0, alpha=1.0, z=5, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", alpha=alpha, zorder=z,
                                 shrinkA=1.5, shrinkB=1.5))


def panel(ax, x0, x1, title):
    ax.add_patch(FancyBboxPatch((x0, 0.4), x1 - x0, 23.2, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=PANEL, ec=PANEL_EDGE, lw=0.7, zorder=0))
    txt(ax, (x0 + x1) / 2, 22.3, title, size=6.8, color=INK, fontweight="bold")


def bars(ax, x0, y0, w, h, vals, color, mark_i, mark_color, mark):
    n = len(vals); bw = w / n
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=AXIS, lw=0.5, zorder=2))
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + i * bw + bw * 0.2, y0), bw * 0.6, h * v, fc=color, lw=0, zorder=3))
        txt(ax, x0 + (i + 0.5) * bw, y0 - 0.85, str(i + 1), size=5.4, color=MUTED)
    ax.plot([x0 + (mark_i + 0.5) * bw], [y0 + h * vals[mark_i] + 0.55], marker=mark, ms=4.4,
            mfc=mark_color, mec="white", mew=0.7, ls="none", zorder=6)
    gx = x0 + (T_STAR - 0.5) * bw
    ax.plot([gx, gx], [y0, y0 + h + 0.5], color=INK_2, ls=(0, (3, 2)), lw=0.6, zorder=4)
    txt(ax, gx - 0.8, y0 + h + 0.15, r"$t^\star$", size=5.8, color=INK, ha="right")


def draw():
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(5.5, 2.4), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 55); ax.set_ylim(0, 24)
        ax.set_aspect("equal"); ax.axis("off")

        # ================================================================ stage 1
        panel(ax, 0.4, 16.7, "(1) Encode with a frozen proxy")
        sx, sw, bh, gap = 1.3, 4.3, 2.35, 2.85
        for t in range(T):
            y = 17.0 - t * gap
            gold = t + 1 == T_STAR
            ax.add_patch(FancyBboxPatch((sx, y), sw, bh, boxstyle="round,pad=0.15", fc=AGENTS[t % 3],
                                        ec=ORANGE_DARK if gold else AXIS, lw=1.0 if gold else 0.5, zorder=3))
            txt(ax, sx + sw / 2, y + bh / 2, rf"$s_{t+1}$", size=6.4, color=INK)
        txt(ax, sx + sw / 2, 20.5, r"failed trajectory $\tau$", size=6)
        txt(ax, sx + sw + 0.45, 17.0 - (T_STAR - 1) * gap + bh / 2, r"$t^\star$", size=6, color=ORANGE_DARK, ha="left")

        px = 8.5
        ax.add_patch(Polygon([[px, 3.6], [px, 18.2], [px + 2.8, 15.6], [px + 2.8, 6.2]], closed=True,
                             fc=PURPLE, ec=PURPLE_DARK, lw=0.7, zorder=3))
        txt(ax, px + 1.4, 11.6, r"$\mathcal{M}$", size=9, color="white")
        txt(ax, px + 1.4, 9.4, "frozen", size=5.2, color="white")
        txt(ax, px + 1.4, 19.6, "proxy", size=6)
        arrow(ax, (sx + sw + 1.5, 10.9), (px - 0.2, 10.9))

        rx, ry, cw = 12.9, 12.2, 0.7
        for t in range(T):
            for k in range(4):
                ax.add_patch(Rectangle((rx + k * cw, ry + (T - 1 - t) * 1.05), cw * 0.85, 0.9, fc=PURPLE,
                                       alpha=0.35 + 0.16 * ((t * 3 + k) % 4), lw=0, zorder=3))
        txt(ax, rx + 1.4, 19.6, r"step repr. $v_t$", size=6)
        arrow(ax, (px + 3.0, 13.6), (rx - 0.3, 15.2))
        mx, my, cs = 12.9, 4.3, 0.7
        for t in range(T):
            for i in range(t):
                strong = (t, i) in {(3, 2), (4, 2), (5, 2), (5, 4)}
                ax.add_patch(Rectangle((mx + i * cs, my + (T - 1 - t) * cs), cs * 0.85, cs * 0.85, fc=ORANGE,
                                       alpha=0.95 if strong else 0.3, lw=0, zorder=3))
        txt(ax, mx + 2.1, 2.2, r"attention $m_{i,t}$", size=6)
        arrow(ax, (px + 3.0, 8.2), (mx - 0.3, 7.0))

        # ================================================================ stage 2
        panel(ax, 18.2, 36.2, "(2) Spectral base score")
        cx, cy = 19.2, 13.6
        for k in range(3):
            ax.add_patch(FancyBboxPatch((cx + k * 0.45, cy + k * 0.5), 3.6, 5.4, boxstyle="round,pad=0.15",
                                        fc="white", ec=AXIS, lw=0.5, zorder=3 - 0.1 * k))
        for j in range(4):
            ax.add_patch(Rectangle((cx + 1.6, cy + 1.2 + j * 1.0), 2.0, 0.65, fc=AGENTS[j % 3], lw=0, zorder=4))
        txt(ax, cx + 2.2, 20.4, "reference corpus", size=6)
        txt(ax, cx + 2.2, 12.2, "unlabeled failures", size=5.2, color=MUTED)

        vx, vy, vw, vh = 26.2, 13.6, 9.0, 6.0
        sv = np.array([1.0, 0.55, 0.40, 0.31, 0.25, 0.21, 0.18, 0.15, 0.12, 0.10])
        ax.add_patch(Rectangle((vx, vy), vw, vh, fc="white", ec=AXIS, lw=0.5, zorder=2))
        bw = vw / len(sv)
        for i, v in enumerate(sv):
            inband = 1 <= i <= 4
            ax.add_patch(Rectangle((vx + i * bw + bw * 0.18, vy), bw * 0.64, vh * v * 0.92,
                                   fc=ORANGE_DARK if inband else PURPLE, alpha=1 if inband else 0.5, lw=0, zorder=3))
        ax.add_patch(Rectangle((vx + bw, vy - 0.35), 4 * bw, vh + 0.7, fc="none", ec=ORANGE_DARK, lw=0.7,
                               ls=(0, (2, 1.5)), zorder=4))
        txt(ax, vx + vw / 2, 20.4, r"SVD: $R = U\Sigma V^{\top}$", size=6)
        txt(ax, vx + 3 * bw, 12.2, r"spectral band $\mathcal{C}$", size=5.6, color=ORANGE_DARK)
        arrow(ax, (cx + 4.7, 16.6), (vx - 0.3, 16.6))

        txt(ax, 27.2, 9.9, r"$\pi_{\mathcal{C}}(v_t)=\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\langle v_t,\,V_{:,c}\rangle^2$",
            size=6.6, color=INK)
        txt(ax, 27.2, 7.6, r"$S(s_t)=1/(\pi_{\mathcal{C}}(v_t)+\epsilon)$", size=6.6, color=INK)
        bars(ax, 19.4, 1.5, 15.4, 4.0, BASE, PURPLE, T_BASE - 1, PURPLE_DARK, "o")
        txt(ax, 34.9, 6.5, r"argmax lands after $t^\star$", size=5.4, color=PURPLE_DARK, ha="right")

        # ================================================================ stage 3
        panel(ax, 37.7, 54.6, "(3) Attention-guided rescoring")
        xs = [39.8 + i * 2.55 for i in range(T)]; gy = 16.2
        W = {(2, 3): 0.6, (2, 4): 0.75, (2, 5): 0.9, (0, 1): 0.45, (1, 2): 0.4, (3, 4): 0.35, (4, 5): 0.4}
        for (i, t), w in W.items():
            arrow(ax, (xs[t], gy + 0.55), (xs[i], gy + 0.55), color=ORANGE_DARK, lw=0.4 + 1.5 * w,
                  rad=0.32 + 0.09 * (t - i), alpha=0.3 + 0.65 * w, z=3)
        for i, x in enumerate(xs):
            gold = i + 1 == T_STAR
            ax.add_patch(plt.Circle((x, gy), 0.75, fc=ORANGE_DARK if gold else "white",
                                    ec=ORANGE_DARK if gold else AXIS, lw=0.7, zorder=4))
            txt(ax, x, gy, str(i + 1), size=5.4, color="white" if gold else INK)
        txt(ax, (xs[0] + xs[-1]) / 2, 20.6, "later steps route blame back to the", size=5.4, color=MUTED)
        txt(ax, (xs[0] + xs[-1]) / 2, 19.6, "predecessors they attend to", size=5.4, color=MUTED)
        txt(ax, (xs[0] + xs[-1]) / 2, 13.9, r"dependency weights $w_{i,t}$, $\sum_{i<t} w_{i,t}=1$", size=5.8)

        txt(ax, 46.15, 11.2, r"$B_i=\sum_{t>i} w_{i,t}\,S(s_t)\,/\,\sum_{t>i} w_{i,t}$", size=6.6, color=INK)
        txt(ax, 46.15, 8.6, r"$\widetilde S(s_i)=S(s_i)+\gamma\,B_i$", size=7, color=INK)
        bars(ax, 39.0, 1.5, 14.6, 4.0, FINAL, ORANGE, T_STAR - 1, ORANGE_DARK, "^")
        txt(ax, 53.4, 6.5, r"$\hat t=\arg\max_i \widetilde S(s_i)$, $\hat a=a_{\hat t}$", size=5.6, color=ORANGE_DARK, ha="right")

        arrow(ax, (16.9, 11.5), (18.1, 11.5), lw=1.1)
        arrow(ax, (36.4, 11.5), (37.6, 11.5), lw=1.1)

        OUT_PDF.parent.mkdir(parents=True, exist_ok=True); OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(OUT_PDF.relative_to(REPO)); print(OUT_PNG.relative_to(REPO))


if __name__ == "__main__":
    draw()
