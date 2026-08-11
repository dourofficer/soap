"""Transcript-card figures illustrating DOWNSTREAM CONTAMINATION (manuscript intro).

One early wrong action corrupts the context every later step is generated from, so the
steps that follow it are themselves confused -- repeated recovery attempts, the same
dead end revisited, a scroll loop that never terminates. This module draws that as an
abbreviated transcript: a red decisive-error card followed by its consequences, with a
"corrupted context" rail running down the side, sized to drop straight into a
``wrapfigure`` at ``0.48\\linewidth``.

It is a QUALITATIVE figure. The card text is hand-curated (abbreviated verbatim quotes
from the trajectory JSON, with the CJK page titles paraphrased in English); only the
per-step SCORES are computed. Curation lives in ``EXAMPLES`` below -- edit the prose
there, not in the drawing code. ``**double asterisks**`` mark an inline bold span, used
on the clause that says why the decisive step is the error.

The scores are NOT hardcoded and NOT re-tuned for the figure: the frozen config is read
from the manuscript's own bookkeeping (the same ``table1_main_selection.tsv`` that
:mod:`src.analysis.qualitative` reads), and the recorded base/SOAP accuracies are
re-derived over the manuscript seed window and asserted before anything is drawn. The
scored trajectories are then obtained in ``split='all'`` apply mode, so an example need
not have fallen in that seed's test partition. Scores rank WITHIN a trajectory, never
across, so the bar behind each number is normalised to the largest score DISPLAYED in
that figure.

Which score is printed is one switch, ``SCORE_SOURCE`` below (or ``--score-source``):

    final      the rescored SOAP score -- the default
    oriented   the base score in the comparable "higher = more anomalous" convention
    base       the raw scorer output (``proj`` is "LOWER = more anomalous"; the column
               header says so, and the bars are inverted to stay readable)
    none       draw no scores at all

    # from the repo root; writes manuscript/assets/ + caches the scores
    python -m src.analysis.contamination_figure --config configs/datasets/ww.yaml \\
        --model qwen3.5-9b --subset hand-crafted

    # redraw only (instant, no GPU): reuses artifacts/contamination_figure/scores.json
    python -m src.analysis.contamination_figure --config configs/datasets/ww.yaml \\
        --model qwen3.5-9b --subset hand-crafted --score-source oriented

    # recompute the scores, or draw one example, or send the PDFs elsewhere
    python -m src.analysis.contamination_figure --config configs/datasets/ww.yaml \\
        --model qwen3.5-9b --subset hand-crafted --force --only 21 --out-dir /tmp/fig
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch     # noqa: E402

from ..common import paths                                         # noqa: E402
from ..common.cli import base_parser, load_and_narrow              # noqa: E402

DEFAULT_SELECTION_TSV = "outputs/manuscript-tables/table1_main_selection.tsv"
DEFAULT_OUT_DIR = "manuscript/assets"
CACHE = Path("artifacts/contamination_figure/scores.json")

# Which score the cards print. See the module docstring; --score-source overrides.
SCORE_SOURCE = "final"
SCORE_SOURCES = {
    # column in the reproduction's per-step frame -> (column header, higher = worse?)
    "final":    ("SOAP score", True),
    "oriented": ("base score", True),
    "base":     ("base score\n(lower = worse)", False),
}
SHOW_SCORE_BARS = True     # faint proportional bar behind each number

# ── curated content ─────────────────────────────────────────────────────────
# Keyed by trajectory index in data/ww/hand-crafted/. `elide_after` names the step
# after which the "steps a-b: ..." skip row is drawn.
EXAMPLES = {
    1: dict(
        task="Task: search and return martial-arts classes within a five-minute walk "
             "of the New York Stock Exchange, 7–9 pm.",
        elide_after=16,
        elide="steps 17–23: re-instructions; more failed clicks",
        cards=[
            dict(step=12, role="WebSurfer", error=True, tag="decisive error", lead=None,
                 body="I clicked 'NY Jidokwan Taekwondo'. Screenshot: KEYENCE "
                      "industrial microscopes — **an ad page, unrelated to the task**."),
            dict(step=16, role="WebSurfer", error=False, tag="recovery attempt",
                 lead="Orchestrator: Return to the list of martial arts schools.",
                 body="I clicked the browser back button."),
            dict(step=24, role="WebSurfer", error=False, tag="derailed again",
                 lead="Orchestrator: Focus on addresses and class schedules.",
                 body="I clicked the control. Screenshot: the same irrelevant "
                      "KEYENCE site as before."),
            dict(step=28, role="WebSurfer", error=False,
                 tag="final retry; task unresolved", lead=None,
                 body="I clicked the browser back button."),
        ],
    ),
    21: dict(
        task="Task: search and return the NASA award number supporting R. G. Arendt, "
             "from the paper linked in a June 6, 2023 Universe Today article.",
        elide_after=12,
        elide="steps 13–23: two more page-scrolls, 18% then 27%",
        cards=[
            dict(step=4, role="WebSurfer", error=True, tag="decisive error",
                 lead="Orchestrator: Search for the article, then open the paper "
                      "linked at its bottom.",
                 body="I typed 'Carolyn Collins Petersen … site:universetoday.com'. "
                      "Annotated cause: **never reaches the paper's acknowledgment "
                      "section**."),
            dict(step=8, role="WebSurfer", error=False,
                 tag="the article, not the paper", lead=None,
                 body="I clicked 'There Are Hundreds of Mysterious Filaments…'. "
                      "The viewport shows 9% of the article."),
            dict(step=12, role="WebSurfer", error=False, tag="hunting for the link",
                 lead="Orchestrator: Scroll through the article to find the paper.",
                 body="I scrolled down one page in the browser. Now 9% down."),
            dict(step=24, role="WebSurfer", error=False,
                 tag="still scrolling; unresolved", lead=None,
                 body="I scrolled down one page in the browser. Now 37% down."),
        ],
    ),
}

# ── geometry (inches) ───────────────────────────────────────────────────────
W = 2.64            # 0.48 * 5.5in ICLR text width -> embeds at 1:1
RAIL_X = 0.105      # x of the dashed "corrupted context" rail
CARD_L = 0.26
SCORE_W = 0.40      # right-hand score column
PAD_X = 0.070       # card interior horizontal padding
PAD_TOP, PAD_BOT = 0.052, 0.055
GAP = 0.055         # vertical gap between cards
ELIDE_H = 0.130

FS_HEAD, FS_BODY, FS_SMALL, FS_SCORE = 6.4, 6.2, 5.7, 6.3
LH_BODY, LH_SMALL = 0.086, 0.079

# ── ink (house palette, manuscript/main.tex) ────────────────────────────────
RED, RED_FILL, RED_EDGE = "#990000", "#fbf0f0", "#d9a6a6"
INK, MUTED = "#1a1a1a", "#6f6f6f"
CARD_FILL, CARD_EDGE = "#f6f6f6", "#d2d2d2"
BAR, BAR_RED = "#dcdcdc", "#e8c4c4"

RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}


# ── scores: the manuscript's frozen config, verified, applied to every traj ──
def _compute_scores(cfg: dict, model: str, subset: str, tsv: Path) -> dict:
    """Per-step scores for every trajectory, under the manuscript's frozen config.

    Imported lazily: drawing from the cache must not require torch or a GPU.
    """
    from .qualitative import _load_cells, _verify_cell, _base_view
    from ..reproduce.core import ReproContext, reproduce_row

    cells = _load_cells(cfg, tsv)
    cell = next((c for c in cells if c["model"] == model and c["subset"] == subset), None)
    if cell is None:
        raise SystemExit(f"no cell in {tsv} for model={model!r} subset={subset!r}")

    sub_cfg = dict(cfg, poolings=[cell["hp"]["pooling"]])
    ctx = ReproContext(sub_cfg, model, subset, n_ranges=int(cfg.get("n_ranges", 4)))
    seeds = [int(s) for s in cell["seeds"]]

    # Gate first: the SAME config must re-derive both recorded main-table numbers.
    verification = _verify_cell(cell, {s: reproduce_row(ctx, {**cell["hp"], "seed": s},
                                                        split="test") for s in seeds})
    print(f"[contam] {model}/{cell['column']}: base "
          f"{verification['reproduced_base_step_acc']:.4f} (rec {cell['recorded_svd']}), "
          f"soap {verification['reproduced_soap_step_acc']:.4f} "
          f"(rec {cell['recorded_soap']}) — verified")

    out = {}
    for seed in seeds:
        r = reproduce_row(ctx, {**cell["hp"], "seed": seed}, split="all")
        _, base_ranks = _base_view(r)
        per_traj = {}
        for start, end in r.keeper.traj_ranges:
            traj = int(r.keeper.index[start].traj_idx)
            df = r.per_step[r.per_step["traj_idx"] == traj].reset_index(drop=True)
            per_traj[str(traj)] = {
                str(int(row.step_idx)): {
                    "base": float(row.base), "oriented": float(row.oriented),
                    "final": float(row.final), "final_rank": int(row["rank"]),
                    "base_rank": int(base_ranks[start + i]),
                    "is_mistake": bool(row.is_mistake), "role": str(row.role),
                    "n_steps": int(row.n_steps),
                }
                for i, row in df.iterrows()
            }
        out[str(seed)] = per_traj
        ctx._seed_cache.pop((seed, "all"), None)
        ctx._seed_cache.pop((seed, "test"), None)

    return {"dataset": paths._ds(cfg), "model": model, "subset": subset,
            "column": cell["column"], "frozen_config": cell["hp"],
            "manuscript_seeds": seeds, "selection_tsv": str(tsv),
            "verification": verification, "scores": out}


def _load_scores(cfg, model, subset, tsv: Path, cache: Path, force: bool) -> dict:
    if cache.exists() and not force:
        blob = json.loads(cache.read_text())
        if (blob.get("model"), blob.get("subset")) == (model, subset):
            print(f"[contam] scores from cache {cache} "
                  f"(--force to recompute; seeds {blob['manuscript_seeds']})")
            return blob
    blob = _compute_scores(cfg, model, subset, tsv)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(blob, indent=2, default=str))
    print(f"[contam] wrote {cache}")
    return blob


# ── inline-bold text: parse, measure, wrap, draw ────────────────────────────
def _rich_words(text: str) -> list[tuple[str, bool, bool]]:
    """``'a **b c**. d'`` -> ``[(word, bold, glue), ...]``.

    ``glue`` marks a token that abutted the previous one in the source with no
    whitespace between them -- the trailing '.' of ``**...**.`` -- so it is drawn
    without a leading space. Without it every bold span would be followed by a
    floating space before its punctuation.
    """
    out: list[tuple[str, bool, bool]] = []
    prev_trailing_space = True
    for i, chunk in enumerate(text.split("**")):
        parts = chunk.split()
        if not parts:
            prev_trailing_space = prev_trailing_space or chunk[-1:].isspace()
            continue
        for j, word in enumerate(parts):
            glue = (j == 0 and bool(out) and not prev_trailing_space
                    and not chunk[:1].isspace())
            out.append((word, i % 2 == 1, glue))
        prev_trailing_space = chunk[-1:].isspace()
    return out


def _measure(fig, text: str, fontsize: float, **kw) -> float:
    """Width of ``text`` in inches, as this figure will actually render it."""
    t = fig.text(0, 0, text, fontsize=fontsize, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def _space_w(fig, fontsize: float, **kw) -> float:
    return _measure(fig, "m m", fontsize, **kw) - 2 * _measure(fig, "m", fontsize, **kw)


def _wrap_rich(fig, text: str, max_in: float, fontsize: float, **kw) -> list[list]:
    """Greedy wrap of a ``**bold**`` string into lines of (word, bold, glue, width)."""
    space = _space_w(fig, fontsize, **kw)
    lines, cur, cur_w = [], [], 0.0
    for word, bold, glue in _rich_words(text):
        ww = _measure(fig, word, fontsize, fontweight="bold" if bold else "normal", **kw)
        advance = ww if (not cur or glue) else space + ww
        # A glued token stays on its line even if it overruns: it is punctuation
        # trailing a bold span, and orphaning it onto the next line looks broken.
        if cur and not glue and cur_w + advance > max_in:
            lines.append(cur)
            cur, cur_w = [(word, bold, False, ww)], ww
        else:
            cur.append((word, bold, glue, ww))
            cur_w += advance
    if cur:
        lines.append(cur)
    return lines


def _draw_rich(ax, x: float, y: float, line: list, fontsize: float, color: str,
               space: float, **kw) -> None:
    for i, (word, bold, glue, ww) in enumerate(line):
        if i and not glue:
            x += space
        ax.text(x, y, word, fontsize=fontsize, color=color, va="top", ha="left",
                fontweight="bold" if bold else "normal", zorder=3, **kw)
        x += ww


# ── score formatting ────────────────────────────────────────────────────────
def _scale(values: list[float]) -> tuple[float, str]:
    """Power-of-1000 multiplier putting the largest value in [1, 1000), plus its unit.

    The printed number is ``value * mult``, so the header carries the RECIPROCAL
    exponent: scores of ~0.004 print as ~4 under a "(x 10^-3)" header.
    """
    top = max((abs(v) for v in values), default=1.0)
    exp = 0
    if top > 0:
        while top < 1.0:
            top, exp = top * 1000, exp + 3
        while top >= 1000.0:
            top, exp = top / 1000, exp - 3
    return 10.0 ** exp, ("" if exp == 0 else f"($\\times 10^{{{-exp}}}$)")


def _fmt(v: float) -> str:
    a = abs(v)
    return f"{v:.2f}" if a < 100 else f"{v:.0f}"


# ── the figure ──────────────────────────────────────────────────────────────
def draw(example: dict, scores: dict | None, source: str, out_pdf: Path,
         preview: Path | None = None) -> None:
    """Render one trajectory's transcript cards to ``out_pdf``."""
    cards = example["cards"]
    show_scores = source != "none" and scores is not None
    header, higher_worse = SCORE_SOURCES.get(source, ("score", True))

    vals, mult, unit = [], 1.0, ""
    if show_scores:
        vals = [scores[str(c["step"])][source] for c in cards]
        mult, unit = _scale(vals)
        vals = [v * mult for v in vals]

    with plt.rc_context(RC):
        card_r = W - 0.04 - (SCORE_W if show_scores else 0.0)
        text_w = (card_r - CARD_L) - 2 * PAD_X

        # Pass 1: wrap everything, total the height it needs.
        fig = plt.figure(figsize=(W, 1.0), dpi=300)
        task_lines = _wrap_rich(fig, example["task"], card_r - CARD_L, FS_SMALL,
                                style="italic")
        elide_lines = _wrap_rich(fig, example["elide"], text_w - 0.14, FS_SMALL,
                                 style="italic")
        for c in cards:
            c["_lead"] = (_wrap_rich(fig, c["lead"], text_w, FS_SMALL, style="italic")
                          if c.get("lead") else [])
            c["_body"] = _wrap_rich(fig, c["body"], text_w, FS_BODY)
            c["_h"] = (PAD_TOP + 0.088
                       + len(c["_lead"]) * LH_SMALL + (0.020 if c["_lead"] else 0.0)
                       + len(c["_body"]) * LH_BODY + PAD_BOT)
        sp_body = _space_w(fig, FS_BODY)
        sp_small = _space_w(fig, FS_SMALL, style="italic")
        plt.close(fig)

        head_h = 0.055 + len(task_lines) * LH_SMALL + (0.15 if show_scores else 0.10)
        H = (head_h + sum(c["_h"] for c in cards) + GAP * (len(cards) - 1)
             + ELIDE_H + 0.045)

        # Pass 2: draw, in inch coordinates.
        fig = plt.figure(figsize=(W, H), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")

        y = H - 0.055
        for ln in task_lines:
            _draw_rich(ax, CARD_L, y, ln, FS_SMALL, MUTED, sp_small, style="italic")
            y -= LH_SMALL
        if show_scores:
            ax.text(W - 0.04, y - 0.012, f"{header} {unit}".strip(), fontsize=FS_SMALL,
                    color=MUTED, va="top", ha="right", linespacing=1.15)
        y -= 0.15 if show_scores else 0.10

        rail_top = y
        for i, c in enumerate(cards):
            top, bot = y, y - c["_h"]
            err = c["error"]
            ax.add_patch(FancyBboxPatch(
                (CARD_L, bot), card_r - CARD_L, c["_h"],
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=0.85 if err else 0.55,
                edgecolor=RED if err else CARD_EDGE,
                facecolor=RED_FILL if err else CARD_FILL, zorder=2))

            ty = top - PAD_TOP
            ax.text(CARD_L + PAD_X, ty, f"Step {c['step']}  ·  {c['role']}",
                    fontsize=FS_HEAD, color=INK, fontweight="bold",
                    va="top", ha="left", zorder=3)
            # The tag states the card's status in words, so the red fill is never the
            # only thing separating the decisive error from its consequences.
            ax.text(card_r - PAD_X, ty, ("× " if err else "") + c["tag"],
                    fontsize=FS_SMALL, color=RED if err else MUTED,
                    fontweight="bold" if err else "normal",
                    va="top", ha="right", zorder=3)
            ty -= 0.088

            for ln in c["_lead"]:
                _draw_rich(ax, CARD_L + PAD_X, ty, ln, FS_SMALL, MUTED, sp_small,
                           style="italic")
                ty -= LH_SMALL
            if c["_lead"]:
                ty -= 0.020
            for ln in c["_body"]:
                _draw_rich(ax, CARD_L + PAD_X, ty, ln, FS_BODY, INK, sp_body)
                ty -= LH_BODY

            if show_scores:
                sx0, sx1 = card_r + 0.045, W - 0.04
                mid = (top + bot) / 2
                v = vals[i]
                if SHOW_SCORE_BARS:
                    # Bars compare only the steps DRAWN here, which is the frame the
                    # scores live in anyway: they rank within a trajectory, never across.
                    frac = (v / max(vals)) if higher_worse else (min(vals) / v)
                    frac = min(1.0, max(0.0, frac)) if v else 0.0
                    ax.add_patch(FancyBboxPatch(
                        (sx0, mid - 0.058), max(0.012, frac * (sx1 - sx0)), 0.020,
                        boxstyle="round,pad=0,rounding_size=0.008", linewidth=0,
                        facecolor=BAR_RED if err else BAR, zorder=2))
                ax.text(sx1, mid + 0.052, _fmt(v), fontsize=FS_SCORE,
                        color=RED if err else INK, fontweight="bold" if err else "normal",
                        va="center", ha="right", zorder=3)

            if i == 0:
                rail_top = bot          # contamination begins where the error ends
            y = bot - GAP
            if c["step"] == example["elide_after"]:
                ey = y - 0.012
                ax.text(CARD_L + PAD_X + 0.03, ey - 0.012, r"$\vdots$", fontsize=FS_BODY,
                        color="#9a9a9a", va="center", ha="center", zorder=3)
                for j, ln in enumerate(elide_lines):
                    _draw_rich(ax, CARD_L + PAD_X + 0.11, ey + 0.021 - j * LH_SMALL,
                               ln, FS_SMALL, MUTED, sp_small, style="italic")
                y -= ELIDE_H

        rail_bot = y + GAP
        ax.add_patch(FancyArrowPatch(
            (RAIL_X, rail_top - 0.02), (RAIL_X, rail_bot + 0.01),
            arrowstyle="-|>,head_width=1.7,head_length=3.4", mutation_scale=1.0,
            linewidth=0.8, linestyle=(0, (2.2, 1.6)), color=RED,
            shrinkA=0, shrinkB=0, zorder=1))
        ax.text(RAIL_X - 0.075, (rail_top + rail_bot) / 2, "corrupted context",
                fontsize=FS_SMALL, color=RED, rotation=90, va="center", ha="center")

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf")
        if preview is not None:
            preview.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview, format="png", dpi=400)
        plt.close(fig)
    print(f"[contam] {out_pdf}  ({W:.2f} × {H:.2f} in)")


def run(cfg: dict, args) -> None:
    model = cfg["models"][0] if len(cfg["models"]) == 1 else args.model
    subset = cfg["subsets"][0] if len(cfg["subsets"]) == 1 else args.subset
    if not model or not subset:
        raise SystemExit("pass --model and --subset (one cell per run)")

    source = args.score_source
    if source not in {*SCORE_SOURCES, "none"}:
        raise SystemExit(f"--score-source must be one of {[*SCORE_SOURCES, 'none']}")

    blob, seed = None, None
    if source != "none":
        blob = _load_scores(cfg, model, subset, Path(args.selection_tsv),
                            Path(args.cache), args.force)
        seed = str(args.score_seed if args.score_seed is not None
                   else blob["manuscript_seeds"][0])
        if seed not in blob["scores"]:
            raise SystemExit(f"seed {seed} not scored; have {list(blob['scores'])}")
        print(f"[contam] scores: {source!r} from seed {seed}, config "
              f"{blob['frozen_config']}")

    wanted = [int(t) for t in args.only] if args.only else list(EXAMPLES)
    out_dir = Path(args.out_dir)
    for traj in wanted:
        if traj not in EXAMPLES:
            raise SystemExit(f"no curated example for trajectory {traj}; "
                             f"have {list(EXAMPLES)}")
        scores = blob["scores"][seed].get(str(traj)) if blob else None
        if blob and scores is None:
            raise SystemExit(f"trajectory {traj} not in the scored subset")
        stem = f"contamination_{subset.replace('-', '')}_traj{traj}"
        draw(EXAMPLES[traj], scores, source, out_dir / f"{stem}.pdf",
             preview=(Path(args.preview_dir) / f"{stem}.png") if args.preview_dir else None)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--score-source", default=SCORE_SOURCE,
                   help=f"one of {[*SCORE_SOURCES, 'none']} (default {SCORE_SOURCE})")
    p.add_argument("--score-seed", type=int, default=None,
                   help="which manuscript-window seed's scores to print "
                        "(default: the first)")
    p.add_argument("--only", nargs="*", default=None,
                   help="draw only these trajectory indices")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--preview-dir", default=None, help="also write PNG previews here")
    p.add_argument("--selection-tsv", default=DEFAULT_SELECTION_TSV)
    p.add_argument("--cache", default=str(CACHE))
    args = p.parse_args()
    run(load_and_narrow(args), args)


if __name__ == "__main__":
    main()
