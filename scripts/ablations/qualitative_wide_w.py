"""Qualitative examples for the wide-window cells: rescoring flips wrong to right.

Three TraceElephant cells select a wide window in Table 1 (visible in
fig_sensitivity_appendix): Qwen3.5-9B TE-Cap (w=5), Qwen3.5-9B TE-Mag (w=4),
DeepSeek-8B TE-Cap (w=all). For each, this script re-runs the frozen backprop
config on the frozen triple's test splits (asserting the recorded accuracy) and
records every trajectory the rescoring FLIPS: the base score's argmax is wrong,
the rescored argmax is exactly the gold decisive step.

Outputs under ``artifacts/ablations/qualitative_wide_w/<model>/<subset>/``: one
directory per example (``scores.pdf/png``, ``steps.tsv``, ``meta.json``, the raw
``trajectory.json``), a per-cell ``candidates.tsv``, and a top-level
``MANIFEST.md``. The score figure shows the base score and the rescored score,
each argmax marked, the gold step dashed; no title, horizontal grid only.

    python scripts/ablations/qualitative_wide_w.py [--device cpu] [--only qwen3.5-9b/captain]
    python scripts/ablations/qualitative_wide_w.py --replot   # restyle from saved data
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402
from matplotlib.ticker import MaxNLocator          # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main import config as C                       # noqa: E402
from main.reproduce import ReproContext, reproduce_row  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "ablations" / "qualitative_wide_w"

# The three wide-window cells and the w each one selected (asserted against the
# selection table so this script can never drift from Table 1).
CELLS = [
    ("qwen3.5-9b", "captain", "5"),
    ("qwen3.5-9b", "magentic", "4"),
    ("deepseek-8b", "captain", "all"),
]

# House style (scripts/ablations/plot_figures.py, layer_line): SOAP solid dark purple
# with round markers, base score dotted purple, the reference mark dashed orange.
PURPLE, PURPLE_DARK, ORANGE = "#807EAF", "#4F4D8A", "#E8834F"
COLORS = {"base": PURPLE, "soap": PURPLE_DARK}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    # Matched 2026-08-31 to plot_figures.RC so the panel pairs with fig_datasize
    # at equal rendered scale in the merged manuscript figure.
    "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
}
SUBSET_LABELS = {"captain": "TE-Cap", "magentic": "TE-Mag"}
MODEL_LABELS = {"qwen3.5-9b": "Qwen3.5-9B", "deepseek-8b": "DeepSeek-8B"}
READABLE_STEPS = (6, 30)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=2.5, width=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# ---------------------------------------------------------------------- scoring
def anchor_row(cfg, model: str, subset: str, expect_w: str) -> dict:
    sel = pd.read_csv(C.select_dir(cfg) / "selection.tsv", sep="\t")
    r = sel[(sel.model == model) & (sel.subset == subset) & (sel.row == "backprop")]
    assert len(r) == 1, f"no backprop row for {model}/{subset}"
    row = r.iloc[0].to_dict()
    assert str(row["w"]) == expect_w, (
        f"{model}/{subset}: selected w={row['w']!r}, expected {expect_w!r} — "
        "the selection changed; update CELLS.")
    return row


# ------------------------------------------------------------------------- plot
def example_fig(steps: pd.DataFrame, cand: dict, out_dir: Path) -> None:
    x = steps["step_idx"].to_numpy()
    base = steps["base"].to_numpy(float)
    soap = steps["final"].to_numpy(float)
    with plt.rc_context(RC):
        # Narrow: one step per ~0.2in so the trace is not stretched across empty
        # space; the merged figure's minipage widths (0.41 / 0.34) keep this panel's
        # rendered height equal to fig_datasize's (2.0 x 1.5 in).
        fig, ax = plt.subplots(figsize=(max(2.4, min(4.2, 1.0 + 0.2 * len(x))), 1.5),
                               facecolor=SURFACE)
        ax.plot(x, base, color=COLORS["base"], ls=":", lw=1.3, zorder=2, label="base score")
        ax.plot(x, soap, color=COLORS["soap"], lw=1.5, marker="o", ms=3.2, mew=0, zorder=3,
                label="SOAP")
        lo = float(min(base.min(), soap.min()))
        hi = float(max(base.max(), soap.max()))
        # The bottom buffer holds the legend line and the t* flag. The triangle
        # sits ON the x-axis (clip_on lets it straddle the spine); the legend takes
        # the bottom corner AWAY from t*, and the t* label rides high enough above
        # the floor to clear the legend's text band even where their x-ranges meet.
        pad_lo, pad_hi = 0.30 * (hi - lo), 0.16 * (hi - lo)
        y0 = lo - pad_lo
        # Argmax of each score: a large dark-edged marker, labelled in place — SOAP's
        # above the marker, the base score's below, so the two never collide.
        for pred, y, colour, shape, name, dy in (
                (cand["base_pred_step"], base, COLORS["base"], "s", "base argmax", -1),
                (cand["soap_pred_step"], soap, COLORS["soap"], "o", "SOAP argmax", +1)):
            i = int(np.flatnonzero(x == pred)[0])
            ax.plot([pred], [y[i]], marker=shape, ms=7, mfc=colour, mec=INK,
                    mew=0.9, ls="none", zorder=5)
            left = (pred - x.min()) > (x.max() - pred)
            ax.annotate(name, xy=(pred, y[i]), xytext=(-3 if left else 3, 6 * dy),
                        textcoords="offset points", ha="right" if left else "left",
                        va="bottom" if dy > 0 else "top", fontsize=6.5, color=INK_2,
                        zorder=6, bbox={"boxstyle": "round,pad=0.12", "fc": SURFACE,
                                        "ec": "none", "alpha": 0.8})
        # Decisive step: an orange marker on the axis floor, its name beside it.
        gold = cand["true_step"]
        ax.plot([gold], [y0], marker="^", ms=6, color=ORANGE, mec=ORANGE,
                clip_on=False, ls="none", zorder=5)
        ax.annotate(r"decisive $t^\star$", xy=(gold, y0), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color=ORANGE, zorder=6,
                    bbox={"boxstyle": "round,pad=0.12", "fc": SURFACE, "ec": "none", "alpha": 0.8})
        med = float(np.median(np.concatenate([base, soap])))
        if med > 0 and hi / med > 50:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.yaxis.get_offset_text().set(fontsize=6.5, color=MUTED)
            ax.set_ylim(lo - pad_lo, hi + pad_hi)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        _style(ax)
        ax.set_xlabel("step index", color=INK_2, labelpad=1)
        ax.set_ylabel("attribution score", color=INK_2, labelpad=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        handles = [
            Line2D([], [], color=COLORS["soap"], lw=1.5, marker="o", ms=3.2, mew=0, label="SOAP"),
            Line2D([], [], color=COLORS["base"], ls=":", lw=1.3, label="base score"),
        ]
        # One horizontal legend line in the buffer, in the corner away from t*.
        side = "lower left" if gold >= (x.min() + x.max()) / 2 else "lower right"
        ax.legend(handles=handles, frameon=False, loc=side, ncol=2,
                  handlelength=1.3, columnspacing=0.8, borderaxespad=0.1,
                  handletextpad=0.4, labelcolor=INK_2)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"scores.{ext}", dpi=220, bbox_inches="tight",
                        facecolor=SURFACE)
        plt.close(fig)


# ------------------------------------------------------------------------ driver
def run_cell(cfg, model: str, subset: str, expect_w: str, device: str) -> pd.DataFrame:
    row = anchor_row(cfg, model, subset, expect_w)
    # Load ONLY the anchor position's activations: the bundle loader defaults to every
    # stored position (~30x the I/O), and this script scores exactly one.
    import main.reproduce as MR
    from main.stores import load_representations as _load_all
    MR.load_representations = lambda *a, **k: _load_all(
        *a, **{**k, "weight_names": [row["position"]]})
    ctx = ReproContext.from_config(cfg, model, subset)
    ctx.device = device
    seeds = [int(s) for s in str(row["seeds"]).split(",")]
    label = f"{MODEL_LABELS[model]} {SUBSET_LABELS[subset]}"
    cell_dir = OUT_ROOT / model / subset
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)

    flips, accs = [], []
    for seed in seeds:
        r = reproduce_row(ctx, {**row, "seed": seed}, split="test")
        accs.append(r.metrics["step@1"])
        # Base argmax per trajectory (base is already "higher = error"; ties break
        # toward the earliest step, the metrics' convention).
        base_argmax = {}
        for (start, end) in r.keeper.traj_ranges:
            seg = r.base[start:end].tolist()
            best = min(range(len(seg)), key=lambda i: (-seg[i], i))
            base_argmax[int(r.keeper.index[start].traj_idx)] = \
                int(r.keeper.index[start + best].step_idx)

        for p in r.predictions.itertuples(index=False):
            gold = p.true_step
            if gold is None or pd.isna(gold) or not p.step_correct:
                continue
            if base_argmax[int(p.traj_idx)] == gold:
                continue                       # base already right: not a flip
            flips.append({
                "model": model, "subset": subset, "seed": seed,
                "traj_idx": int(p.traj_idx), "n_steps": int(p.n_steps),
                "true_step": int(gold), "true_agent": p.true_agent,
                "base_pred_step": base_argmax[int(p.traj_idx)],
                "soap_pred_step": int(p.pred_step),
                "soap_pred_agent": p.pred_agent,
                "w": expect_w,
                "_repro": r,
            })

    got, want = sum(accs) / len(accs), float(row["step_acc_test"])
    assert abs(got - want) < 1e-9, f"{label}: reproduced {got} != recorded {want}"
    print(f"[ok] {label} w={expect_w}: step acc {100*got:.2f} verified; "
          f"{len(flips)} flips")

    df = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")}
                       for c in flips])
    if df.empty:
        df.to_csv(cell_dir / "candidates.tsv", sep="\t", index=False)
        return df
    # Rank: legible plots first (readable step count, gold step not at the very
    # start), then smaller trajectories.
    lo, hi = READABLE_STEPS
    df["_readable"] = (~df["n_steps"].between(lo, hi)).astype(int)
    df["_prefix"] = df["true_step"].clip(upper=2).rsub(2)
    df = df.sort_values(by=["_readable", "_prefix", "n_steps", "seed", "traj_idx"])
    # One example per trajectory: the same story under another seed adds nothing.
    seeds_per_traj = df.groupby("traj_idx")["seed"].apply(
        lambda s: ",".join(str(v) for v in sorted(set(s))))
    df["flip_seeds"] = df["traj_idx"].map(seeds_per_traj)
    df = df.drop_duplicates(subset="traj_idx", keep="first").reset_index(drop=True)
    df.insert(0, "pick_rank", df.index + 1)
    df.drop(columns=["_readable", "_prefix"]).to_csv(
        cell_dir / "candidates.tsv", sep="\t", index=False)

    by_key = {(c["seed"], c["traj_idx"]): c for c in flips}
    for _, cand in df.iterrows():
        c = by_key[(int(cand["seed"]), int(cand["traj_idx"]))]
        r = c["_repro"]
        ex_dir = cell_dir / f"traj-{cand['traj_idx']}_seed-{cand['seed']}"
        ex_dir.mkdir(exist_ok=True)
        steps = r.per_step[r.per_step["traj_idx"] == cand["traj_idx"]][
            ["model", "subset", "seed", "traj_idx", "n_steps", "step_idx", "role",
             "base", "final", "rank"]].copy()
        steps["is_gold"] = steps["step_idx"] == cand["true_step"]
        steps.to_csv(ex_dir / "steps.tsv", sep="\t", index=False)

        src_json = C.data_root(cfg) / subset / f"{cand['traj_idx']}.json"
        shutil.copyfile(src_json, ex_dir / "trajectory.json")
        raw = json.loads(src_json.read_text())
        meta = {
            "cell": {"model": model, "subset": subset, "label": label,
                     "frozen_seeds": seeds},
            "frozen_config": dict(r.config),
            "selection": {k: (int(cand[k]) if isinstance(cand[k], np.integer)
                              else cand[k])
                          for k in ("pick_rank", "seed", "traj_idx", "n_steps",
                                    "flip_seeds")},
            "gold": {"step": int(cand["true_step"]), "agent": cand["true_agent"],
                     "mistake_reason": raw.get("mistake_reason"),
                     "question_ID": raw.get("question_ID")},
            "predictions": {"base": int(cand["base_pred_step"]),
                            "soap": int(cand["soap_pred_step"])},
            "verified_step_acc_test": got,
        }
        (ex_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        example_fig(steps, {**{k: int(cand[k]) for k in
                                ("true_step", "base_pred_step", "soap_pred_step")},
                            "w": expect_w}, ex_dir)
    return df


def write_manifest(results: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Qualitative examples: cells that select a WIDE context window",
        "",
        "Trajectories the rescoring FLIPS — the base score's argmax is wrong, the",
        "rescored argmax is the gold decisive step — for the three Table-1 cells",
        "whose selected w is large, on the frozen triples' test splits; recorded",
        "accuracies re-verified before selection. `flip seeds` lists every seed of",
        "the triple on which this trajectory flips. Generated by",
        "scripts/ablations/qualitative_wide_w.py; per-cell details in",
        "`<model>/<subset>/candidates.tsv`, one directory per example.",
        "",
        "| cell | seed | traj | steps | gold | base pred | SOAP pred | flip seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, df in results.items():
        for _, r in df.iterrows():
            lines.append(
                f"| {label} | {r.seed} | {r.traj_idx} | {r.n_steps} | {r.true_step} "
                f"| {r.base_pred_step} | {r.soap_pred_step} | {r.flip_seeds} |")
    (OUT_ROOT / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def replot() -> None:
    """Redraw every example's scores.pdf/png from its saved steps.tsv + meta.json —
    no rescoring, so style changes are cheap."""
    n = 0
    for meta_path in sorted(OUT_ROOT.glob("*/*/traj-*/meta.json")):
        ex_dir = meta_path.parent
        m = json.loads(meta_path.read_text())
        steps = pd.read_csv(ex_dir / "steps.tsv", sep="\t")
        example_fig(steps, {"true_step": m["gold"]["step"],
                            "base_pred_step": m["predictions"]["base"],
                            "soap_pred_step": m["predictions"]["soap"],
                            "w": m["frozen_config"]["w"]}, ex_dir)
        n += 1
    print("replotted", n, "examples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", action="append",
                    help="restrict to model/subset, e.g. qwen3.5-9b/captain")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures from saved steps.tsv, no rescoring")
    a = ap.parse_args()
    if a.replot:
        replot()
        return
    cfg = C.load_config(REPO / "configs-main/traceelephant.yaml")
    cfg["device"] = a.device
    results = {}
    for model, subset, w in CELLS:
        if a.only and f"{model}/{subset}" not in a.only:
            continue
        df = run_cell(cfg, model, subset, w, a.device)
        results[f"{MODEL_LABELS[model]} {SUBSET_LABELS[subset]} (w={w})"] = df
    write_manifest(results)
    print(f"wrote {OUT_ROOT.relative_to(REPO)}/MANIFEST.md")


if __name__ == "__main__":
    main()
