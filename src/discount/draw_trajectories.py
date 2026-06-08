"""Per-trajectory score visualization for any row of the reduced table.

For one (model, subset):
  1. Read outputs/tables/discounted/reduced/{model}__{subset}.tsv and select
     the row at --row-index. Works for both SVD and classifier rows; the
     strategy column determines which reproduction path is taken.
  2. For SVD rows: reproduce val + test SVD scores and orient via
     row['svd_orient']. For classifier rows: reproduce val + test classifier
     scores; no orientation needed.
  3. Apply the discount pass with row['gamma'], row['w'] using the attention
     weighting at row['layer_range'].
  4. For each trajectory in the test split, save a two-panel figure:
       left  = undiscounted scores
       right = discounted scores
     True mistake step in red; mistake step and argmax step labelled with
     their agent role. All other steps are unannotated blue dots.

Filename layout: {prefix}_{traj_idx}_{a}{b}.png where
  prefix : 'svd' or 'classifier'
  a      : 'T'/'F' — undiscounted argmax matches the true mistake step
  b      : 'T'/'F' — discounted argmax matches the true mistake step
  ('N'    if the trajectory has no recorded mistake step)

usage:
    python -m attribscope.discount.draw_trajectories \
        --model qwen3-8b --subset hand-crafted --row-index 0 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attribscope.discount.discount import apply_discount, orient_svd_scores
from attribscope.discount.reproduce import reproduce_classifier, reproduce_svd
from attribscope.discount.weights import aggregate_attn

MODELS  = ["deepseek-8b", "llama-3.1-8b", "qwen3-8b", "qwen3-14b"]
SUBSETS = ["algorithm-generated", "hand-crafted"]

REDUCED_ROOT_DEFAULT = Path("outputs/tables/discounted/reduced")
ATTN_ROOT_DEFAULT    = Path("outputs/weighting_attn")
REPS_ROOT_DEFAULT    = Path("outputs/representation-full")
DATA_ROOT_DEFAULT    = Path("data/ww")
OUT_ROOT_DEFAULT     = Path("outputs/gallery")

CLF_STRATEGIES = ("classifier_pseudo", "classifier_oracle")


# ── selection & reproduction ─────────────────────────────────────────────────

def _select_row(reduced_path: Path, row_index: int) -> pd.Series:
    if not reduced_path.exists():
        raise FileNotFoundError(f"reduced table not found: {reduced_path}")
    df = pd.read_csv(reduced_path, sep="\t")
    if not 0 <= row_index < len(df):
        raise IndexError(
            f"row_index {row_index} out of range [0, {len(df)}) for {reduced_path}"
        )
    return df.iloc[row_index]


def _reproduce(row, model, subset, reps_root, data_root, device):
    """Returns (bundle, test_undiscounted_oriented_or_raw, filename_prefix)."""
    strategy = row["strategy"]
    if strategy == "svd":
        bundle = reproduce_svd(row, model, subset, reps_root, data_root, device)
        test_undisc = orient_svd_scores(
            bundle.test_scores, strategy=row["svd_orient"],
        ).cpu()
        return bundle, test_undisc, "svd"
    if strategy in CLF_STRATEGIES:
        bundle = reproduce_classifier(row, model, subset, reps_root, data_root, device)
        return bundle, bundle.test_scores.cpu(), "classifier"
    raise ValueError(f"unknown strategy: {strategy!r}")


def _coerce_w(v):
    """TSV read-back may give str. Normalise to "all" or int."""
    if isinstance(v, str):
        return "all" if v == "all" else int(v)
    return v


def _pred_flag(scores_traj: np.ndarray, step_indices: list[int],
               mistake_step: int | None) -> str:
    if mistake_step is None or mistake_step not in step_indices:
        return "N"
    pred_step = step_indices[int(np.argmax(scores_traj))]
    return "T" if pred_step == mistake_step else "F"


# ── plotting ─────────────────────────────────────────────────────────────────

def _draw_panel(ax, step_indices, scores, roles, mistake_step, title):
    x = np.asarray(step_indices)
    y = np.asarray(scores)
    ax.plot(x, y, color="gray", linewidth=1, zorder=1)
    ax.plot(x, y, "o", color="steelblue", markersize=7, zorder=2)

    annotate_idx = {int(np.argmax(y))}
    if mistake_step is not None and mistake_step in step_indices:
        mi = step_indices.index(mistake_step)
        ax.plot([x[mi]], [y[mi]], "o", color="red", markersize=8, zorder=3)
        annotate_idx.add(mi)

    for i in annotate_idx:
        ax.annotate(
            roles[i], xy=(x[i], y[i]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_title(title, fontsize=10)
    ax.margins(x=0.02)


def _orient_display(row: pd.Series) -> str:
    v = row.get("svd_orient", "")
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    return str(v)


def _config_text(model: str, subset: str, row: pd.Series) -> str:
    threshold = row.get("threshold", "")
    if isinstance(threshold, float) and pd.isna(threshold):
        threshold = ""
    line1 = (
        f"model: {model}  |  subset: {subset}  |  strategy: {row['strategy']}  |  "
        f"weight: {row['weight']}  pooling: {row['pooling']}  "
        f"method: {row['method']}  c: [{row['c_begin']}, {row['c_end']}]  "
        f"centered: {row['centered']}  threshold: {threshold}  seed: {int(row['seed'])}"
    )
    line2 = (
        f"discount  →  orient: {_orient_display(row)}  |  "
        f"layer_range: {row['layer_range']}  |  "
        f"gamma: {row['gamma']}  |  w: {row['w']}"
    )
    line3 = (
        f"undisc test step/agent: {row['undisc_step_acc_test']:.3f}/"
        f"{row['undisc_agent_acc_test']:.3f}   "
        f"→  disc: {row['disc_step_acc_test']:.3f}/"
        f"{row['disc_agent_acc_test']:.3f}   "
        f"(diff: {row['diff_step_acc_test']:+.3f}/"
        f"{row['diff_agent_acc_test']:+.3f})"
    )
    return "\n".join([line1, line2, line3])


def _save_one(
    traj_idx, step_indices, roles, mistake_step,
    undisc_y, disc_y, config_text, out_path,
):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 4.5))
    _draw_panel(ax_l, step_indices, undisc_y, roles, mistake_step,
                "Undiscounted")
    _draw_panel(ax_r, step_indices, disc_y,   roles, mistake_step,
                "Discounted")

    fig.suptitle(f"Trajectory {traj_idx}", fontsize=11)
    fig.text(0.5, 0.02, config_text, ha="center", va="bottom",
             fontsize=8, family="monospace")
    fig.subplots_adjust(bottom=0.28, top=0.90, left=0.05, right=0.98,
                        wspace=0.13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── main pipeline ────────────────────────────────────────────────────────────

def run(
    model: str, subset: str, row_index: int,
    reduced_root: Path, attn_root: Path, reps_root: Path, data_root: Path,
    out_root: Path, n_ranges: int, device: str,
) -> None:
    row = _select_row(reduced_root / f"{model}__{subset}.tsv", row_index)
    bundle, test_undisc, prefix = _reproduce(
        row, model, subset, reps_root, data_root, device,
    )

    weightings, bounds = aggregate_attn(
        attn_root, model, subset, n_ranges=n_ranges, device="cpu",
    )
    labels = [f"{lo}-{hi}" for (lo, hi) in bounds]
    if row["layer_range"] not in labels:
        raise RuntimeError(
            f"layer_range {row['layer_range']!r} not in {labels}; "
            f"--n-ranges {n_ranges} likely mismatches the sweep."
        )
    weighting = weightings[labels.index(row["layer_range"])]

    test_disc = apply_discount(
        test_undisc, bundle.test_keeper, weighting,
        gamma=float(row["gamma"]), w=_coerce_w(row["w"]),
    )

    out_dir = out_root / f"{model}__{subset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_text = _config_text(model, subset, row)
    keeper = bundle.test_keeper

    n_written = 0
    for start, end in keeper.traj_ranges:
        entries = list(keeper.index[start:end])
        if not entries:
            continue
        traj_idx     = entries[0].traj_idx
        step_indices = [e.step_idx for e in entries]
        roles        = [e.role     for e in entries]
        mistake_step = next((e.step_idx for e in entries if e.is_mistake), None)

        undisc_y = test_undisc[start:end].cpu().numpy()
        disc_y   = test_disc[start:end].cpu().numpy()

        a = _pred_flag(undisc_y, step_indices, mistake_step)
        b = _pred_flag(disc_y,   step_indices, mistake_step)
        out_path = out_dir / f"{prefix}_{traj_idx}_{a}{b}.png"

        _save_one(traj_idx, step_indices, roles, mistake_step,
                  undisc_y, disc_y, config_text, out_path)
        n_written += 1

    print(f"wrote {n_written} figures to {out_dir} (prefix={prefix!r}, row={row_index})")
    summary = {
        k: row[k] for k in
        ["strategy", "weight", "pooling", "method", "c_begin", "c_end",
         "centered", "threshold", "seed", "svd_orient", "layer_range",
         "gamma", "w", "diff_step_acc_test"]
    }
    print(f"selected row: {summary}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",     required=True, choices=MODELS)
    ap.add_argument("--subset",    required=True, choices=SUBSETS)
    ap.add_argument("--row-index", required=True, type=int,
                    help="0-based row index into the reduced TSV.")
    ap.add_argument("--reduced-root", type=Path, default=REDUCED_ROOT_DEFAULT)
    ap.add_argument("--attn-root",    type=Path, default=ATTN_ROOT_DEFAULT)
    ap.add_argument("--reps-root",    type=Path, default=REPS_ROOT_DEFAULT)
    ap.add_argument("--data-root",    type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--out-root",     type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--n-ranges",     type=int,  default=4,
                    help="Must match n_ranges used in the discount sweep.")
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    run(
        model=args.model, subset=args.subset, row_index=args.row_index,
        reduced_root=args.reduced_root, attn_root=args.attn_root,
        reps_root=args.reps_root,       data_root=args.data_root,
        out_root=args.out_root,         n_ranges=args.n_ranges,
        device=args.device,
    )


if __name__ == "__main__":
    main()