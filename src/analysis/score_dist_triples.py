"""Per-step base scores for EVERY seed triple of the Protocol-2 sweep.

``score_dist`` scores one anchor config per (model, subset) -- the window that reached
the manuscript. This scores all 18 sliding windows instead, so every triple can be
inspected and the best-separating one chosen on evidence.

Each triple selects its OWN base config (that is what Protocol 2 varies), so the config
is read per window from ``exp-august/outputs/<ds>/tables/325/triples_selection.tsv``
(rows ``SVD (proj)``) rather than assumed constant. Within a window, all three seeds are
scored under that window's config, each fitting its SVD on its own train split.

Work is grouped by (model, subset, seed) so the representations for a seed are loaded
once and shared by every window that contains it, and the SVD is fitted once per
(pooling, position). Each window's mean test step@1 is checked against the recorded
``step_acc_test``.

Writes ``artifacts/score-dist/<ds>/triples/steps.tsv`` (lean: one row per scored step)
and ``configs.tsv`` (one row per window, carrying the hyperparameters and the check).

    # from v2/
    python -m src.analysis.score_dist_triples --config configs/datasets/ww.yaml
    python -m src.analysis.score_dist_triples --config configs/datasets/ww.yaml --model qwen3.5-9b
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.config import V2_ROOT
from ..common.provenance import RunTimer
from ..metrics import compute_metrics
from ..score.svd import fit_one, score_config, N_COMPONENTS
from ..stores import load_representations, split_files, list_rep_files

ARTIFACTS_ROOT = "artifacts/score-dist"


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def load_windows(cfg, dataset: str) -> pd.DataFrame:
    """One row per (model, subset, seed-window) with that window's base config.

    Default source is the main pipeline's own selection
    (``tables/<tag>/triples_selection.tsv``); ``--set selection_tsv=...`` points at
    another registry (e.g. the archived exp-august one) — the SVD row schema is
    identical."""
    tsv = Path(cfg.get("selection_tsv",
                       paths.tables_root(cfg) / "triples_selection.tsv"))
    assert tsv.exists(), f"no triples selection table for {dataset!r}: {tsv}"
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["row"].astype(str).str.startswith("SVD")]
    df = df[df["position"].notna()]
    assert not df.empty, f"no SVD rows in {tsv}"
    return df


def run(cfg) -> None:
    device = cfg.get("device", "cpu")
    dataset = paths._ds(cfg)
    out_dir = Path(cfg.get("artifacts_root", V2_ROOT / ARTIFACTS_ROOT)) / dataset / "triples"
    windows = load_windows(cfg, dataset)

    with RunTimer(cfg, "analysis") as rec:
        rec.note(mode="triples", n_windows=int(len(windows)), device=device)
        rows: list[dict] = []
        cfg_rows: list[dict] = []

        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                w = windows[(windows.model == model) & (windows.subset == subset)]
                if w.empty:
                    print(f"[triples] {model}/{subset}: no windows — skipped")
                    continue

                # seed -> the windows needing it, so each seed is loaded exactly once
                by_seed: dict[int, list] = defaultdict(list)
                for _, r in w.iterrows():
                    for s in str(r["seeds"]).split(","):
                        by_seed[int(s)].append(r)

                rep_dir = paths.reps_root(cfg) / model / subset
                data_dir = paths.data_root(cfg) / subset
                files = list_rep_files(rep_dir)
                acc: dict[str, dict[int, float]] = defaultdict(dict)

                for seed in sorted(by_seed):
                    items = by_seed[seed]
                    poolings = sorted({r["pooling"] for r in items})
                    positions = sorted({r["position"] for r in items})
                    parts = split_files(files, cfg["splits"], seed)
                    ld = lambda fs: load_representations(      # noqa: E731
                        rep_dir, data_dir, poolings, weight_names=positions,
                        files=fs, device=device)
                    train, test = ld(parts["train"]), ld(parts["test"])
                    fits: dict[tuple, dict] = {}

                    for r in items:
                        key = (r["pooling"], r["position"])
                        if key not in fits:
                            fits[key] = fit_one(train.stores[key].R, N_COMPONENTS)
                        cb, ce = int(r["c_begin"]), int(r["c_end"])
                        scores = score_config(test.stores[key].R, fits[key], r["method"],
                                              cb, ce, _as_bool(r["centered"]),
                                              _as_bool(r["weighted"]))
                        keeper = test.keeper
                        acc[r["seeds"]][seed] = compute_metrics(
                            scores, keeper, ks=[1], direction=r["direction"]
                        )[f"step@1_{r['direction']}"]

                        for start, end in keeper.traj_ranges:
                            seg = scores[start:end]
                            for i, e in enumerate(keeper.index[start:end]):
                                rows.append({
                                    "dataset": dataset, "model": model, "subset": subset,
                                    "triple": r["seeds"], "seed": seed,
                                    "traj_idx": e.traj_idx, "step_idx": e.step_idx,
                                    "n_steps": end - start,
                                    "is_mistake": bool(e.is_mistake),
                                    "score": float(seg[i]),
                                })
                    del train, test, fits

                for _, r in w.iterrows():
                    got = float(np.mean(list(acc[r["seeds"]].values())))
                    want = float(r["step_acc_test"])
                    ok = abs(got - want) < 5e-4
                    cfg_rows.append({
                        "dataset": dataset, "model": model, "subset": subset,
                        "triple": r["seeds"], "pooling": r["pooling"],
                        "method": r["method"], "position": r["position"],
                        "c_begin": int(r["c_begin"]), "c_end": int(r["c_end"]),
                        "centered": _as_bool(r["centered"]),
                        "weighted": _as_bool(r["weighted"]),
                        "direction": r["direction"],
                        "step_acc_recorded": want, "step_acc_reproduced": got,
                        "verified": ok,
                    })
                    if not ok:
                        print(f"    *** MISMATCH {model}/{subset} {r['seeds']}: "
                              f"{got:.4f} vs {want:.4f}")
                n_ok = sum(c["verified"] for c in cfg_rows if c["model"] == model
                           and c["subset"] == subset)
                print(f"[triples] {model}/{subset}: {len(w)} windows, "
                      f"{n_ok}/{len(w)} verified, {len(by_seed)} seeds loaded")

        if cfg.get("dry_run"):
            print(f"[triples] dry run — {len(rows)} rows not written")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "steps.tsv", sep="\t", index=False)
        pd.DataFrame(cfg_rows).to_csv(out_dir / "configs.tsv", sep="\t", index=False)
        rec.add_output(out_dir / "steps.tsv")
        rec.add_output(out_dir / "configs.tsv")
        print(f"  wrote {out_dir}/steps.tsv ({len(rows)} steps) + configs.tsv")


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
