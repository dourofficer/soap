"""Score-stage runner: per (model, subset, seed), BOTH poolings, ONE TSV.

Loads train/val/test reps (both poolings, shared keeper), fits SVD per
(pooling, position) on train, scores the grid, and writes
``outputs/<ds>/scores/<tag>/<model>/<subset>/seed-<n>.tsv`` with ``pooling`` as a
column.

    # from v2/
    python -m src.score.run --config configs/score/correct-full.yaml
    python -m src.score.run --config configs/score/correct-full.yaml \
        --model qwen3.5-9b --seed 1 --set positions=[act/15]
"""
from __future__ import annotations

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..metrics import KeeperContext
from ..stores import load_representations, split_files, list_rep_files
from .scorers import SCORERS
from .svd import score_from_entry, fit_one, N_COMPONENTS
from .ensemble import member_positions, ensemble_rows

OUT_COLS = ["seed", "pooling", "position", "method", "c_begin", "c_end",
            "centered", "weighted", "direction", "k",
            "step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test"]

DEFAULT_METHODS = ["proj", "resid", "angres", "maha", "norm_l2", "norm_l1"]


def _positions(reps, cfg) -> list[str]:
    want = cfg.get("positions", "all")
    have = reps.positions()
    if want == "all" or want is None:
        return have
    missing = [p for p in want if p not in have]
    if missing:
        raise SystemExit(f"positions {missing} not in reps (have {have})")
    return list(want)


def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda")
    ks = cfg["ks"]
    methods = cfg.get("methods", DEFAULT_METHODS)
    weighted = cfg.get("weighted", [False])
    poolings = cfg["poolings"]
    n_comp = cfg.get("n_components", N_COMPONENTS)
    force = cfg.get("force", False)
    dry = cfg.get("dry_run", False)

    with RunTimer(cfg, "scores") as run_rec:
        run_rec.note(methods=methods, weighted=weighted, poolings=poolings,
                     n_components=n_comp, ks=ks)
        for model in cfg["models"]:
            for subset in cfg["subsets"]:
                rep_dir = paths.reps_root(cfg) / model / subset
                data_dir = paths.data_root(cfg) / subset
                files = list_rep_files(rep_dir)
                for seed in cfg["seeds"]:
                    out = paths.scores_root(cfg) / model / subset / f"seed-{seed}.tsv"
                    if out.exists() and not force:
                        print(f"[skip] {out}")
                        continue
                    print(f"[score] {model}/{subset} seed={seed}  ({len(files)} trajs)")
                    if dry:
                        continue

                    parts = split_files(files, cfg["splits"], seed)
                    load = lambda fl: load_representations(
                        rep_dir, data_dir, poolings=poolings, files=fl, device=device)
                    train, val, test = load(parts["train"]), load(parts["val"]), load(parts["test"])
                    val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
                    positions = _positions(val, cfg)

                    # ── The scoring grid ─────────────────────────────────────
                    # Cells are (pooling x position) x (method x band x centered x
                    # weighted). Pooling and position are the OUTER loops for one reason:
                    # the SVD fit is per (pooling, position) and is the only expensive
                    # step here, so it is computed once and then reused by (a) the whole
                    # inner config grid and (b) the layer ensemble below, which needs the
                    # same fits for its member positions.
                    #
                    # Everything inside `score_from_entry` is deliberately NOT a Python
                    # loop over metrics: all config score vectors for a position are
                    # stacked and evaluated in two batched passes (one per direction).
                    # Scoring is a cheap matmul; the metric ranking is what dominates,
                    # and batching it is the difference between minutes and hours.
                    ens_cfg = cfg.get("ensemble", {}) or {}
                    rows = []
                    for pooling in poolings:
                        fits = {}
                        for position in tqdm(positions, desc=f"{pooling}", leave=False):
                            entry = fit_one(train.stores[(pooling, position)].R, n_comp)
                            fits[position] = entry
                            rows += score_from_entry(
                                pooling, position, entry,
                                val.stores[(pooling, position)].R,
                                test.stores[(pooling, position)].R,
                                val_ctx, test_ctx, methods, weighted, ks,
                                n_components=n_comp, device=None)
                        if ens_cfg.get("enabled"):
                            members = member_positions(positions)
                            rows += ensemble_rows(
                                pooling, members, fits,
                                {p: train.stores[(pooling, p)].R for p in members},
                                {p: val.stores[(pooling, p)].R for p in members},
                                {p: test.stores[(pooling, p)].R for p in members},
                                val_ctx, test_ctx, methods, weighted, ks, n_components=n_comp)
                    for r in rows:
                        r["seed"] = seed
                    df = pd.DataFrame(rows)[OUT_COLS].sort_values(
                        "step_acc_test", ascending=False, kind="mergesort")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, sep="\t", index=False)
                    run_rec.add_output(out)
                    print(f"  wrote {out}  ({len(df)} rows)")
                    del train, val, test
                    if device == "cuda":
                        torch.cuda.empty_cache()


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
