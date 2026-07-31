"""Paper-setting consistency gate: the Real row must equal the main 325 artifacts.

The Real pseudo-source re-runs the target's OWN train split through the full xfit paper
path (per-seed fit -> grid -> ens-mid3 -> reduce -> sweep -> reduce). If the plumbing is
faithful, every reduced value it produces is byte-equal to the archived main-run tables
— the same fit corpus, the same splits, the same selection universe. This module checks
exactly that, per proxy x paper seed, for every reduced table both paths share:

    base_by_method_{test,val}.tsv          (per method in cfg["methods"])
    base_ext_<scorer>_{test,val}.tsv       (winner per seed)
    crr_ext_<scorer>_{test,val}.tsv        }  winner per seed: config identity +
    backprop_ext_<scorer>_{test,val}.tsv   }  undisc/disc metrics

Equality is EXACT (string-level float compare after parse); a mismatch report includes
the max-abs-diff so a pure floating-point drift (a different torch/CUDA stack than the
archived run) is distinguishable from a real plumbing bug. ``paper.assert_real`` picks
error | warn | off. ``verify_alignment`` additionally recomputes every question-aligned
fit set and diffs it against the JSONs the score stage recorded — it needs no GPU and
can run before any scoring.

    python -m src.xfit.verify --set setting=paper [--dataset ww]
"""
from __future__ import annotations

import json

import pandas as pd

from ..common import paths
from . import align
from .common import (load_config, source_tag, target_cfg, setting, paper_cfg,
                     paper_targets, paper_jobs, paper_seeds, REAL_SOURCE)
from .score import _align_path

BASE_ID = ["pooling", "position", "method", "c_begin", "c_end", "centered", "direction"]
BASE_METRICS = ["step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test"]
SWEEP_ID = BASE_ID + ["orient", "score_norm", "strategy", "layer_range", "gamma", "w"]
SWEEP_METRICS = [f"{p}_{m}" for p in ("undisc", "disc")
                 for m in ("step_acc_val", "agent_acc_val", "step_acc_test", "agent_acc_test")]


def _read(root, proxy, subset, name) -> pd.DataFrame | None:
    p = root / proxy / subset / name
    return pd.read_csv(p, sep="\t") if p.exists() else None


def _diff_rows(real_row, main_row, id_cols, metric_cols, where, mismatches):
    for c in id_cols:
        a, b = str(real_row[c]), str(main_row[c])
        if a != b:
            mismatches.append((where, f"config:{c}", a, b, None))
    for c in metric_cols:
        a, b = float(real_row[c]), float(main_row[c])
        if a != b:
            mismatches.append((where, f"metric:{c}", a, b, abs(a - b)))


def verify_real(cfg, only_dataset=None) -> list:
    """Compare every xfitp-real reduced value against the main 325 tables."""
    pc = paper_cfg(cfg)
    mismatches, compared = [], 0
    for _, dataset, subset in paper_targets(cfg):
        if only_dataset not in (None, dataset):
            continue
        seeds = paper_seeds(cfg, dataset)
        real_cfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(REAL_SOURCE, cfg))
        main_cfg = target_cfg(dataset, subset, cfg)     # the 325 tag
        real_root, main_root = paths.reduced_root(real_cfg), paths.reduced_root(main_cfg)
        for proxy in cfg["proxies"]:
            for conv in ("test", "val"):
                # per-method base winners
                r = _read(real_root, proxy, subset, f"base_by_method_{conv}.tsv")
                m = _read(main_root, proxy, subset, f"base_by_method_{conv}.tsv")
                if r is not None and m is not None:
                    for seed in seeds:
                        for method in cfg["methods"]:
                            rr = r[(r["seed"] == seed) & (r["method"] == method)]
                            mm = m[(m["seed"] == seed) & (m["method"] == method)]
                            if len(rr) == 1 and len(mm) == 1:
                                compared += 1
                                _diff_rows(rr.iloc[0], mm.iloc[0], BASE_ID, BASE_METRICS,
                                           f"{dataset}/{subset}/{proxy} by_method[{method}] "
                                           f"s{seed} {conv}", mismatches)
                # per-scorer winners: base + both strategies
                for scorer in cfg["methods"]:
                    for stem, idc, met in (
                            (f"base_ext_{scorer}", BASE_ID, BASE_METRICS),
                            (f"crr_ext_{scorer}", SWEEP_ID, SWEEP_METRICS),
                            (f"backprop_ext_{scorer}", SWEEP_ID, SWEEP_METRICS)):
                        r = _read(real_root, proxy, subset, f"{stem}_{conv}.tsv")
                        m = _read(main_root, proxy, subset, f"{stem}_{conv}.tsv")
                        if r is None or m is None:
                            continue
                        for seed in seeds:
                            rr, mm = r[r["seed"] == seed], m[m["seed"] == seed]
                            if len(rr) == 1 and len(mm) == 1:
                                compared += 1
                                _diff_rows(rr.iloc[0], mm.iloc[0], idc, met,
                                           f"{dataset}/{subset}/{proxy} {stem} s{seed} {conv}",
                                           mismatches)
    print(f"[verify] Real row: {compared} winner rows compared, "
          f"{len(mismatches)} mismatching fields")
    for where, field, a, b, d in mismatches[:40]:
        extra = f" (|diff|={d:.3e})" if d is not None else ""
        print(f"  MISMATCH {where}: {field} real={a} main={b}{extra}")
    diffs = [d for *_x, d in mismatches if d is not None]
    if diffs:
        print(f"  max metric |diff| = {max(diffs):.3e}")
    if compared == 0:
        print("  (nothing compared — run the paper score/rescore stages first)")
    return mismatches


def verify_alignment(cfg, only_dataset=None) -> list:
    """Recompute every question-aligned fit set and diff vs the recorded _align JSONs."""
    from ..stores import list_rep_files, split_files
    from pathlib import Path
    problems = []
    for source, dataset, subset in paper_jobs(cfg):
        if source == REAL_SOURCE or only_dataset not in (None, dataset):
            continue
        tcfg = target_cfg(dataset, subset, cfg, split_tag=source_tag(source, cfg))
        rep_dir = paths.reps_root(tcfg) / cfg["proxies"][0] / subset
        files = list_rep_files(rep_dir)
        for seed in paper_seeds(cfg, dataset):
            p = _align_path(tcfg, subset, seed)
            if not p.exists():
                continue
            stems = [Path(f).stem for f in split_files(files, tcfg["splits"], seed)["train"]]
            got, _ = align.fit_files(dataset, subset, seed, source, stems, cfg["pools"])
            recorded = json.loads(p.read_text())["files"]
            if got != recorded:
                problems.append((str(p), recorded, got))
                print(f"  ALIGN DRIFT {p}")
    print(f"[verify] alignment: {len(problems)} drifted records")
    return problems


def run(cfg, only_dataset=None) -> None:
    if setting(cfg) != "paper":
        raise SystemExit("verify is a paper-setting stage (--set setting=paper)")
    pc = paper_cfg(cfg)
    problems = verify_alignment(cfg, only_dataset)
    mismatches = verify_real(cfg, only_dataset) if pc["real_row"] else []
    if (mismatches or problems) and pc["assert_real"] == "error":
        raise SystemExit(f"verify FAILED: {len(mismatches)} field mismatches, "
                         f"{len(problems)} alignment drifts (paper.assert_real=error)")
    if mismatches or problems:
        print(f"[verify] WARNING: {len(mismatches)} mismatches / {len(problems)} drifts "
              f"(paper.assert_real={pc['assert_real']})")
    else:
        print("[verify] OK")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    run(load_config(args.overrides), only_dataset=args.dataset)


if __name__ == "__main__":
    main()
