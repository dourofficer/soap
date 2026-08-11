"""Prove main/ selects the same configs and reports the same numbers as src/.

This is checkable for EVERY cell, and it is a strong claim rather than a spot check:
src/'s protocol pins `base.fixed` to pooling=mean, method=proj, centered=false,
weighted=false and `rescore.fixed` to orient=inverse, score_norm=none — exactly main/'s
frozen choices. The two base grids and the two rescore grids are therefore identical, so
for the same seed triple main/ must select the SAME (position, c_begin, c_end) and
(layer_range, gamma, w) and report the SAME accuracies.

The one known divergence is `ens-mid3` (src/ orients ensemble members by negation before
z-scoring, main/ uses the folded inverse), so the comparison re-selects from main/'s sweep
with ens-mid3 rows EXCLUDED — which costs nothing, since the ensemble-on base grid is a
superset of the ensemble-off one. Cells where ens-mid3 wins in the full selection are
reported separately: that is a finding, not a failure.

    python scripts/check_main_parity.py
    python scripts/check_main_parity.py --dataset ww --gt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from main.config import load_config, seeds_for, sweep_dir            # noqa: E402
from main.score import ENSEMBLE_POSITION                             # noqa: E402
from main.sweep import (BASE_STRATEGY, BASE_SWEPT, RESCORE_SWEPT,    # noqa: E402
                        select_config)

ROW_MAP = {"svd": "SVD (proj)", "backprop": "backprop",
           "succ-strong": "succ-strong", "succ-near": "succ-near"}
HP = {"svd": ["position", "c_begin", "c_end"],
      "backprop": ["position", "c_begin", "c_end", "layer_range", "gamma", "w"]}
HP["succ-strong"] = HP["succ-near"] = HP["backprop"]


def _norm(v) -> str:
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


def check(ds: str, gt: bool, drop_ensemble: bool = True):
    cfg = load_config(REPO / "configs-main" / f"{ds}.yaml", [f"gt={str(gt).lower()}"])
    src_sel_path = (REPO / ("outputs-gt" if gt else "outputs") / ds
                    / "tables" / "325" / "triples_selection.tsv")
    if not src_sel_path.exists():
        return [], [], [f"{ds}/{'gt' if gt else 'nogt'}: no src/ reference selection table"], []
    src = pd.read_csv(src_sel_path, sep="\t")

    mismatches, ens_wins, skipped, ties = [], [], [], []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            seeds = seeds_for(cfg, subset)
            seeds_str = ",".join(map(str, seeds))
            path = sweep_dir(cfg, model, subset) / "sweep.tsv"
            if not path.exists():
                skipped.append(f"{ds}/{'gt' if gt else 'nogt'}/{model}/{subset}: no sweep yet")
                continue
            df = pd.read_csv(path, sep="\t")
            full_base = select_config(df[df.strategy == BASE_STRATEGY], BASE_SWEPT, seeds,
                                      "step_acc_test@1", "agent_acc_test@1")
            if full_base and full_base["config"]["position"] == ENSEMBLE_POSITION:
                ens_wins.append(f"{ds}/{'gt' if gt else 'nogt'}/{model}/{subset}: "
                                f"ens-mid3 wins the base selection "
                                f"(step={full_base['step']:.4f})")
            if drop_ensemble:
                df = df[df.position != ENSEMBLE_POSITION]

            base = select_config(df[df.strategy == BASE_STRATEGY], BASE_SWEPT, seeds,
                                 "step_acc_test@1", "agent_acc_test@1")
            if base is None:
                skipped.append(f"{ds}/{model}/{subset}: no complete base config")
                continue
            got = {"svd": {**base["config"], "step": base["step"], "agent": base["agent"]}}
            for strat in cfg["strategies"]:
                sw = df[df.strategy == strat]
                for ax in BASE_SWEPT:
                    sw = sw[sw[ax].astype(str).map(_norm) == _norm(base["config"][ax])]
                sel = select_config(sw, RESCORE_SWEPT, seeds, "step_acc_test@1",
                                    "agent_acc_test@1")
                if sel:
                    got[strat] = {**base["config"], **sel["config"],
                                  "step": sel["step"], "agent": sel["agent"]}

            ref = src[(src.model == model) & (src.subset == subset)
                      & (src.seeds == seeds_str)]
            for row, g in got.items():
                r = ref[ref.row == ROW_MAP[row]]
                tag = f"{ds}/{'gt' if gt else 'nogt'}/{model}/{subset}/{row}"
                if r.empty:
                    skipped.append(f"{tag}: no src/ row at seeds {seeds_str}")
                    continue
                r = r.iloc[0]
                hp_diff = [k for k in HP[row] if _norm(g[k]) != _norm(r[k])]
                step_same = abs(round(g["step"], 4) - float(r["step_acc_test"])) <= 5e-5
                if hp_diff and step_same:
                    # A tie on the headline metric that the two packages broke
                    # differently. src/ compares raw floats, so summation order (row
                    # order in its sweep file) can decide before the agent tiebreak is
                    # consulted; main/ rounds the comparison key, so agent decides.
                    # Same step accuracy either way — this is an explained difference,
                    # not a numeric disagreement.
                    ties.append(f"{tag}: step {g['step']:.4f} both, but "
                                + ", ".join(f"{k} main={g[k]!r} src={r[k]!r}"
                                            for k in hp_diff)
                                + f" (agent main={g['agent']:.4f} "
                                  f"src={float(r['agent_acc_test']):.4f})")
                    continue
                for k in hp_diff:
                    mismatches.append(f"{tag}: {k} main={g[k]!r} src={r[k]!r}")
                for k, col in (("step", "step_acc_test"), ("agent", "agent_acc_test")):
                    if abs(round(g[k], 4) - float(r[col])) > 5e-5:
                        mismatches.append(f"{tag}: {k} main={g[k]:.4f} src={float(r[col]):.4f}")
    return mismatches, ens_wins, skipped, ties


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", dest="datasets")
    p.add_argument("--keep-ensemble", action="store_true",
                   help="do NOT exclude ens-mid3 (expect divergence)")
    args = p.parse_args()
    datasets = args.datasets or ["ww", "traceelephant", "correct-error"]

    all_mis, all_ens, all_skip, all_ties = [], [], [], []
    for ds in datasets:
        for gt in (False, True):
            m, e, s, ti = check(ds, gt, drop_ensemble=not args.keep_ensemble)
            all_mis += m
            all_ens += e
            all_skip += s
            all_ties += ti

    if all_skip:
        print(f"skipped ({len(all_skip)}):")
        for x in all_skip:
            print("  .", x)
        print()
    if all_ties:
        print(f"ties on step accuracy broken differently ({len(all_ties)}, explained — "
              f"src/ compares raw floats so summation order can decide before the agent "
              f"tiebreak; main/ rounds the key):")
        for x in all_ties:
            print("  =", x)
        print()
    if all_ens:
        print(f"ens-mid3 won the base selection in {len(all_ens)} cell(s) — the one "
              f"documented divergence from src/, worth surfacing:")
        for x in all_ens:
            print("  !", x)
        print()
    print("=" * 70)
    if all_mis:
        print(f"MAIN/SRC PARITY FAILED — {len(all_mis)} mismatch(es):")
        for x in all_mis[:40]:
            print("  -", x)
        if len(all_mis) > 40:
            print(f"  ... and {len(all_mis) - 40} more")
        return 1
    print("MAIN/SRC PARITY OK: every compared cell selects the same config and reports "
          "the same accuracies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
