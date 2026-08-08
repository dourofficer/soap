"""w='all' parity check: succ sweeps vs exp-august's frozen backprop sweeps.

At w='all' the successor-side masking is the identity, so every succ sweep row must
reproduce the corresponding ``strategy == backprop, w == all`` row of exp-august's
``sweep_triples.tsv`` EXACTLY (same base tables, same code path). This proves the
patched plumbing computes SOAP where the two formulations coincide.

    # from repo root, after run_rescore_triples.py
    python exp-soap/check_parity.py --config exp-soap/configs/ww.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(EXP))

import pandas as pd

from run_rescore_triples import exp_out, aug_out
from src.common import paths
from src.common.cli import base_parser, load_and_narrow

KEY = ["seed", "pooling", "position", "method", "c_begin", "c_end", "centered",
       "weighted", "orient", "score_norm", "layer_range", "gamma"]
METRICS = ["undisc_step_acc_val", "undisc_agent_acc_val", "undisc_step_acc_test",
           "undisc_agent_acc_test", "disc_step_acc_val", "disc_agent_acc_val",
           "disc_step_acc_test", "disc_agent_acc_test"]
TOL = 1e-9


def _wall(df):
    return df[df["w"].astype(str) == "all"]


def main() -> None:
    cfg = load_and_narrow(base_parser(__doc__).parse_args())
    bad = 0
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            aug_f = aug_out(cfg) / "rescore" / paths.tag(cfg) / model / subset / "sweep_triples.tsv"
            if not aug_f.exists():
                print(f"[skip] {model}/{subset}: no exp-august sweep")
                continue
            aug = pd.read_csv(aug_f, sep="\t")
            ref = _wall(aug[aug["strategy"] == "backprop"])
            f = (exp_out(cfg) / "rescore" / paths.tag(cfg) / model / subset
                 / "sweep_triples-succ.tsv")
            if not f.exists():
                print(f"[skip] {model}/{subset}: no succ sweep")
                continue
            succ = pd.read_csv(f, sep="\t")
            for strat in cfg["strategies"]:
                got = _wall(succ[succ["strategy"] == strat])
                # ref covers the full orient/score_norm grid; the succ sweep may be
                # trimmed to the protocol's fixed values — every succ row must find
                # exactly one reference row, not vice versa.
                m = ref.merge(got, on=KEY, suffixes=("_ref", "_got"))
                if len(got) == 0 or len(m) != len(got):
                    print(f"[FAIL] {model}/{subset}/{strat}: join {len(m)} rows "
                          f"(ref {len(ref)}, got {len(got)})")
                    bad += 1
                    continue
                worst = max((m[f"{c}_ref"] - m[f"{c}_got"]).abs().max() for c in METRICS)
                status = "ok" if worst <= TOL else "FAIL"
                bad += status == "FAIL"
                print(f"[{status}] {model}/{subset}/{strat}: {len(m)} w=all rows, "
                      f"max |diff| = {worst:.2e}")
    if bad:
        sys.exit(f"{bad} parity failure(s)")
    print("parity: all matched")


if __name__ == "__main__":
    main()
