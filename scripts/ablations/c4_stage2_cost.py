#!/usr/bin/env python
"""C4 — SOAP's cost AFTER extraction: the SVD fit (setup) and scoring + rescoring (inference).

Everything here runs on stored representations and attention masses, so it is what a
practitioner pays once the backbone has been run. Per subset, on the first seed of the
frozen triple and the configuration selected for Table 1:

  setup      fit_svd on the reference split's step matrix          -> seconds
  inference  score_steps on every test step (band projection)      -> ms per trajectory
             build the dependency matrices and apply the correction -> ms per trajectory
             (aggregate_attn, which reads the attention files of the whole subset once,
              is reported apart, as a per-subset load)

Measured on CPU and on one GPU. Also writes the reference-split file stems per subset,
so the extraction rows of c4_extract_cost.py can be summed into SOAP's setup cost.

CPU timings pin the torch intra-op thread pool (`--cpu-threads`, default 4), the same
setting as ../attrib-prompting/scripts/cost_rb_stage2.py; see its docstring for why the
lazy default is not comparable.

    python scripts/ablations/c4_stage2_cost.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main import config as C                                            # noqa: E402
from main.rescore import WCache, aggregate_attn, apply_strategy         # noqa: E402
from main.score import fit_svd, score_steps                             # noqa: E402
from main.stores import list_rep_files, load_representations, split_files  # noqa: E402

OUT_DIR = REPO / "results-ablations" / "c4_extract_cost"
POOLING = "mean"
# The configuration selected for Table 1 (tab:anchors / results-nogt/ww/select/selection.tsv).
SELECTED = {
    "algorithm-generated": {"position": "act/27", "c_begin": 1, "c_end": 7, "range_idx": 0,
                            "gamma": 0.6, "w": 1},
    "hand-crafted":        {"position": "act/31", "c_begin": 0, "c_end": 5, "range_idx": 2,
                            "gamma": 0.1, "w": 2},
}
FIELDS = ["subset", "device", "seed", "n_ref_traj", "n_ref_steps", "n_test_traj", "n_test_steps",
          "svd_fit_s", "score_ms_per_traj", "attn_load_s_subset", "rescore_ms_per_traj",
          "cpu_threads"]


def clock(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def measure(cfg, model, subset, device, repeats=5):
    sel = SELECTED[subset]
    seed = cfg["seeds"][subset][0]
    rep_dir = C.reps_root(cfg) / model / subset
    data_dir = Path(cfg["data_dir"]) / subset
    files = list_rep_files(rep_dir)
    parts = split_files(files, cfg["splits"], seed)
    train = load_representations(rep_dir, data_dir, poolings=[POOLING],
                                 weight_names=[sel["position"]], device=device, files=parts["train"])
    test = load_representations(rep_dir, data_dir, poolings=[POOLING],
                                weight_names=[sel["position"]], device=device, files=parts["test"])
    R_train = train.stores[(POOLING, sel["position"])].R.to(device)
    R_test = test.stores[(POOLING, sel["position"])].R.to(device)
    n_test = len(parts["test"])

    # setup: the SVD of the reference matrix
    t0 = clock(device)
    for _ in range(repeats):
        V = fit_svd(R_train, cfg["n_components"])
    svd_s = (clock(device) - t0) / repeats

    # inference, stage 2a: the base score of every test step
    t0 = clock(device)
    for _ in range(repeats):
        s_test = score_steps(R_test, V, sel["c_begin"], sel["c_end"])
    score_ms = (clock(device) - t0) / repeats / n_test * 1e3

    # inference, stage 2b: dependency matrices + the correction
    t0 = clock(device)
    weightings, _ = aggregate_attn(C.attn_root(cfg), model, subset, device=device)
    attn_load_s = clock(device) - t0
    t0 = clock(device)
    for _ in range(repeats):
        WC = WCache([weightings[sel["range_idx"]]], test.keeper, [sel["w"]], device=device)
        mats = WC.mats(0, sel["w"])
        S = apply_strategy(s_test, test.keeper, mats, "backprop", [sel["gamma"]])
    rescore_ms = (clock(device) - t0) / repeats / n_test * 1e3
    assert S.shape[0] == R_test.shape[0]

    return {"subset": subset, "device": device, "seed": seed,
            "n_ref_traj": len(parts["train"]), "n_ref_steps": int(R_train.shape[0]),
            "n_test_traj": n_test, "n_test_steps": int(R_test.shape[0]),
            "svd_fit_s": round(svd_s, 4), "score_ms_per_traj": round(score_ms, 4),
            "attn_load_s_subset": round(attn_load_s, 3),
            "rescore_ms_per_traj": round(rescore_ms, 3)}, [Path(f).stem for f in parts["train"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs-main" / "ww.yaml"))
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--cpu-threads", type=int, default=4,
                    help="torch intra-op threads for the CPU timings (see the docstring)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, ref_stems = [], {}
    for subset in cfg["subsets"]:
        for device in ("cpu", "cuda"):
            row, stems = measure(cfg, args.model, subset, device)
            row["cpu_threads"] = args.cpu_threads
            rows.append(row)
            ref_stems[subset] = stems
            print(row, flush=True)
    with (out_dir / f"soap_stage2_{args.model}.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"soap_reference_stems_{args.model}.json").write_text(json.dumps(ref_stems, indent=1))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
