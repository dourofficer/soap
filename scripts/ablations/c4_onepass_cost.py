#!/usr/bin/env python
"""C4 — SOAP's extraction under the one-pass convention: the pass over the whole trajectory.

Attention is causal, so one forward pass over a trajectory that fits the 8,192-token
budget yields every step's hidden states and attention mass at once; the per-step
extractor (`main/extract.py`) re-reads the shared prefix instead. The manuscript's cost
appendix (2026-09-17) charges SOAP that ONE pass. Its time is the activation pass of the
LAST scoreable step, whose input is the whole trajectory in its context, timed through
the extractor's own `extract_hidden` on one GPU, several repeats after a warm-up.

Per trajectory: the last step's input length, the pass time (mean over repeats), and
the peak GPU memory. A trajectory longer than the budget is truncated to its last
8,192 tokens as in extraction, so its "one pass" is one budget-sized pass.

    python scripts/ablations/c4_onepass_cost.py --subset algorithm-generated --subset hand-crafted
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from main import config as C                                              # noqa: E402
from main.data import build_step_input, iter_scoreable_steps, load_dataset  # noqa: E402
from main.extract import (DTYPES, _force_sdpa, extract_hidden, get_adapter)  # noqa: E402

OUT_DIR = REPO / "results-ablations" / "c4_extract_cost"
FIELDS = ["subset", "traj", "n_steps", "last_step", "tokens", "pass_s", "peak_gb", "repeats"]
GB = float(2 ** 30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs-main" / "ww.yaml"))
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--subset", action="append", dest="subsets")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    subsets = args.subsets or cfg["subsets"]
    max_tokens = cfg["max_tokens"]
    model_path = str(REPO / cfg["model_paths"][args.model])
    adapter = get_adapter(model_path)
    model, tokenizer = adapter.load(model_path, DTYPES[cfg.get("dtype", "bfloat16")], {"": "cuda"})
    _force_sdpa(model)
    tk = adapter.template_kwargs()
    n_layers = adapter.num_layers(model)
    layers = ["embed"] + [f"act/{i}" for i in adapter.extract_block_indices(model)] \
        + [f"act/{n_layers - 1}_normed"]
    final_norm = adapter.final_norm(model)
    device = next(model.parameters()).device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"soap_onepass_{args.model}.tsv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for subset in subsets:
            trajs = load_dataset(C.data_root(cfg), subset)
            for k, traj in enumerate(trajs, 1):
                steps = iter_scoreable_steps(traj)
                last = steps[-1]
                enc = build_step_input(traj, last, tokenizer, max_tokens=max_tokens,
                                       template_kwargs=tk, with_gt=False)
                input_ids = enc["input_ids"].to(device)
                extract_hidden(model, input_ids, enc["ctx_len"], layers, final_norm)  # warm-up
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    extract_hidden(model, input_ids, enc["ctx_len"], layers, final_norm)
                    torch.cuda.synchronize()
                sec = (time.perf_counter() - t0) / args.repeats
                row = {"subset": subset, "traj": Path(traj.filename).stem, "n_steps": len(steps),
                       "last_step": last, "tokens": int(input_ids.shape[1]),
                       "pass_s": round(sec, 4),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
                       "repeats": args.repeats}
                w.writerow(row)
                fh.flush()
                print(f"  {subset} {k}/{len(trajs)} {row['traj']}: tokens={row['tokens']} "
                      f"pass={sec:.3f}s peak={row['peak_gb']:.2f}GB", flush=True)
    print(f"wrote {path}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
