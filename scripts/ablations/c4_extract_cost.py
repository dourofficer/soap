#!/usr/bin/env python
"""C4 — SOAP's extraction cost as run, per trajectory: time, tokens read, peak GPU memory.

The extractor (`main/extract.py`) encodes one step per forward pass and runs two passes
per step: the activation stage and the attention stage, each over the same input, the
step in its context under the 8,192-token budget. This script runs exactly those two
stages through the extractor's own functions, without writing artifacts, and records
per trajectory:

  * `passes_act`, `passes_attn`  — forward passes of each stage
  * `tokens_act`, `tokens_attn`  — tokens read by each stage (sum of input lengths)
  * `act_s`, `attn_s`, `total_s` — wall-clock seconds, GPU-synchronized
  * `peak_act_gb`, `peak_attn_gb` — peak allocated GPU memory during each stage
  * `peak_reserved_gb`            — peak memory the allocator reserved over both stages
  * `weights_gb`                  — memory held by the loaded model alone
  * `hard_truncated`              — steps whose own content exceeded the budget

Rows are appended as they finish, so a partial run is usable. One GPU, no batching,
the model in bfloat16 — the extractor's own settings.

    python scripts/ablations/c4_extract_cost.py --subset algorithm-generated --subset hand-crafted
    python scripts/ablations/c4_extract_cost.py --subset hand-crafted --limit 5   # smoke test
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
from main.extract import (DTYPES, _force_sdpa, activations_for, attention_for,  # noqa: E402
                          get_adapter)

OUT_DIR = REPO / "results-ablations" / "c4_extract_cost"
FIELDS = ["subset", "traj", "n_steps", "passes_act", "passes_attn", "tokens_act",
          "tokens_attn", "act_s", "attn_s", "total_s", "peak_act_gb", "peak_attn_gb",
          "peak_reserved_gb", "weights_gb", "hard_truncated"]
GB = float(2 ** 30)


def token_counts(traj, tokenizer, max_tokens, tk):
    """Tokens each stage reads, from the extractor's own input builder."""
    act = attn = hard = 0
    for s in iter_scoreable_steps(traj):
        enc = build_step_input(traj, s, tokenizer, max_tokens=max_tokens,
                               template_kwargs=tk, with_gt=False)
        n = int(enc["input_ids"].shape[1])
        hard += int(enc["hard_truncated"])
        if n <= enc["ctx_len"]:
            continue                      # empty step: the activation stage skips it
        act += n
        ctx = [m for m in enc["step_tokens"] if m != s]
        if any(m >= 0 for m in ctx):      # the attention stage needs a real predecessor
            attn += n
    return act, attn, hard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs-main" / "ww.yaml"))
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--subset", action="append", dest="subsets")
    ap.add_argument("--limit", type=int, default=None)
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
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / GB
    print(f"model loaded: {weights_gb:.2f} GB of weights on {torch.cuda.get_device_name(0)}",
          flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        trajs = load_dataset(C.data_root(cfg), subset)
        if args.limit:
            trajs = trajs[: args.limit]
        path = out_dir / f"soap_{args.model}_{subset}.tsv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
            w.writeheader()
            for k, traj in enumerate(trajs, 1):
                steps = iter_scoreable_steps(traj)
                tok_act, tok_attn, hard = token_counts(traj, tokenizer, max_tokens, tk)

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                hidden = activations_for(traj, model, tokenizer, max_tokens, layers,
                                         final_norm, tk, False)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                peak_act = torch.cuda.max_memory_allocated() / GB
                reserved = torch.cuda.max_memory_reserved() / GB
                torch.cuda.reset_peak_memory_stats()
                flat = attention_for(traj, model, tokenizer, max_tokens, tk, with_gt=False)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                peak_attn = torch.cuda.max_memory_allocated() / GB
                reserved = max(reserved, torch.cuda.max_memory_reserved() / GB)

                row = {"subset": subset, "traj": Path(traj.filename).stem,
                       "n_steps": len(steps), "passes_act": len(hidden),
                       "passes_attn": sum(1 for key in flat if key.endswith(".raw_attn")),
                       "tokens_act": tok_act, "tokens_attn": tok_attn,
                       "act_s": round(t1 - t0, 3), "attn_s": round(t2 - t1, 3),
                       "total_s": round(t2 - t0, 3), "peak_act_gb": round(peak_act, 3),
                       "peak_attn_gb": round(peak_attn, 3),
                       "peak_reserved_gb": round(reserved, 3),
                       "weights_gb": round(weights_gb, 3), "hard_truncated": hard}
                w.writerow(row)
                fh.flush()
                del hidden, flat
                print(f"  {subset} {k}/{len(trajs)} {row['traj']}: steps={row['n_steps']} "
                      f"passes={row['passes_act']}+{row['passes_attn']} "
                      f"tokens={tok_act + tok_attn} time={row['total_s']:.1f}s "
                      f"peak={max(peak_act, peak_attn):.2f}GB", flush=True)
        print(f"  wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
