"""Verify v2's extraction stage against the already-extracted safetensors.

Lives OUTSIDE v2/ on purpose: it compares the v2 implementation against artifacts
produced by the legacy pipeline, so it is a migration check, not part of v2 itself.

For each (dataset x model x subset) it re-extracts N trajectories with v2's code and
compares, tensor by tensor, against the existing files:

  * activations — keys must match exactly and every tensor must be bit-identical
    (same code path, same inputs, deterministic forward).
  * attention   — same, for raw_attn / raw_attn_per_head / attn_residual_mass /
    ctx_indices.

Models are loaded ONCE and reused across every dataset/subset to keep this cheap.

    cd /root/dataDisk/home/thanhdo/attribscope
    python verify_v2_extraction.py                      # everything, 3 trajs per subset
    python verify_v2_extraction.py --n 3 --models qwen3.5-9b
    python verify_v2_extraction.py --datasets ww --stages attention
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent
V2 = REPO / "v2"
sys.path.insert(0, str(V2))          # v2 is importable as `src.*`

from src.data import load_dataset                                    # noqa: E402
from src.models import get_adapter                                   # noqa: E402
from src.extract.activations import extract_trajectory as extract_acts   # noqa: E402
from src.extract.attention import (                                  # noqa: E402
    extract_trajectory as extract_attn, _force_sdpa,
)
from src.data.context import build_context                           # noqa: E402
import functools                                                     # noqa: E402

# dataset -> (data dir, legacy outputs dir, subsets)
DATASETS = {
    "ww": ("data/ww", "outputs-ww", ["algorithm-generated", "hand-crafted"]),
    "traceelephant": ("data/traceelephant", "outputs-traceelephant", ["magentic", "captain"]),
    "correct-error": ("data/correct-error", "outputs-correct-error",
                      ["arc", "gaia", "hotpot", "math500", "mmlu_pro", "musique", "wikimqa"]),
}
MODEL_PATHS = {
    "qwen3.5-9b":  "../hub/Qwen/Qwen3.5-9B",
    "deepseek-8b": "../hub/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}
MAX_TOKENS = 8192


def load_st(path: Path) -> dict:
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def compare(new: dict, ref_path: Path) -> tuple[bool, str]:
    """Exact-match comparison of a freshly extracted payload against a stored one."""
    ref = load_st(ref_path)
    if set(new) != set(ref):
        missing, extra = sorted(set(ref) - set(new))[:3], sorted(set(new) - set(ref))[:3]
        return False, f"key mismatch (n_new={len(new)} n_ref={len(ref)}) missing={missing} extra={extra}"
    worst, worst_key = 0.0, None
    for k in ref:
        a, b = new[k].float(), ref[k].float()
        if a.shape != b.shape:
            return False, f"shape {k}: {tuple(a.shape)} vs {tuple(b.shape)}"
        d = (a - b).abs().max().item()
        if d > worst:
            worst, worst_key = d, k
    return worst == 0.0, f"max|d|={worst:.3e}" + (f" @ {worst_key}" if worst else "")


def pick(traj_dir: Path, n: int, ref_dir: Path) -> list:
    """First n trajectories (numeric order) that have a reference file to compare to."""
    trajs = load_dataset(str(traj_dir.parent), subset=traj_dir.name)
    out = []
    for t in trajs:
        if (ref_dir / t.filename.replace(".json", ".safetensors")).exists():
            out.append(t)
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="trajectories per subset")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    ap.add_argument("--stages", nargs="+", default=["activations", "attention"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results, failures = [], []
    for model in args.models:
        path = MODEL_PATHS[model]
        adapter = get_adapter(path)
        print(f"\n{'='*78}\nloading {model} ({path})\n{'='*78}", flush=True)
        mdl, tok = adapter.load(path, torch.bfloat16, {"": args.device})
        _force_sdpa(mdl)                       # needed by the attention extractor
        final_norm = adapter.final_norm(mdl)
        n_layers = adapter.num_layers(mdl)
        blocks = adapter.extract_block_indices(mdl)
        layers = ["embed"] + [f"act/{i}" for i in blocks] + [f"act/{n_layers - 1}_normed"]
        ctx_fn = functools.partial(build_context, template_kwargs=adapter.template_kwargs())

        for ds in args.datasets:
            data_root, legacy, subsets = DATASETS[ds]
            for sub in subsets:
                traj_dir = REPO / data_root / sub
                for stage in args.stages:
                    kind = "activations" if stage == "activations" else "attention"
                    ref_dir = REPO / legacy / kind / model / sub
                    if not ref_dir.exists():
                        print(f"[skip] no reference: {ref_dir}")
                        continue
                    trajs = pick(traj_dir, args.n, ref_dir)
                    if not trajs:
                        print(f"[skip] no comparable trajectories: {ds}/{sub}")
                        continue
                    for t in trajs:
                        ref = ref_dir / t.filename.replace(".json", ".safetensors")
                        if stage == "activations":
                            hidden = extract_acts(t, mdl, tok, MAX_TOKENS, layers,
                                                  "all", ctx_fn, final_norm)
                            new = {f"{s}.{k}": v for s, d in hidden.items() for k, v in d.items()}
                        else:
                            new = extract_attn(t, mdl, tok, MAX_TOKENS, adapter,
                                               query_pool="mean")
                        ok, detail = compare(new, ref)
                        tag = "OK  " if ok else "FAIL"
                        line = f"[{tag}] {model:11s} {ds:13s} {sub:19s} {stage:11s} {t.filename:16s} {detail}"
                        print(line, flush=True)
                        results.append(ok)
                        if not ok:
                            failures.append(line)
        del mdl
        torch.cuda.empty_cache()

    n_ok = sum(results)
    print(f"\n{'='*78}")
    print(f"{n_ok}/{len(results)} comparisons bit-identical")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("ALL EXTRACTION OUTPUTS VERIFIED IDENTICAL")


if __name__ == "__main__":
    main()
