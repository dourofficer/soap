"""Extract pooled hidden states for the SYNTHETIC fit corpus (gaia + assistantbench only).

Same forward-pass / pooling / output schema as ``src.extract.activations`` — it reuses
``extract_trajectory`` verbatim — but over a FILTERED file list (the fit pool) and driven
by ``src/xfit/config.yaml`` instead of a dataset manifest. Output mirrors the core layout
so ``src.stores.load_representations`` reads it unchanged::

    outputs/synthetic/activations/<proxy>/<source>/*.safetensors

The proxy model is loaded ONCE and reused across all of its sources (loading weights is
the dominant fixed cost). Trajectories whose output exists are skipped (resumable).

    # from repo root
    python -m src.xfit.extract                       # all proxies x all sources
    python -m src.xfit.extract --proxy qwen3.5-9b --source magentic-qwen9b --gpu 0
"""
from __future__ import annotations

import argparse
import functools
import json
import time

import torch
from safetensors.torch import save_file

from ..data import load_dataset, build_context
from ..data.trajectory import extract_metadata
from ..extract.activations import extract_trajectory
from ..models import get_adapter
from .common import load_config, kept_files, synth_data_dir, synth_reps_dir, iter_sources


def _load_proxy(model_path: str, dtype, device: str):
    device_map = "auto" if device == "auto" else {"": device}
    adapter = get_adapter(model_path)
    model, tokenizer = adapter.load(model_path, dtype, device_map)
    n_layers = adapter.num_layers(model)
    blocks = adapter.extract_block_indices(model)
    layers = ["embed"] + [f"act/{i}" for i in blocks] + [f"act/{n_layers - 1}_normed"]
    final_norm = adapter.final_norm(model)
    context_fn = functools.partial(build_context, template_kwargs=adapter.template_kwargs())
    return model, tokenizer, layers, final_norm, context_fn


def extract_source(proxy, model, tokenizer, layers, final_norm, context_fn,
                   source, pools, max_tokens, force=False, dry=False) -> int:
    """Extract one synthetic source's fit-pool trajectories for one proxy. Returns #written."""
    keep = set(kept_files(source, pools))
    trajs = [t for t in load_dataset(str(synth_data_dir(source).parent), subset=source)
             if t.filename in keep]
    out_dir = synth_reps_dir(proxy, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"proxy": proxy, "source": source, "pools": pools, "layers": layers,
         "pool": "all", "max_tokens": max_tokens, "n_trajs": len(trajs)}, indent=2))
    print(f"[extract] {proxy}/{source}: {len(trajs)} trajs (pools={pools})")
    if dry:
        return 0

    written = 0
    t0 = time.perf_counter()
    for traj in trajs:
        out_path = out_dir / traj.filename.replace(".json", ".safetensors")
        if out_path.exists() and not force:
            continue
        hidden = extract_trajectory(traj, model, tokenizer, max_tokens, layers,
                                    "all", context_fn, final_norm)
        flat = {f"{s}.{k}": v.contiguous() for s, d in hidden.items() for k, v in d.items()}
        assert flat, f"no hidden states for {source}/{traj.filename}"
        save_file(flat, out_path, metadata={"payload_metadata": json.dumps(extract_metadata(traj))})
        written += 1
    print(f"  {proxy}/{source}: wrote {written} in {time.perf_counter() - t0:.1f}s")
    return written


def run(cfg: dict, only_proxy: str | None = None, only_source: str | None = None,
        device: str = "cuda", dtype=torch.bfloat16, force=False, dry=False) -> None:
    pools = cfg["pools"]
    max_tokens = cfg["max_tokens"]
    proxies = [only_proxy] if only_proxy else list(cfg["proxies"])
    # (harness, gen, source) -> just the source names, optionally narrowed.
    sources = [s for _, _, s in iter_sources(cfg) if only_source in (None, s)]

    for proxy in proxies:
        model_path = cfg["model_paths"][proxy]
        if dry:
            model = tokenizer = layers = final_norm = context_fn = None
            for source in sources:
                keep = set(kept_files(source, pools))
                trajs = [t for t in load_dataset(str(synth_data_dir(source).parent), subset=source)
                         if t.filename in keep]
                print(f"[dry] {proxy}/{source}: {len(trajs)} trajs")
            continue
        print(f"== loading proxy {proxy} ({model_path}) ==")
        model, tokenizer, layers, final_norm, context_fn = _load_proxy(model_path, dtype, device)
        for source in sources:
            extract_source(proxy, model, tokenizer, layers, final_norm, context_fn,
                           source, pools, max_tokens, force=force, dry=dry)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proxy", default=None, help="Restrict to one proxy model.")
    p.add_argument("--source", default=None, help="Restrict to one synthetic source.")
    p.add_argument("--gpu", default="0", help="CUDA device index (or 'auto'/'cpu').")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()

    cfg = load_config(args.overrides)
    device = args.gpu if args.gpu in ("auto", "cpu") else f"cuda:{args.gpu}"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    run(cfg, only_proxy=args.proxy, only_source=args.source, device=device,
        dtype=dtype, force=args.force, dry=args.dry_run)


if __name__ == "__main__":
    main()
