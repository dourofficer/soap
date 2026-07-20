"""Extract pooled hidden states along the residual stream . One .safetensors per trajectory with flat keys
``{step}.{pool}.{shorthand}`` and a ``payload_metadata`` header — the exact schema
src.stores.load_representations consumes.

Shorthands: tuple idx 0 = ``embed``; idx k+1 = ``act/k``; final = ``act/{N-1}_normed``.

    # from v2/ (writes into the inputs tree so downstream stages read it directly)
    python -m src.extract.activations \
        --model qwen3.5-9b --model-path ../../hub/Qwen/Qwen3.5-9B \
        --input data/correct-full --subset magentic \
        --output outputs/correct-full/activations/qwen3.5-9b/magentic \
        --pool all --layers all --max_tokens 8192
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from tqdm import tqdm
from safetensors.torch import save_file

from ..data import Trajectory, load_dataset, iter_scoreable_steps, build_context
from ..data.trajectory import extract_metadata
from ..models import get_adapter, find_decoder, num_hidden_layers


# ── shorthand <-> tuple index ───────────────────────────────────────────────
def shorthand_to_layer(shorthand: str) -> int:
    base = shorthand.removesuffix("_normed")
    if base == "embed":
        return 0
    if base.startswith("act/"):
        return int(base[4:]) + 1
    raise ValueError(f"unknown shorthand {shorthand!r}")


def all_shorthands(n_layers: int) -> list[str]:
    return ["embed"] + [f"act/{i}" for i in range(n_layers)] + [f"act/{n_layers - 1}_normed"]


# ── pooling ─────────────────────────────────────────────────────────────────
def pool_last(h: Tensor, ctx_len: int) -> Tensor:
    assert h.shape[0] > ctx_len
    return h[-1].half().cpu()


def pool_mean(h: Tensor, ctx_len: int) -> Tensor:
    assert h.shape[0] > ctx_len
    return h[ctx_len:].float().mean(dim=0).half().cpu()


def _apply_pool(h, ctx_len, pool) -> dict[str, Tensor]:
    if pool == "last":
        return {"last": pool_last(h, ctx_len)}
    if pool == "mean":
        return {"mean": pool_mean(h, ctx_len)}
    if pool == "all":
        return {"last": pool_last(h, ctx_len), "mean": pool_mean(h, ctx_len)}
    raise ValueError(pool)


# ── forward pass + pool ─────────────────────────────────────────────────────
def extract_hidden(model, input_ids, attention_mask, ctx_len, layers, pool, final_norm):
    n_layers = num_hidden_layers(model.config)
    valid = set(all_shorthands(n_layers))
    normed_sh = f"act/{n_layers - 1}_normed"
    wanted = valid if layers == "all" else set(layers)
    bad = wanted - valid
    if bad:
        raise ValueError(f"unknown shorthands {bad}")
    raw_wanted = wanted - {normed_sh}

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, use_cache=False,
                    output_hidden_states=True)
    hs = out.hidden_states
    result = {}
    for sh in raw_wanted:
        h = hs[shorthand_to_layer(sh)][0].float()
        for stat, vec in _apply_pool(h, ctx_len, pool).items():
            result[f"{stat}.{sh}"] = vec
    if normed_sh in wanted:
        with torch.no_grad():
            raw = hs[shorthand_to_layer(normed_sh)][0]
            hn = final_norm(raw.unsqueeze(0))[0].float()
        for stat, vec in _apply_pool(hn, ctx_len, pool).items():
            result[f"{stat}.{normed_sh}"] = vec
    return result


def extract_trajectory(traj, model, tokenizer, max_tokens, layers, pool, context_fn, final_norm):
    device = next(model.parameters()).device
    hidden = {}
    for step_idx in iter_scoreable_steps(traj):
        enc = context_fn(traj, step_idx, tokenizer, max_tokens=max_tokens)
        input_ids = enc["input_ids"].to(device)
        ctx_len = enc["ctx_len"]
        if input_ids.shape[1] <= ctx_len:
            continue
        hidden[step_idx] = extract_hidden(model, input_ids, None, ctx_len, layers, pool, final_norm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return hidden


def parse_args():
    p = argparse.ArgumentParser(description="Extract pooled hidden states.")
    p.add_argument("--model", required=True, help="Shorthand (output dir name + metadata).")
    p.add_argument("--model-path", required=True, help="HF name/local path to load.")
    p.add_argument("--input", required=True, help="Data root (contains <subset>/*.json).")
    p.add_argument("--subset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", nargs="+", default=["all"])
    p.add_argument("--pool", choices=["last", "mean", "all"], default="all")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    device_map = "auto" if args.device == "auto" else {"": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    adapter = get_adapter(args.model_path)
    model, tokenizer = adapter.load(args.model_path, dtype, device_map)
    n_layers = adapter.num_layers(model)
    layers = args.layers[0] if len(args.layers) == 1 and args.layers[0] == "all" else args.layers
    if layers == "all":
        blocks = adapter.extract_block_indices(model)
        layers = ["embed"] + [f"act/{i}" for i in blocks] + [f"act/{n_layers - 1}_normed"]
        print(f"extractable blocks: {blocks}")
    final_norm = adapter.final_norm(model)
    context_fn = functools.partial(build_context, template_kwargs=adapter.template_kwargs())

    trajs = load_dataset(args.input, subset=args.subset)
    end = args.end_idx if args.end_idx is not None else len(trajs)
    trajs = trajs[args.start_idx:end]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"model": args.model, "layers": layers if layers != "all" else "all",
         "pool": args.pool, "max_tokens": args.max_tokens, "dtype": args.dtype,
         "subset": args.subset}, indent=2))

    t0 = time.perf_counter()
    for traj in tqdm(trajs):
        out_path = out_dir / traj.filename.replace(".json", ".safetensors")
        if out_path.exists():
            continue
        hidden = extract_trajectory(traj, model, tokenizer, args.max_tokens, layers,
                                    args.pool, context_fn, final_norm)
        flat = {f"{s}.{k}": t.contiguous() for s, d in hidden.items() for k, t in d.items()}
        assert flat, f"no hidden states for {traj.filename}"
        save_file(flat, out_path, metadata={"payload_metadata": json.dumps(extract_metadata(traj))})
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
