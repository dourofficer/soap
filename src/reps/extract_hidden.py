"""
attribscope/reps/extract_hidden.py — Extract and save pooled hidden states for trajectories.

For each scoreable step in each trajectory, runs one forward pass and saves
pooled hidden-state vectors along the residual stream for every requested layer.

Output layout
-------------
One .safetensors file per trajectory under --output:
    Flat keys: "{step_idx}.{shorthand}.{pool_stat}"

    Examples:
        "3.last.embed"           step 3, embedding layer, last-token
        "3.mean.act/35"          step 3, layer 35, mean over output tokens
        "3.last.act/35_normed"   step 3, final layer post-norm, last-token

    Metadata stored in the safetensors header under "payload_metadata" (JSON).

Usage
-----
# Specific layers, last-token pooling
python -m attribscope.reps.extract_hidden \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B" \
    --input  data/ww/hand-crafted \
    --output outputs/hidden/llama-3.1-8b/hand-crafted \
    --layers embed act/0 act/35 act/35_normed \
    --pool   last \
    --max_tokens 8192

# All layers, both pooling strategies
python -m attribscope.reps.extract_hidden \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct" \
    --input  data/ww/hand-crafted \
    --output outputs/hidden/llama-3.1-8b/hand-crafted \
    --layers all \
    --pool   all \
    --max_tokens 16000 \
    --start_idx 0 --end_idx 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from typing import Any, Callable
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel
)
from safetensors.torch import save_file

from ..data.trajectory import Trajectory, load_dataset
from ..data.context import (
    iter_scoreable_steps, 
    build_context, 
    build_context_base
)

from .hidden import extract_hidden

# CONTEXT_FNS = {
#     "Qwen3-8B":      build_context_template,
#     "Qwen3-8B-Base": build_context_base,
#     "Llama-3.1-8B-Instruct": build_context_template,
#     "Llama-3.1-8B":  build_context_base
# }

# CONTEXT_FNS = {
#     "Qwen3-8B":      build_context_base,
#     "Qwen3-8B-Base": build_context_base,
#     "Llama-3.1-8B-Instruct": build_context_base,
#     "Llama-3.1-8B":  build_context_base
# }


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectory_hidden(
    traj:             Trajectory,
    model:            PreTrainedModel,
    tokenizer:        None,
    max_tokens:       int,
    layers:           list[str] | str,   # list of ints or "all"
    pool:             str,               # "last" | "mean" | "all"
    context_strategy: str = "dependency",
    context_fn:       Callable = build_context,
    pbar=None,
) -> dict[int, dict[str, Tensor]]:
    """Extract pooled hidden states for all scoreable steps in a trajectory.

    Parameters
    ----------
    layers : list of layer indices (0 = embedding, 1..N = transformer blocks)
             or "all" to extract every layer.
    pool   : "last" | "mean" | "all"

    Returns
    -------
    dict mapping step_idx → flat {"{shorthand}.{stat}": Tensor}
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    hidden: dict[int, dict[str, Tensor]] = {}
    device = next(model.parameters()).device

    for step_idx in iter_scoreable_steps(traj):
        encoded = context_fn(
            traj,
            step_idx,
            tokenizer,
            max_tokens=max_tokens,
            strategy=context_strategy,
        )

        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        ctx_len = encoded["ctx_len"]
        seq_len = input_ids.shape[1]

        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file",     traj.filename),
                ("seq_len",  seq_len),
                ("ctx_len",  ctx_len),
                ("step_idx", step_idx),
                ("n_steps",  len(traj.history)),
            ]))

        # Skip steps with no output tokens
        if seq_len <= ctx_len:
            continue

        step_hidden = extract_hidden(
            model, input_ids, attention_mask, ctx_len, layers, pool,
        )
        hidden[step_idx] = step_hidden

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return hidden

def extract_trajectories_hidden(
    trajectories:     list[Trajectory],
    out_dir:          Path,
    model:            PreTrainedModel,
    tokenizer:        None,
    max_tokens:       int,
    layers:           list[str] | str,   # list of ints or "all"
    pool:             str,               # "last" | "mean" | "all"
    context_strategy: str = "dependency", # "dependency" | "all"
    context_fn:       Callable = build_context,
):
    pbar = tqdm(trajectories)
    for traj in pbar:
        out_path = out_dir / traj.filename.replace(".json", ".safetensors")
        if out_path.exists():
            pbar.write(f"  skip: {traj.filename}")
            continue

        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))

        hidden = extract_trajectory_hidden(
            traj, model, tokenizer, max_tokens,
            layers, pool, context_strategy,
            context_fn, pbar,
        )

        # Flatten to "{step_idx}.{shorthand}.{stat}" → Tensor
        # e.g. "3.hidden/35.last", "3.hidden/35_normed.mean"
        flat_dict = {
            f"{step_idx}.{key}": tensor.contiguous()
            for step_idx, step_dict in hidden.items()
            for key, tensor in step_dict.items()
        }
        assert flat_dict, f"No hidden states extracted for trajectory {traj.filename}"
        header_metadata = {
            "payload_metadata": json.dumps(_extract_metadata(traj))
        }
        save_file(flat_dict, out_path, metadata=header_metadata)


def _extract_metadata(traj: Trajectory) -> dict:
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract pooled hidden states along the residual stream for trajectory datasets."
    )
    p.add_argument("--model",   required=True, help="HF model name or local path.")
    p.add_argument("--input",   required=True, help="Dataset directory.")
    p.add_argument("--output",  required=True, help="Output directory for .safetensors files.")
    p.add_argument(
        "--layers", required=True, nargs="+",
        help=(
            "Layer indices to extract (0 = embedding, 1..N = after transformer block). "
            "Pass 'all' to extract every layer."
        ),
    )
    p.add_argument(
        "--pool", required=True, choices=["last", "mean", "all"],
        help=(
            "'last'  — hidden state of the last output token. "
            "'mean'  — mean over all output tokens. "
            "'all'   — both strategies."
        ),
    )
    p.add_argument("--max_tokens",  type=int, default=8192)
    p.add_argument("--start_idx",   type=int, default=0)
    p.add_argument("--end_idx",     type=int, default=None)
    p.add_argument("--device",      default=None)
    p.add_argument("--dtype",       choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--subset",      default=None)
    p.add_argument(
        "--context", choices=["dependency", "all"], default="dependency",
        help="Context selection strategy for hand-crafted trajectories.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────────
    if args.device == "auto": device_map = "auto"
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = {"":device}

    dtype_map = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Resolve --layers ──────────────────────────────────────────────────────
    layers = args.layers
    if len(layers) == 1 and layers[0] == "all": 
        layers = layers[0]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model → {args.device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        # device_map={"": device},
        device_map=device_map,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    print(f"  {n_params / 1e9:.2f}B parameters  |  {n_layers} transformer layers.")

    # ── Context builder ──────────────────────────────────────────────────────
    model_key  = Path(args.model).parts[-1]
    # context_fn = CONTEXT_FNS[model_key]
    context_fn = build_context

    # ── Validate --layers against actual model depth ──────────────────────────
    if layers != "all":
        print(f"  Target layers: {layers}  |  pool: {args.pool}")
    else:
        print(f"  Target layers: all ({n_layers + 1} total)  |  pool: {args.pool}")

    # ── Load data ─────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"  {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    # ── Output dir + config ───────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model":    args.model,
        "layers":   layers if layers != "all" else "all",
        "pool":     args.pool,
        "max_tokens": args.max_tokens,
        "dtype":    args.dtype,
        "subset":   subset,
        "context":  args.context,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── Extract ───────────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    extract_trajectories_hidden(
        trajectories, out_dir, model, tokenizer, args.max_tokens,
        layers, args.pool, args.context, context_fn
    )

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s  "
        f"({elapsed / max(len(trajectories), 1):.1f}s/traj)"
    )


if __name__ == "__main__":
    main()