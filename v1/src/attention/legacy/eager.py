"""Eager attention-mass extractor — LEGACY / validation reference.

Superseded on the main pipeline by ``src/attention/streaming.py`` (low memory,
no (N,N) materialization). Kept because it is the exact, simple implementation
used to validate ``streaming`` (both must agree to ~1e-4 on a short trajectory).
Now shares the trajectory-metadata + key-mask helpers with the live extractor and
uses the same ``--model``/``--model-path`` interface (path resolved by the caller
from the dataset manifest).

Extract aggregated attention-mass step-to-step weights from a proxy LLM.

Mathematical definition
-----------------------
For each scored step t and each predecessor step i with token spans
T_t and T_i, we capture the post-softmax attention probability matrix
A^(l,h) ∈ [0,1]^(N x N) at every layer l and head h of the proxy model,
then compute the attention mass routed from step t into step i:

    m^(l,h)_{i,t} = (1/|T_t|) sum_{p in T_t} sum_{q in T_i} A^(l,h)_{p,q}

This is the mean (over query tokens of step t) of the probability mass
landing in step i's key tokens. Because A is row-stochastic and causal,
m^(l,h)_{i,t} lies in [0, 1] and sum_i m^(l,h)_{i,t} <= 1.

Per-step output keys
--------------------
    "{step_idx}.raw_attn"            : (L, n_ctx)     float32
    "{step_idx}.raw_attn_per_head"   : (L, H, n_ctx)  float32
    "{step_idx}.attn_residual_mass"  : (L,)           float32
    "{step_idx}.ctx_indices"         : (n_ctx,)       int64

Layer indexing matches streaming: layer 0 = layers[0].self_attn (no
off-by-one to the embedding slot). Loads with attn_implementation="eager"
so the post-softmax probabilities our hooks depend on are materialized.

CLI example
-----------
    python -m src.attention.legacy.eager \\
        --model llama-3.1-8b \\
        --model-path /path/to/Llama-3.1-8B-Instruct \\
        --subset hand-crafted --input data/ww \\
        --output-root outputs/weighting_attn \\
        --max_tokens 8192 --query-pool mean --device cuda --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from safetensors.torch import save_file

# [data_v2 swap] rollback: uncomment the two ...data lines, delete the ...data_v2 ones
# from ...data.trajectory import Trajectory, load_dataset
# from ...data.context    import iter_scoreable_steps, separate_steps
from ...data_v2.trajectory import Trajectory, load_dataset
from ...data_v2.context    import iter_scoreable_steps, separate_steps
from ...models import get_adapter
from ...utils.metadata import extract_metadata
from .._common import build_key_mask


# ──────────────────────────────────────────────────────────────────────────
# Per-step attention reduction
# ──────────────────────────────────────────────────────────────────────────

def _make_attn_hook(
    layer_idx:    int,
    query_idx:    Tensor,           # (q_len,) long on device — positions in T_t
    key_mask:     Tensor,           # (n_ctx, N) float on device
    query_pool:   str,              # "mean" | "last"
    out_per_head: dict[int, Tensor],
):
    """Forward hook reducing an attention layer's (1, H, N, N) probs on the fly
    to a (H, n_ctx) summary, then dropping the weights so the outer model does
    not accumulate the full attentions tuple (~GBs/layer)."""
    def hook(module, args, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return output
        attn = output[1]
        if attn is None:
            return output

        A = attn[0]                                          # (H, N, N)
        qi = query_idx.to(A.device, non_blocking=True)
        km = key_mask .to(A.device, non_blocking=True)

        if query_pool == "mean":
            A_t = A.index_select(1, qi).float().mean(dim=1)  # (H, N)
        elif query_pool == "last":
            A_t = A[:, qi[-1], :].float()                    # (H, N)
        else:
            raise ValueError(f"Unknown query_pool={query_pool!r}")

        m = A_t @ km.T                                        # (H, n_ctx)
        out_per_head[layer_idx] = m.cpu()
        return (output[0], None) + tuple(output[2:])
    return hook


# ──────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_trajectory_attention(
    traj:        Trajectory,
    model:       PreTrainedModel,
    tokenizer:   PreTrainedTokenizer,
    max_tokens:  int,
    adapter,
    query_pool:  str          = "mean",
    pbar:        tqdm | None  = None,
) -> dict[str, Tensor]:
    """Per scored step: one forward pass with hooks installed, capturing
    aggregated attention mass to every in-graph predecessor step."""
    flat: dict[str, Tensor] = {}
    device = next(model.parameters()).device

    layers       = adapter.decoder_layers(model)
    block_idxs   = adapter.extract_block_indices(model)
    attn_modules = [(i, layers[i].self_attn) for i in block_idxs]
    template_kw  = adapter.template_kwargs()

    for step_idx in iter_scoreable_steps(traj):
        encoded     = separate_steps(
            traj, step_idx, tokenizer,
            max_tokens=max_tokens, template_kwargs=template_kw,
        )
        input_ids   = encoded["input_ids"].to(device)
        step_tokens = encoded["step_tokens"]

        ctx_step_ids = sorted(m for m in step_tokens if m != step_idx)
        if not ctx_step_ids:
            continue

        seq_len = input_ids.shape[1]
        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file",     traj.filename),
                ("seq_len",  seq_len),
                ("step_idx", step_idx),
                ("n_steps",  len(traj.history)),
            ]))

        query_idx = torch.tensor(step_tokens[step_idx], device=device,
                                 dtype=torch.long)
        key_mask  = build_key_mask(ctx_step_ids, step_tokens, seq_len, device)

        out_per_head: dict[int, Tensor] = {}
        handles = [
            mod.register_forward_hook(
                _make_attn_hook(
                    layer_idx=bi, query_idx=query_idx, key_mask=key_mask,
                    query_pool=query_pool, out_per_head=out_per_head,
                ),
            )
            for bi, mod in attn_modules
        ]

        try:
            _ = model(input_ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if len(out_per_head) != len(attn_modules):
            raise RuntimeError(
                f"Captured {len(out_per_head)}/{len(attn_modules)} layers at step "
                f"{step_idx} of {traj.filename}; hook misfire?"
            )

        per_head = torch.stack(
            [out_per_head[bi] for bi, _ in attn_modules], dim=0,
        ).float().contiguous()                                       # (L, H, n_ctx)

        head_mean = per_head.mean(dim=1).contiguous()                # (L, n_ctx)
        residual  = (1.0 - per_head.sum(dim=-1)
                                  .mean(dim=-1)).contiguous()         # (L,)

        flat[f"{step_idx}.raw_attn"]            = head_mean
        flat[f"{step_idx}.raw_attn_per_head"]   = per_head
        flat[f"{step_idx}.attn_residual_mass"]  = residual
        flat[f"{step_idx}.ctx_indices"]         = torch.tensor(
            ctx_step_ids, dtype=torch.long,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return flat


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract aggregated attention-mass step-to-step weights (eager).",
    )
    p.add_argument("--model",       required=True,
                   help="Model shorthand (used for the output dir + metadata).")
    p.add_argument("--model-path",  required=True,
                   help="Resolved HF name/local path to load (from manifest model_paths).")
    p.add_argument("--subset",      default=None)
    p.add_argument("--input",       required=True, help="Dataset directory.")
    p.add_argument("--output-root", required=True, help="Output directory for .safetensors files.")
    p.add_argument("--max_tokens",  type=int, default=8192)
    p.add_argument("--start_idx",   type=int, default=0)
    p.add_argument("--end_idx",     type=int, default=None)
    p.add_argument("--device",      default=None)
    p.add_argument("--dtype",       choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument(
        "--query-pool", choices=["mean", "last"], default="mean",
        help="Aggregation over query tokens in T_t.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device != "auto":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = {"": device}
    else:
        device_map = "auto"
    torch_dtype = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }[args.dtype]

    model_path = args.model_path
    adapter    = get_adapter(model_path)

    print(f"Loading model → {args.device} ({args.dtype}, eager attention)")
    model, tokenizer = adapter.load(model_path, torch_dtype, device_map, eager=True)

    trajectories = load_dataset(args.input, subset=args.subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"Processing {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    out_root = Path(args.output_root) / args.model / args.subset
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "config.json").write_text(json.dumps({
        "model":      args.model,
        "subset":     args.subset,
        "max_tokens": args.max_tokens,
        "query_pool": args.query_pool,
        "dtype":      args.dtype,
        "attn_block_indices": adapter.extract_block_indices(model),
    }, indent=2))

    pbar = tqdm(trajectories, desc="Processing trajectories")
    for traj in pbar:
        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))
        out_path = out_root / f"{Path(traj.filename).stem}.safetensors"
        if out_path.exists():
            continue

        flat = extract_trajectory_attention(
            traj, model, tokenizer,
            max_tokens=args.max_tokens,
            adapter=adapter,
            query_pool=args.query_pool,
            pbar=pbar,
        )
        if not flat:
            continue

        save_file(
            flat, out_path,
            metadata={"payload_metadata": json.dumps(extract_metadata(traj))},
        )


if __name__ == "__main__":
    main()
