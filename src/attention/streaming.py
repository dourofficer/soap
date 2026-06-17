"""Streaming attention-mass extraction (low memory).

Why this exists
---------------
The sibling extractor `eager.py` relies on `output_attentions=True`, which
materialises the full (1, H, N, N) post-softmax matrix for every attention
layer inside a single forward. That OOMs on long sequences / large models
(e.g. Qwen3.5-9B at 8k tokens).

This extractor never materialises (N, N). Instead of hand-reconstructing Q/K
(which is fragile across architectures — gated q_proj, partial RoPE, mRoPE,
per-head norms), we temporarily override the registered ``sdpa`` attention
function with a thin wrapper. The wrapper:

  1. Receives the query/key tensors the model is *about to* attend with —
     already projected, gate-split, per-head-normed and RoPE'd by the module,
     so they are correct for whatever architecture is loaded.
  2. Delegates the real output to the genuine ``sdpa_attention_forward`` (so the
     model proceeds normally and memory stays O(N) via SDPA).
  3. Separately slices Q to the rows in T_t, scores them against the full K,
     applies the causal mask + softmax, and reduces to (H, n_ctx).

Peak attention memory is O(H * |T_t| * N) instead of O(H * N^2).

Because only full-softmax-attention layers dispatch through the attention
interface, hybrid models such as Qwen3.5 (where 3/4 of layers are Gated
DeltaNet linear-attention) are handled for free: the wrapper is simply never
called on the DeltaNet layers, so the captured layers are exactly the
full-attention ones. The stored L axis therefore indexes those layers; their
decoder indices are written to config.json as ``attn_block_indices``.

Output layout (unchanged, consumed by process_weighting / reweight_scores):
    "{step_idx}.raw_attn"           : (L, n_ctx)     float32
    "{step_idx}.raw_attn_per_head"  : (L, H, n_ctx)  float32
    "{step_idx}.attn_residual_mass" : (L,)           float32
    "{step_idx}.ctx_indices"        : (n_ctx,)       int64

Notes
-----
* Text-only: assumes no image/audio inputs, so the vision tower (if any) is
  never run and never reaches the overridden sdpa.
* The model must dispatch through ``sdpa``; main() forces that after load.

CLI example
-----------
    python -m src.attention.streaming \
        --model        qwen3.5-9b \
        --subset       hand-crafted \
        --input        data/ww \
        --output-root  outputs/attention \
        --max_tokens   8192 \
        --query-pool   mean \
        --device       auto \
        --dtype        bfloat16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from safetensors.torch import save_file

from ..data.trajectory import Trajectory, load_dataset
from ..data.context    import iter_scoreable_steps, separate_steps
from ..models import get_adapter

HUB = "/home/thanhdo/hub"
MODELS = {
    "llama-3.1-8b": f"{HUB}/meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b":     f"{HUB}/Qwen/Qwen3-8B",
    "qwen3-4b":     f"{HUB}/Qwen/Qwen3-4B",
    "qwen3-14b":    f"{HUB}/Qwen/Qwen3-14B",
    "qwen3.5-9b":   f"{HUB}/Qwen/Qwen3.5-9B",
    "deepseek-8b":  f"{HUB}/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


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


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """GQA: replicate KV heads. (B, H_kv, N, d) -> (B, H_kv * n_rep, N, d)."""
    if n_rep == 1:
        return x
    bsz, n_kv, slen, head_dim = x.shape
    x = x[:, :, None].expand(bsz, n_kv, n_rep, slen, head_dim)
    return x.reshape(bsz, n_kv * n_rep, slen, head_dim)


def _build_key_mask(
    ctx_step_ids: list[int],
    step_tokens:  dict[int, list[int]],
    seq_len:      int,
    device:       torch.device,
) -> Tensor:
    """Build the (n_ctx, N) one-hot mask used to sum-over-T_i in one matmul."""
    n_ctx = len(ctx_step_ids)
    M = torch.zeros(n_ctx, seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M


# ──────────────────────────────────────────────────────────────────────────
# sdpa override: real output via genuine sdpa + our chunked reduction
# ──────────────────────────────────────────────────────────────────────────

class _StreamState:
    """Per-step reduction state read by the sdpa wrapper during a forward."""
    armed:        bool = False
    query_idx:    Tensor | None = None     # (|T_t|,) absolute query positions
    key_mask:     Tensor | None = None     # (n_ctx, N) one-hot per predecessor
    query_pool:   str = "mean"             # "mean" | "last"
    out_per_head: dict = {}                # {layer_idx: (H, n_ctx) cpu tensor}


_STREAM = _StreamState()


@torch.no_grad()
def _reduce(module, query: Tensor, key: Tensor, scaling: float | None) -> None:
    """query: (1, H_q, N, d); key: (1, H_kv, N, d). Both post-RoPE, model dtype."""
    n_q, n_kv = query.shape[1], key.shape[1]
    if n_q != n_kv:
        key = _repeat_kv(key, n_q // n_kv)

    dev = query.device
    qi  = _STREAM.query_idx.to(dev, non_blocking=True)
    km  = _STREAM.key_mask.to(dev, non_blocking=True)

    q_t    = query[0].index_select(1, qi)          # (H, |T_t|, d)
    k_full = key[0]                                # (H, N, d)
    scale  = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(q_t, k_full.transpose(-1, -2)) * scale   # (H, |T_t|, N)

    N          = k_full.shape[1]
    key_pos    = torch.arange(N, device=dev)
    not_causal = key_pos[None, :] > qi[:, None]                    # (|T_t|, N)
    scores     = scores.masked_fill(not_causal[None], float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1)                  # (H, |T_t|, N)
    A_t   = probs.mean(dim=1) if _STREAM.query_pool == "mean" else probs[:, -1, :]
    _STREAM.out_per_head[module.layer_idx] = (A_t @ km.T).cpu()    # (H, n_ctx)


def _streaming_sdpa(module, query, key, value, attention_mask,
                    scaling=None, dropout=0.0, **kwargs):
    """Drop-in for sdpa_attention_forward: real output + our reduction."""
    out = sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )
    if _STREAM.armed:
        _reduce(module, query, key, scaling)
    return out


@contextmanager
def _override_sdpa():
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _streaming_sdpa
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig


# ──────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_trajectory_qk_attention(
    traj:        Trajectory,
    model:       PreTrainedModel,
    tokenizer:   PreTrainedTokenizer,
    max_tokens:  int,
    adapter,
    query_pool:  str          = "mean",
    pbar:        tqdm | None  = None,
) -> dict[str, Tensor]:
    """Per scored step: one forward with the sdpa wrapper installed."""
    flat: dict[str, Tensor] = {}
    device      = next(model.parameters()).device
    template_kw = adapter.template_kwargs()

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

        query_idx = torch.tensor(step_tokens[step_idx], device=device, dtype=torch.long)
        key_mask  = _build_key_mask(ctx_step_ids, step_tokens, seq_len, device)

        _STREAM.out_per_head = {}
        _STREAM.query_idx    = query_idx
        _STREAM.key_mask     = key_mask
        _STREAM.query_pool   = query_pool
        _STREAM.armed        = True
        try:
            with _override_sdpa():
                _ = model(input_ids, use_cache=False)
        finally:
            _STREAM.armed = False

        out_per_head = _STREAM.out_per_head
        if not out_per_head:
            raise RuntimeError(
                f"No attention layers captured at step {step_idx} of "
                f"{traj.filename}. Is the model dispatching through sdpa?"
            )

        block_idxs = sorted(out_per_head)
        per_head   = torch.stack(
            [out_per_head[i] for i in block_idxs], dim=0,
        ).float().contiguous()                                       # (L, H, n_ctx)
        head_mean  = per_head.mean(dim=1).contiguous()               # (L, n_ctx)
        residual   = (1.0 - per_head.sum(dim=-1)
                                  .mean(dim=-1)).contiguous()         # (L,)

        flat[f"{step_idx}.raw_attn"]            = head_mean
        flat[f"{step_idx}.raw_attn_per_head"]   = per_head
        flat[f"{step_idx}.attn_residual_mass"]  = residual
        flat[f"{step_idx}.ctx_indices"]         = torch.tensor(
            ctx_step_ids, dtype=torch.long,
        )

    return flat


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract aggregated attention mass via a low-memory sdpa override.",
    )
    p.add_argument("--model",       required=True, choices=list(MODELS))
    p.add_argument("--subset",      default=None)
    p.add_argument("--input",       required=True, help="Dataset directory.")
    p.add_argument("--output-root", required=True, help="Output directory.")
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


def _force_sdpa(model) -> None:
    """Ensure attention dispatches through 'sdpa' so the override is hit."""
    for cfg in {id(model.config): model.config,
                **({id(model.config.text_config): model.config.text_config}
                   if getattr(model.config, "text_config", None) is not None else {})}.values():
        cfg._attn_implementation = "sdpa"


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

    model_path = MODELS[args.model]
    adapter    = get_adapter(model_path)
    print(f"Loading model -> {args.device} ({args.dtype}, sdpa + streaming override)")
    model, tokenizer = adapter.load(model_path, torch_dtype, device_map)  # eager=False -> sdpa
    _force_sdpa(model)

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
        "impl":       "sdpa_streaming",
        "attn_block_indices": adapter.extract_block_indices(model),
    }, indent=2))

    pbar = tqdm(trajectories, desc="Processing trajectories")
    for traj in pbar:
        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))
        out_path = out_root / f"{Path(traj.filename).stem}.safetensors"
        if out_path.exists():
            continue

        flat = extract_trajectory_qk_attention(
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
            metadata={"payload_metadata": json.dumps(_extract_metadata(traj))},
        )


if __name__ == "__main__":
    main()