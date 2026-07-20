"""Streaming attention-mass extraction (low memory).

Overrides the registered ``sdpa`` attention with a wrapper that delegates the real
output to genuine SDPA (so memory stays O(N)) and separately reduces step-t query
attention into per-predecessor mass — never materialising (N,N). Handles Llama-3.1 /
Qwen3 and hybrid Qwen3.5 (only full-attention layers dispatch through sdpa).

Output schema (consumed by src.rescore.weights.aggregate_attn):
    "{step}.raw_attn"           (L, n_ctx)     "{step}.ctx_indices"        (n_ctx,)
    "{step}.raw_attn_per_head"  (L, H, n_ctx)  "{step}.attn_residual_mass" (L,)

    # from v2/
    python -m src.extract.attention \
        --model qwen3.5-9b --model-path ../../hub/Qwen/Qwen3.5-9B \
        --input data/correct-full --subset magentic \
        --output-root outputs/correct-full/attention --max_tokens 8192
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from safetensors.torch import save_file

from ..data import Trajectory, load_dataset, iter_scoreable_steps, separate_steps
from ..data.trajectory import extract_metadata
from ..models import get_adapter


def build_key_mask(ctx_step_ids, step_tokens, seq_len, device) -> Tensor:
    """(n_ctx, N) one-hot mask summing token positions per predecessor step."""
    M = torch.zeros(len(ctx_step_ids), seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class _StreamState:
    armed = False
    query_idx = None
    key_mask = None
    query_pool = "mean"
    out_per_head = {}


_STREAM = _StreamState()


@torch.no_grad()
def _reduce(module, query, key, scaling):
    n_q, n_kv = query.shape[1], key.shape[1]
    if n_q != n_kv:
        key = _repeat_kv(key, n_q // n_kv)
    dev = query.device
    qi = _STREAM.query_idx.to(dev, non_blocking=True)
    km = _STREAM.key_mask.to(dev, non_blocking=True)
    q_t = query[0].index_select(1, qi)
    k_full = key[0]
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(q_t, k_full.transpose(-1, -2)) * scale
    N = k_full.shape[1]
    key_pos = torch.arange(N, device=dev)
    not_causal = key_pos[None, :] > qi[:, None]
    scores = scores.masked_fill(not_causal[None], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1)
    A_t = probs.mean(dim=1) if _STREAM.query_pool == "mean" else probs[:, -1, :]
    _STREAM.out_per_head[module.layer_idx] = (A_t @ km.T).cpu()


def _streaming_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask,
                                 scaling=scaling, dropout=dropout, **kw)
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


@torch.no_grad()
def extract_trajectory(traj, model, tokenizer, max_tokens, adapter, query_pool="mean"):
    flat = {}
    device = next(model.parameters()).device
    tk = adapter.template_kwargs()
    for step_idx in iter_scoreable_steps(traj):
        enc = separate_steps(traj, step_idx, tokenizer, max_tokens=max_tokens, template_kwargs=tk)
        input_ids = enc["input_ids"].to(device)
        step_tokens = enc["step_tokens"]
        ctx_step_ids = sorted(m for m in step_tokens if m != step_idx)
        if not ctx_step_ids:
            continue
        seq_len = input_ids.shape[1]
        _STREAM.out_per_head = {}
        _STREAM.query_idx = torch.tensor(step_tokens[step_idx], device=device, dtype=torch.long)
        _STREAM.key_mask = build_key_mask(ctx_step_ids, step_tokens, seq_len, device)
        _STREAM.query_pool = query_pool
        _STREAM.armed = True
        try:
            with _override_sdpa():
                model(input_ids, use_cache=False)
        finally:
            _STREAM.armed = False
        oph = _STREAM.out_per_head
        if not oph:
            raise RuntimeError(f"no attention captured at step {step_idx} of {traj.filename}")
        blocks = sorted(oph)
        per_head = torch.stack([oph[i] for i in blocks], dim=0).float().contiguous()
        flat[f"{step_idx}.raw_attn"] = per_head.mean(dim=1).contiguous()
        flat[f"{step_idx}.raw_attn_per_head"] = per_head
        flat[f"{step_idx}.attn_residual_mass"] = (1.0 - per_head.sum(dim=-1).mean(dim=-1)).contiguous()
        flat[f"{step_idx}.ctx_indices"] = torch.tensor(ctx_step_ids, dtype=torch.long)
    return flat


def _force_sdpa(model):
    cfgs = {id(model.config): model.config}
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        cfgs[id(tc)] = tc
    for cfg in cfgs.values():
        cfg._attn_implementation = "sdpa"


def parse_args():
    p = argparse.ArgumentParser(description="Streaming attention-mass extraction.")
    p.add_argument("--model", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--input", required=True, help="Data root (contains <subset>/*.json).")
    p.add_argument("--subset", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--query-pool", choices=["mean", "last"], default="mean")
    return p.parse_args()


def main():
    args = parse_args()
    device_map = "auto" if args.device == "auto" else {"": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    adapter = get_adapter(args.model_path)
    model, tokenizer = adapter.load(args.model_path, dtype, device_map)
    _force_sdpa(model)

    trajs = load_dataset(args.input, subset=args.subset)
    end = args.end_idx if args.end_idx is not None else len(trajs)
    trajs = trajs[args.start_idx:end]
    out_root = Path(args.output_root) / args.model / args.subset
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(
        {"model": args.model, "subset": args.subset, "max_tokens": args.max_tokens,
         "query_pool": args.query_pool, "dtype": args.dtype, "impl": "sdpa_streaming",
         "attn_block_indices": adapter.extract_block_indices(model)}, indent=2))

    for traj in tqdm(trajs):
        out_path = out_root / f"{Path(traj.filename).stem}.safetensors"
        if out_path.exists():
            continue
        flat = extract_trajectory(traj, model, tokenizer, args.max_tokens, adapter, args.query_pool)
        if flat:
            save_file(flat, out_path, metadata={"payload_metadata": json.dumps(extract_metadata(traj))})


if __name__ == "__main__":
    main()
