"""Streaming attention-mass extraction — stage 2, the causal signal.

WHAT THIS PRODUCES AND WHY
--------------------------
The rescoring strategies need a dependency structure: how much did step ``t`` actually
*rely on* step ``i``? We take the model's own answer — the fraction of step ``t``'s query
attention that lands on the tokens of predecessor step ``i``:

    m_{i,t} = mean over t's query tokens of  sum_{q in tokens(i)} A[query, q]

This is a *measurement*, not a model we impose: it reads which earlier steps the proxy
was looking at while producing this one. Aggregating to STEP granularity (rather than
keeping token-level attention) is what makes it usable — attribution is per step, so
token-level detail would only be summed away later at much greater cost.

WHY THE SDPA OVERRIDE (the central trick)
-----------------------------------------
The obvious route, ``output_attentions=True``, materialises the full ``(1, H, N, N)``
post-softmax matrix for EVERY layer inside one forward. At 8k tokens with 32 heads that
is tens of GB per layer — it simply OOMs on the long trajectories this project cares
about.

We also do not hand-reconstruct Q/K from the weights: that is fragile across
architectures (gated projections, partial RoPE, per-head norms, GQA) and silently wrong
when an assumption breaks.

Instead we temporarily swap the *registered* ``sdpa`` attention function for a wrapper.
The wrapper receives the query/key tensors the model is **about to attend with** — already
projected, gated, per-head-normed and RoPE'd by the module itself, so they are correct
for whatever architecture is loaded — then:

  1. delegates to the genuine ``sdpa_attention_forward`` so the model's own forward is
     completely unaffected and memory stays O(N);
  2. separately slices Q down to just the SCORED step's rows and scores those against the
     full K, applying the causal mask and softmax by hand.

Peak attention memory is therefore ``O(H * |T_t| * N)`` instead of ``O(H * N^2)`` — the
scored step is short, so this is a small multiple of the sequence rather than its square.

Hybrid models come out right for free: only full-softmax-attention layers dispatch
through the attention interface, so on Qwen3.5 (3 Gated-DeltaNet layers per 1 attention
layer) the wrapper is simply never called on the linear-attention layers. The stored
``L`` axis is exactly the full-attention layers, and their decoder indices are recorded
in ``config.json`` as ``attn_block_indices``. NOTE: this means ``L`` indexes ATTENTION
BLOCKS, not the activation stage's layer positions — the two are not interchangeable,
and ``layer_range`` labels downstream refer to these rows.

OUTPUT SCHEMA (consumed by ``src.rescore.weights.aggregate_attn``)
------------------------------------------------------------------
    "{step}.raw_attn"           (L, n_ctx)     head-averaged mass into each predecessor
    "{step}.raw_attn_per_head"  (L, H, n_ctx)  same, per head (kept for head analysis)
    "{step}.attn_residual_mass" (L,)           mass NOT landing in any predecessor step
    "{step}.ctx_indices"        (n_ctx,)       which history step each column refers to

``attn_residual_mass`` is the leftover after bucketing: attention that went to the
template scaffolding, the step's own tokens, or an attention sink rather than to a
predecessor step. A step with high residual mass barely consulted its context — useful
on its own, and a sanity check that the buckets are not silently losing mass.

Rows are NOT normalised here; normalisation over predecessors happens at aggregation
time, after a layer band is chosen. Text-only: no vision tower is run, so nothing but
the text stack reaches the override.

    # from v2/
    python -m src.extract.attention \
        --model qwen3.5-9b --model-path ../../hub/Qwen/Qwen3.5-9B \
        --input data/correct-full --subset magentic \
        --output-root outputs/correct-full/attention --max_tokens 8192

    # or for a whole dataset, driven by the manifest:
    DATASET=correct-full ./scripts/extract.sh
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
    """(n_ctx, N) one-hot mask: row j marks every token position belonging to step j.

    Bucketing token-level attention into per-step mass is then a single matmul
    ``A_t @ M.T`` instead of a Python loop over steps — which matters because this runs
    once per layer per step. Rows are disjoint by construction (``src.data.context``
    guarantees every token belongs to exactly one step), so no attention is double
    counted, and positions in no row (template scaffolding, the scored step itself)
    simply fall out into the residual mass.
    """
    M = torch.zeros(len(ctx_step_ids), seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped-query KV heads so each query head has its own K.

    Under GQA/MQA several query heads share one KV head, so ``key`` arrives with fewer
    heads than ``query``. The genuine SDPA path handles that internally; our manual
    score computation needs them aligned, so KV heads are repeated to match.
    """
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
    """Bucket ONE layer's step-t attention into per-predecessor mass.

    ``query``/``key`` are (1, H, N, d), post-projection and post-RoPE — exactly what the
    model is about to attend with, which is why no architecture-specific reconstruction
    is needed here.

    The whole point is that ``scores`` is (H, |T_t|, N), never (H, N, N): we keep only the
    rows of the scored step. Steps:

      1. slice Q to the scored step's token positions (``query_idx``);
      2. score against the FULL K and apply the model's own scaling;
      3. re-apply causality by hand — a query at absolute position p may not see keys at
         positions > p. This must be done explicitly because we bypassed SDPA's internal
         mask; without it, the softmax would normalise over future tokens and the
         resulting distribution would be wrong;
      4. softmax in fp32 (the model may run in bf16; normalising a distribution in bf16
         loses meaningful precision);
      5. pool over the step's query tokens — ``mean`` treats the step as a whole, ``last``
         uses only its final token;
      6. bucket into predecessor steps with one matmul against the key mask.

    Result is (H, n_ctx) per layer, moved to CPU immediately so GPU memory stays flat
    across layers.
    """
    n_q, n_kv = query.shape[1], key.shape[1]
    if n_q != n_kv:
        key = _repeat_kv(key, n_q // n_kv)
    dev = query.device
    qi = _STREAM.query_idx.to(dev, non_blocking=True)
    km = _STREAM.key_mask.to(dev, non_blocking=True)
    q_t = query[0].index_select(1, qi)                       # (H, |T_t|, d)
    k_full = key[0]                                          # (H, N, d)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(q_t, k_full.transpose(-1, -2)) * scale   # (H, |T_t|, N)
    # Causal mask: key position must not exceed the query's absolute position.
    N = k_full.shape[1]
    key_pos = torch.arange(N, device=dev)
    not_causal = key_pos[None, :] > qi[:, None]              # (|T_t|, N)
    scores = scores.masked_fill(not_causal[None], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1)            # (H, |T_t|, N)
    A_t = probs.mean(dim=1) if _STREAM.query_pool == "mean" else probs[:, -1, :]
    _STREAM.out_per_head[module.layer_idx] = (A_t @ km.T).cpu()    # (H, n_ctx)


def _streaming_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """Drop-in for ``sdpa_attention_forward``: real output first, our reduction second.

    Delegating to the genuine implementation keeps the model's forward bit-for-bit
    unchanged — the reduction is a pure side-channel and can never perturb the states the
    activation stage reads.
    """
    out = sdpa_attention_forward(module, query, key, value, attention_mask,
                                 scaling=scaling, dropout=dropout, **kw)
    if _STREAM.armed:
        _reduce(module, query, key, scaling)
    return out


@contextmanager
def _override_sdpa():
    """Install the wrapper in the global attention registry for one forward pass.

    The swap is global, so it is scoped to a context manager and restored in ``finally``:
    leaving it installed would silently add work to every later forward in the process.
    The ``_STREAM.armed`` flag is a second guard, so even an unexpected dispatch outside
    an extraction reduces nothing.
    """
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
        # Stack captured layers in decoder order -> (L, H, n_ctx). For hybrid models only
        # full-attention blocks were ever called, so L is exactly those.
        blocks = sorted(oph)
        per_head = torch.stack([oph[i] for i in blocks], dim=0).float().contiguous()
        flat[f"{step_idx}.raw_attn"] = per_head.mean(dim=1).contiguous()   # head-average
        flat[f"{step_idx}.raw_attn_per_head"] = per_head
        # Each head's row sums to <= 1 over predecessors; the shortfall is attention that
        # went to scaffolding / the step's own tokens / sinks. Averaged over heads.
        flat[f"{step_idx}.attn_residual_mass"] = (1.0 - per_head.sum(dim=-1).mean(dim=-1)).contiguous()
        flat[f"{step_idx}.ctx_indices"] = torch.tensor(ctx_step_ids, dtype=torch.long)
    return flat


def _force_sdpa(model):
    """Pin the attention implementation to 'sdpa' so the override is actually hit.

    A checkpoint may default to flash-attention or eager; either would bypass the
    registry entry we replace, and the extraction would silently capture nothing (which
    ``extract_trajectory`` turns into a hard error rather than empty output). Multimodal
    wrappers keep a separate ``text_config``, so both configs are set.
    """
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
