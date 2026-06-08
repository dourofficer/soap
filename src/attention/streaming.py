"""attribscope/reps/extract_attention_qk.py

Streaming Q/K version of attention-mass extraction.

Why this exists
---------------
The sibling extractor `extract_attention.py` relies on HF's eager attention
+ `output_attentions=True` to materialise the full (1, H, N, N) post-softmax
attention probability matrix at every layer, then reduces it in a forward
hook. That path hits OOM on large models (Qwen3-14B) or in multi-GPU
dispatch setups where activation memory ends up lopsided on GPU 0.

This extractor sidesteps the problem by never materialising the (N, N)
matrix. We register a *pre*-forward hook on each `self_attn` module that:

  1. Captures the inputs to the attention forward (`hidden_states` and
     `position_embeddings` = the RoPE cos/sin tuple).
  2. Re-runs the module's own `q_proj`, `k_proj`, and (for Qwen3) `q_norm`,
     `k_norm`, plus RoPE manually — giving us Q and K post-RoPE without
     touching HF's attention implementation.
  3. Slices Q to only the rows in T_t, computes scores against the full K,
     applies the causal mask and softmax, and reduces to (H, n_ctx).
  4. Returns nothing — the model's normal forward then runs with SDPA (or
     whatever attention implementation it was loaded with), which doesn't
     materialise (N, N) either.

Peak attention-related memory drops from O(H * N^2) to O(H * |T_t| * N),
which for typical step spans is 10-50x less. The trade-off is a small
amount of duplicate compute (q_proj + k_proj are run twice per layer:
once by us, once by HF's forward) and HF-version dependence on the
attention module's input signature.

Output layout — identical to extract_attention.py
-------------------------------------------------
    "{step_idx}.raw_attn"            : (L, n_ctx)     float32
    "{step_idx}.raw_attn_per_head"   : (L, H, n_ctx)  float32
    "{step_idx}.attn_residual_mass"  : (L,)           float32
    "{step_idx}.ctx_indices"         : (n_ctx,)       int64

Downstream `process_weighting` / `reweight_scores` work unchanged.

Validation
----------
Before trusting numbers from this extractor, run both extractors on a
single short trajectory (small enough that eager attention fits) and
diff the safetensors. They should agree to within float32 rounding;
mismatch indicates either a model-architecture-specific detail this
file doesn't handle, or a bug.

CLI example
-----------
    python -m src.attention.streaming \
        --model        qwen3-4b \
        --subset       hand-crafted \
        --input        data/ww \
        --output-root  outputs/attention \
        --max_tokens   2048 \
        --context      all \
        --query-pool   mean \
        --device       auto \
        --dtype        bfloat16

Architecture compatibility
--------------------------
Tested mental model: Llama-3.1 (no per-head norm) and Qwen3 (with q_norm/
k_norm). Both use the same RoPE formulation (rotate-half style). For
other architectures (e.g. Mistral, GPT-NeoX-style RoPE), the RoPE math
or the attention input signature may differ — check before extending.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import OrderedDict

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizer,
)
from safetensors.torch import save_file

from ..data.trajectory import Trajectory, load_dataset
from ..data.context    import iter_scoreable_steps, preprocess_context

# HUB = "/data/hoang/resources/models"
HUB = "/home/thanhdo/hub"
MODELS = {
    "llama-3.1-8b": f"{HUB}/meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b":     f"{HUB}/Qwen/Qwen3-8B",
    "qwen3-4b":     f"{HUB}/Qwen/Qwen3-4B",
    "qwen3-14b":    f"{HUB}/Qwen/Qwen3-14B",
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


# ──────────────────────────────────────────────────────────────────────────
# RoPE helpers — copied verbatim from HF's modeling_llama / modeling_qwen3.
# Both architectures use the same rotate-half formulation, so one impl works.
# If you adapt this file to a different RoPE family (GPT-NeoX interleaved,
# partial RoPE, etc.), this is the function to swap out.
# ──────────────────────────────────────────────────────────────────────────

def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: (B, H, N, d).  cos, sin: (B, N, d).  Returns same-shape (q, k)."""
    cos = cos.unsqueeze(1)  # (B, 1, N, d) — broadcast over heads
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


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
# Pre-forward hook: capture inputs, recompute Q/K, reduce attention
# ──────────────────────────────────────────────────────────────────────────

def _make_qk_hook(
    layer_idx:    int,
    query_idx:    Tensor,            # (|T_t|,) long; absolute query positions
    key_mask:     Tensor,            # (n_ctx, N) float; one-hot per predecessor
    query_pool:   str,               # "mean" | "last"
    out_per_head: dict[int, Tensor],
):
    """Pre-forward hook on a Qwen3Attention / LlamaAttention module.

    Receives the inputs the module is *about to* process, recomputes Q
    and K post-RoPE from scratch using the module's own projection and
    norm layers, then computes the attention reduction we want. The
    model's forward proceeds normally afterward (uses SDPA), so total
    cost is the model's normal forward + our duplicate q_proj/k_proj +
    one (H, |T_t|, N) softmax. Memory peak is dominated by the
    (H, |T_t|, N) scores/probs tensor, which is small relative to the
    (H, N, N) tensor eager attention would create.

    The hook is registered with `with_kwargs=True` so we get the inputs
    as `(args, kwargs)` regardless of how the caller passes them. In
    transformers >= 4.43 the relevant inputs are `hidden_states` and
    `position_embeddings = (cos, sin)`, passed either positionally or
    as keywords depending on the caller.

    Returns `None` (no modification of inputs) — pre-hooks that return
    `None` are no-ops on the forward path.
    """
    def hook(module, args, kwargs):
        # ── Pull hidden_states and (cos, sin) out of args/kwargs ─────
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif args:
            hidden_states = args[0]
        else:
            raise RuntimeError("attention forward called without hidden_states")

        pe = kwargs.get("position_embeddings")
        if pe is None and len(args) > 1:
            pe = args[1]
        if pe is None:
            raise RuntimeError(
                "position_embeddings not found in hook args/kwargs. "
                "Requires transformers >= 4.43, where the model's outer "
                "forward computes RoPE once and passes the (cos, sin) "
                "tuple into each attention layer."
            )
        cos, sin = pe

        bsz, q_len, _ = hidden_states.shape
        head_dim = module.head_dim   # exposed by both Llama and Qwen3 attention

        # ── Project + per-head norm (Qwen3 only) + transpose ────────
        # Note: ordering matters. Qwen3 norms in the (B, q_len, H, d)
        # layout before transposing, so we follow that exactly.
        q = module.q_proj(hidden_states).view(bsz, q_len, -1, head_dim)
        k = module.k_proj(hidden_states).view(bsz, q_len, -1, head_dim)
        if hasattr(module, "q_norm"):
            q = module.q_norm(q)
        if hasattr(module, "k_norm"):
            k = module.k_norm(k)
        q = q.transpose(1, 2).contiguous()   # (B, H_q,  N, d)
        k = k.transpose(1, 2).contiguous()   # (B, H_kv, N, d)

        # ── RoPE ─────────────────────────────────────────────────────
        # cos/sin live on the same device as the attention layer; q/k
        # share that device since we just computed them there.
        q, k = _apply_rope(q, k, cos, sin)

        # ── GQA: replicate K up to H_q ───────────────────────────────
        n_q, n_kv = q.shape[1], k.shape[1]
        if n_q != n_kv:
            k = _repeat_kv(k, n_q // n_kv)

        # ── Move precomputed indices to compute device (no-op if aligned) ─
        qi = query_idx.to(q.device, non_blocking=True)
        km = key_mask.to(q.device, non_blocking=True)

        # ── Restrict to query rows in T_t ────────────────────────────
        # q_t: (H, |T_t|, d) ; k_full: (H, N, d)
        q_t    = q[0].index_select(1, qi)
        k_full = k[0]

        # ── Attention scores for just our rows: (H, |T_t|, N) ────────
        scale  = head_dim ** -0.5
        scores = torch.matmul(q_t, k_full.transpose(-1, -2)) * scale

        # ── Causal mask ──────────────────────────────────────────────
        # Each row p of `scores` is the query at absolute position qi[p];
        # it can only attend to keys 0..qi[p]. Build a (|T_t|, N) bool
        # mask and broadcast over heads.
        N         = k_full.shape[1]
        key_pos   = torch.arange(N, device=scores.device)
        not_causal = key_pos[None, :] > qi[:, None]                    # (|T_t|, N)
        scores    = scores.masked_fill(not_causal[None], float("-inf"))

        # ── Softmax in fp32 (matches HF's behaviour for stability) ───
        probs = torch.softmax(scores.float(), dim=-1)                  # (H, |T_t|, N)

        # ── Query aggregation ────────────────────────────────────────
        if query_pool == "mean":
            A_t = probs.mean(dim=1)                                     # (H, N)
        elif query_pool == "last":
            # Last token of T_t — typically the closing chat-template EOT.
            A_t = probs[:, -1, :]                                       # (H, N)
        else:
            raise ValueError(f"Unknown query_pool={query_pool!r}")

        # ── Sum over key positions in each predecessor step ──────────
        m = A_t @ km.T                                                  # (H, n_ctx)
        out_per_head[layer_idx] = m.cpu()

        # Pre-hook returns None -> don't modify inputs. The model's
        # normal forward runs after this and computes its own attention
        # via SDPA, which is memory-efficient and doesn't materialise
        # the (N, N) matrix.
    return hook


# ──────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_trajectory_qk_attention(
    traj:        Trajectory,
    model:       PreTrainedModel,
    tokenizer:   PreTrainedTokenizer,
    max_tokens:  int,
    context:     str,
    query_pool:  str          = "mean",
    pbar:        tqdm | None  = None,
) -> dict[str, Tensor]:
    """Per scored step: one forward pass with pre-forward Q/K hooks installed."""
    flat: dict[str, Tensor] = {}
    device = next(model.parameters()).device

    attn_modules: list[nn.Module] = [
        block.self_attn for block in model.model.layers
    ]
    n_layers = len(attn_modules)

    for step_idx in iter_scoreable_steps(traj):
        encoded     = preprocess_context(
            traj, step_idx, tokenizer,
            max_tokens=max_tokens, strategy=context,
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
        key_mask  = _build_key_mask(ctx_step_ids, step_tokens, seq_len, device)

        # ── Install pre-forward hooks ───────────────────────────────
        out_per_head: dict[int, Tensor] = {}
        handles = [
            mod.register_forward_pre_hook(
                _make_qk_hook(
                    layer_idx=l, query_idx=query_idx, key_mask=key_mask,
                    query_pool=query_pool, out_per_head=out_per_head,
                ),
                with_kwargs=True,
            )
            for l, mod in enumerate(attn_modules)
        ]

        try:
            # Crucially: no output_attentions=True. The model runs SDPA
            # (or whatever low-memory attention impl it was loaded with)
            # and we ignore its attention output entirely. Our hooks have
            # already done the work before each attention block runs.
            _ = model(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if len(out_per_head) != n_layers:
            raise RuntimeError(
                f"Captured {len(out_per_head)}/{n_layers} layers at step "
                f"{step_idx} of {traj.filename}; hook misfire?"
            )

        per_head  = torch.stack(
            [out_per_head[l] for l in range(n_layers)], dim=0,
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

        # Empty caches on every device — multi-GPU dispatch leaves caches
        # on devices we're not currently working with.
        # if torch.cuda.is_available():
        #     for d in range(torch.cuda.device_count()):
        #         with torch.cuda.device(d):
        #             torch.cuda.empty_cache()

    return flat


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract aggregated attention mass via streaming Q/K hooks.",
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
        "--context", choices=["dependency", "all"], default="dependency",
        help="Context selection strategy for hand-crafted trajectories.",
    )
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

    model_path = MODELS[args.model]
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Note: NOT loading with attn_implementation="eager". SDPA (default)
    # is memory-efficient — the model never materialises (N, N). Our
    # hooks run before each attention forward and do their own
    # computation; we don't depend on HF's attention output.
    print(f"Loading model -> {args.device} ({args.dtype}, SDPA)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # attn_implementation="eager",
    )
    model.eval()

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
        "context":    args.context,
        "query_pool": args.query_pool,
        "dtype":      args.dtype,
        "impl":       "qk_streaming",
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
            context=args.context,
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