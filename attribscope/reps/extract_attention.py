"""attribscope/reps/extract_attention.py

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
m^(l,h)_{i,t} lies in [0, 1] and sum_i m^(l,h)_{i,t} <= 1 — the gap is
mass routed to BOS, chat-template tokens, the scored step's own tokens
(causally accessible from later positions within T_t), etc.

Per-step output keys
--------------------
    "{step_idx}.raw_attn"            : (L, n_ctx)     float32
        m^(l)_{i,t} averaged over heads. Plug-compatible with
        `process_weighting` via sim="raw_attn".
    "{step_idx}.raw_attn_per_head"   : (L, H, n_ctx)  float32
        m^(l,h)_{i,t} before head reduction. Use for head-selection
        ablations: `t.mean(dim=1)` recovers `raw_attn`.
    "{step_idx}.attn_residual_mass"  : (L,)           float32
        Per-layer 1 - sum_i m^(l)_{i,t}, averaged over heads. The
        fraction of step-t attention NOT captured by the predecessor
        context (sinks, headers, self). Useful for diagnosing whether
        the signal is "spread" or "focused".
    "{step_idx}.ctx_indices"         : (n_ctx,)       int64
        Step indices of predecessors, in matching order.

Note on layer indexing
----------------------
Unlike `extract_weights.py` which stores (L+1, n_ctx) with layer 0 = the
embedding, attention only exists inside transformer blocks. Layer 0 here
is `model.model.layers[0].self_attn` (the first block). When using
`resolve_layer` from reweight.py with an "act/k" shorthand, the matching
attention index is exactly k (no off-by-one to the embedding slot).

Implementation notes
--------------------
1. Loads the model with `attn_implementation="eager"`. SDPA and Flash
   don't materialise the post-softmax attention probabilities our hooks
   depend on. Expect ~1.5–3x slowdown vs SDPA.
2. Registers one forward hook per attention module. Each hook reduces
   the (1, H, N, N) attn tensor on the fly to (H, n_ctx) using a
   precomputed (n_ctx, N) one-hot key-mask, then replaces the returned
   weights with None — this stops the outer model from accumulating
   the full `outputs.attentions` tuple (~4 GB/layer at N=8192, H=32, bf16).
3. Peak attention memory is one layer at a time.

Design knobs (CLI flags)
------------------------
--query-pool  {mean, last}
    How to aggregate over query tokens in T_t. "mean" matches the
    bilinearity argument of the dot-product PoC. "last" uses only
    the final token of T_t (the EOS-of-turn / "decision" token).

Head aggregation is performed at save time as a simple mean; the
per-head tensor is also saved so downstream code can re-aggregate
(e.g. drop sink-heavy heads) without re-extracting.

CLI example
-----------
    python -m attribscope.reps.extract_attention \\
        --model        llama-3.1-8b \\
        --subset       hand-crafted \\
        --input        data/ww \\
        --output-root  outputs/weighting_attn \\
        --max_tokens   8192 \\
        --context      dependency \\
        --query-pool   mean \\
        --device       cuda \\
        --dtype        bfloat16
"""
from __future__ import annotations

import argparse
import json
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
# Per-step attention reduction
# ──────────────────────────────────────────────────────────────────────────

def _build_key_mask(
    ctx_step_ids: list[int],
    step_tokens:  dict[int, list[int]],
    seq_len:      int,
    device:       torch.device,
) -> Tensor:
    """Build the (n_ctx, N) one-hot mask used to sum-over-T_i in one matmul.

    Row j marks the token positions in `step_tokens[ctx_step_ids[j]]` as 1.
    Used as `A_t @ M.T` in the hook, which contracts key positions in each
    predecessor step into a single scalar per (head, predecessor).

    Dense (rather than sparse) because n_ctx x N is at most ~50 x 8192 ≈
    1.6 MB in float32, and dense @ dense is faster than sparse @ dense
    at these sizes.
    """
    n_ctx = len(ctx_step_ids)
    M = torch.zeros(n_ctx, seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M


def _make_attn_hook(
    layer_idx:    int,
    query_idx:    Tensor,           # (q_len,) long on device — positions in T_t
    key_mask:     Tensor,           # (n_ctx, N) float on device
    query_pool:   str,              # "mean" | "last"
    out_per_head: dict[int, Tensor],
):
    """Forward hook that reduces an attention layer's (1, H, N, N) probs
    on the fly to a (H, n_ctx) summary.

    HF's attention modules return `(attn_output, attn_weights, ...)`; when
    `output_attentions=True` is set in the outer model call, attn_weights
    has shape `(B=1, H_q, N, N)` post-softmax. For GQA models (Llama-3.1,
    Qwen3), KV heads are already replicated up to H_q inside the eager
    implementation, so this shape matches plain MHA.

    The hook captures the small reduction, then returns a new tuple with
    the attn_weights slot replaced by `None`. Without this, the outer
    model would accumulate a tuple of (1, H, N, N) tensors across all
    layers — for N=8192, H=32, L=32, bf16 that's ~130 GB and would OOM
    immediately.
    """
    def hook(module, args, output):
        # Be defensive about HF's tuple return format — has been 2-tuple
        # vs 3-tuple in different transformers versions. We only need
        # output[0] (attn_output) and output[1] (attn_weights); anything
        # beyond [2:] is passed through unchanged.
        if not isinstance(output, tuple) or len(output) < 2:
            return output
        attn = output[1]
        if attn is None:
            return output

        # attn: (1, H, N, N) post-softmax probs. Drop batch dim.
        A = attn[0]

        # Multi-GPU: A may live on a different device than the precomputed
        # query_idx / key_mask. .to(...) is a no-op when already aligned.
        qi = query_idx.to(A.device, non_blocking=True)
        km = key_mask .to(A.device, non_blocking=True)

        # Aggregate over query positions in T_t. Upcast to float32 before
        # averaging — bf16 accumulation can drift on long sequences.
        if query_pool == "mean":
            # Mean over step-t query tokens. Analogue of the mean-pool
            # in the dot-product PoC, justified by row-stochasticity.
            A_t = A.index_select(1, qi).float().mean(dim=1)   # (H, N)
        elif query_pool == "last":
            # Just the final token of T_t — "decision token" interp.
            # For chat-template inputs this is usually the <|eot_id|>
            # (or equivalent) at the end of the assistant message.
            A_t = A[:, qi[-1], :].float()                     # (H, N)
        else:
            raise ValueError(f"Unknown query_pool={query_pool!r}")

        # Sum over key positions in each predecessor step in a single
        # matmul, vectorised across heads and predecessors:
        #     m[h, i] = sum_{q in T_i} A_t[h, q]
        m = A_t @ km.T                                          # (H, n_ctx)
        out_per_head[layer_idx] = m.cpu()

        # Replace attn_weights with None so the outer model doesn't keep
        # this layer's (1, H, N, N) tensor alive for the rest of the pass.
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
    context:     str,                         # "dependency" | "all"
    query_pool:  str          = "mean",       # "mean" | "last"
    pbar:        tqdm | None  = None,
) -> dict[str, Tensor]:
    """Per scored step: one forward pass with hooks installed, capturing
    aggregated attention mass to every in-graph predecessor step.

    Returns a flat dict of safetensors-ready tensors keyed by
    "{step_idx}.{name}" for the whole trajectory.
    """
    flat: dict[str, Tensor] = {}
    device = next(model.parameters()).device

    # Assumes Llama / Qwen / Mistral-style layout. Swap this list if you
    # adapt the extractor to a different architecture.
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
            # Earliest scoreable step under "dependency" strategy may
            # have no in-graph predecessors. Nothing to discount against.
            continue

        seq_len = input_ids.shape[1]
        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file",     traj.filename),
                ("seq_len",  seq_len),
                ("step_idx", step_idx),
                ("n_steps",  len(traj.history)),
            ]))

        # Precompute reduction inputs once per step.
        query_idx = torch.tensor(step_tokens[step_idx], device=device,
                                 dtype=torch.long)                   # (q_len,)
        key_mask  = _build_key_mask(ctx_step_ids, step_tokens,
                                    seq_len, device)                 # (n_ctx, N)

        # Install hooks on every attention module. Each hook closes over
        # its own layer_idx and a fresh `out_per_head` dict — this means
        # the hooks must be re-created every step, which costs nothing.
        out_per_head: dict[int, Tensor] = {}
        handles = [
            mod.register_forward_hook(
                _make_attn_hook(
                    layer_idx=l, query_idx=query_idx, key_mask=key_mask,
                    query_pool=query_pool, out_per_head=out_per_head,
                ),
            )
            for l, mod in enumerate(attn_modules)
        ]

        try:
            # `output_attentions=True` makes the attention modules actually
            # return probabilities to our hooks. We ignore the returned
            # `outputs.attentions` (it's a tuple of Nones after our hooks
            # have stripped each layer's weights).
            _ = model(input_ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        # Sanity-check: every layer's hook should have fired exactly once.
        if len(out_per_head) != n_layers:
            raise RuntimeError(
                f"Captured {len(out_per_head)}/{n_layers} layers at step "
                f"{step_idx} of {traj.filename}; hook misfire?"
            )

        # Stack per-layer summaries → (L, H, n_ctx). Iterate by index to
        # guard against dict-iteration-order surprises.
        per_head = torch.stack(
            [out_per_head[l] for l in range(n_layers)], dim=0,
        ).float().contiguous()                                       # (L, H, n_ctx)

        # Derived views.
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
        description="Extract aggregated attention-mass step-to-step weights.",
    )
    p.add_argument("--model",       required=True, choices=list(MODELS))
    p.add_argument("--subset",      default=None)
    p.add_argument("--input",       required=True, help="Dataset directory.")
    p.add_argument("--output-root", required=True, help="Output directory for .safetensors files.")
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
        help=(
            "Aggregation over query tokens in T_t. "
            "'mean' averages attention from every step-t token (the analogue "
            "of the mean-pool in the dot-product PoC); 'last' uses only the "
            "final token of T_t."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────
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

    # ── Model + tokenizer ─────────────────────────────────────────────────
    model_path = MODELS[args.model]
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # `attn_implementation="eager"` is mandatory: SDPA and FlashAttention
    # don't materialise the post-softmax probabilities our hooks read.
    print(f"Loading model → {args.device} ({args.dtype}, eager attention)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────────
    trajectories = load_dataset(args.input, subset=args.subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"Processing {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    out_root = Path(args.output_root) / args.model / args.subset
    out_root.mkdir(parents=True, exist_ok=True)

    # Persist the design knobs alongside the data so downstream sweeps
    # know which extraction variant they're reading.
    (out_root / "config.json").write_text(json.dumps({
        "model":      args.model,
        "subset":     args.subset,
        "max_tokens": args.max_tokens,
        "context":    args.context,
        "query_pool": args.query_pool,
        "dtype":      args.dtype,
    }, indent=2))

    # ── Extract ───────────────────────────────────────────────────────────
    pbar = tqdm(trajectories, desc="Processing trajectories")
    for traj in pbar:
        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))
        out_path = out_root / f"{Path(traj.filename).stem}.safetensors"
        if out_path.exists():
            continue

        flat = extract_trajectory_attention(
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