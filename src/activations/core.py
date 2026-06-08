"""
attribscope/reps/hidden.py — Hidden-state extraction along the residual stream.

Public API
----------
layer_to_shorthand(layer_idx)         tuple index → shorthand ("embed" or "act/{i}")
shorthand_to_layer(shorthand)         shorthand → tuple index
all_shorthands(n_layers)              complete ordered list of valid shorthands
pool_last(h, ctx_len)                 last output token → (hidden_dim,)
pool_mean(h, ctx_len)                 mean over output tokens → (hidden_dim,)
extract_hidden(...)                   forward pass; returns pooled residual states

Shorthand scheme
----------------
HuggingFace returns hidden_states as a tuple of (num_hidden_layers + 1) tensors.
We name them as follows:

    Tuple index   Shorthand            Meaning
    ───────────   ───────────────────  ──────────────────────────────────────────
    0             "embed"              Embedding output (before any transformer block)
    1             "act/0"              Residual stream after transformer block 0
    2             "act/1"              Residual stream after transformer block 1
    ...
    N             "act/{N-1}"          Residual stream after transformer block N-1
    (derived)     "act/{N-1}_normed"   act/{N-1} passed through model.model.norm

All raw residuals are stored pre-norm.  The normed variant is only produced for
the final transformer block and must be requested explicitly (or via "all").

Example for a 36-layer model (N=36):
    all_shorthands(36) = ["embed", "act/0", "act/1", ..., "act/35", "act/35_normed"]
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel


# ─────────────────────────────────────────────────────────────────────────────
# Shorthand ↔ tuple-index helpers
# ─────────────────────────────────────────────────────────────────────────────

def layer_to_shorthand(layer_idx: int) -> str:
    """Convert a hidden_states tuple index to its shorthand.

    Parameters
    ----------
    layer_idx : int
        0 for the embedding layer; i ≥ 1 for the residual stream after
        transformer block i-1.

    Examples
    --------
    >>> layer_to_shorthand(0)
    'embed'
    >>> layer_to_shorthand(1)
    'act/0'
    >>> layer_to_shorthand(36)
    'act/35'
    """
    if layer_idx == 0:
        return "embed"
    return f"act/{layer_idx - 1}"


def shorthand_to_layer(shorthand: str) -> int:
    """Convert a shorthand back to its hidden_states tuple index.

    The "_normed" suffix is stripped before conversion; callers that need to
    distinguish normed from raw should check for the suffix themselves.

    Examples
    --------
    >>> shorthand_to_layer("embed")
    0
    >>> shorthand_to_layer("act/0")
    1
    >>> shorthand_to_layer("act/35")
    36
    >>> shorthand_to_layer("act/35_normed")
    36
    """
    base = shorthand.removesuffix("_normed")

    if base == "embed":
        return 0
    if base.startswith("act/"):
        try:
            block_idx = int(base[4:])   # "act/35" → 35
        except ValueError:
            pass
        else:
            return block_idx + 1        # tuple index = block index + 1
    raise ValueError(
        f"Unknown shorthand '{shorthand}'. "
        "Expected 'embed', 'act/{{i}}', or 'act/{{i}}_normed'."
    )


def all_shorthands(n_layers: int) -> list[str]:
    """Return the complete ordered list of valid shorthands for a model with
    `n_layers` transformer blocks.

    Examples
    --------
    >>> all_shorthands(2)
    ['embed', 'act/0', 'act/1', 'act/1_normed']
    """
    names = ["embed"]
    for i in range(n_layers):
        names.append(f"act/{i}")
    names.append(f"act/{n_layers - 1}_normed")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Pooling strategies
# ─────────────────────────────────────────────────────────────────────────────

def pool_last(h: Tensor, ctx_len: int) -> Tensor:
    """Return the hidden state of the last output token.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context (input) tokens at the front of the sequence.
              Output tokens occupy positions [ctx_len:].
              Trailing padding is never present — we extract single samples
              without padding.

    Returns
    -------
    Tensor of shape (hidden_dim,), dtype float16, on CPU.
    """
    assert h.shape[0] > ctx_len, (
        f"No output tokens: seq_len={h.shape[0]}, ctx_len={ctx_len}."
    )
    # h[-1] is always the last real token since we never pad single samples.
    return h[-1].half().cpu()


def pool_mean(h: Tensor, ctx_len: int) -> Tensor:
    """Return the mean hidden state over all output tokens.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context tokens at the front.

    Returns
    -------
    Tensor of shape (hidden_dim,), dtype float16, on CPU.
    """
    assert h.shape[0] > ctx_len, (
        f"No output tokens: seq_len={h.shape[0]}, ctx_len={ctx_len}."
    )
    return h[ctx_len:].float().mean(dim=0).half().cpu()


def _apply_pool(
    h:       Tensor,
    ctx_len: int,
    pool:    Literal["last", "mean", "all"],
) -> dict[str, Tensor]:
    """Dispatch pooling and return {stat_name: Tensor}.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context tokens.
    pool    : "last" | "mean" | "all"

    Returns
    -------
    dict with a subset of keys {"last", "mean"}.
    """
    if pool == "last":
        return {"last": pool_last(h, ctx_len)}
    if pool == "mean":
        return {"mean": pool_mean(h, ctx_len)}
    if pool == "all":
        return {"last": pool_last(h, ctx_len), "mean": pool_mean(h, ctx_len)}
    raise ValueError(f"Unknown pool strategy '{pool}'. Choose from: last, mean, all.")


# ─────────────────────────────────────────────────────────────────────────────
# Final-norm helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_final_norm(model: PreTrainedModel):
    """Retrieve model.model.norm from a HuggingFace CausalLM.

    Valid for Qwen / Llama / Mistral style architectures where the transformer
    body lives at model.model and the final RMSNorm is model.model.norm.
    Raises RuntimeError with a descriptive message if the attribute is absent.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError(
            "model.model not found. This extractor assumes a HuggingFace "
            "CausalLM where the transformer body is at model.model "
            "(Qwen / Llama / Mistral layout)."
        )
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise RuntimeError(
            "model.model.norm not found. Cannot apply the final RMSNorm. "
            "Check the model architecture."
        )
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_hidden(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor | None,
    ctx_len:        int,
    layers:         list[str] | Literal["all"],
    pool:           Literal["last", "mean", "all"],
) -> dict[str, Tensor]:
    """Run one forward pass and return pooled hidden states.

    Parameters
    ----------
    model          : HuggingFace CausalLM.
    input_ids      : (1, seq_len) token ids.
    attention_mask : (1, seq_len) or None.
    ctx_len        : number of context tokens; output tokens start at this position.
    layers         : list of shorthand strings to extract, or "all".
                     Valid shorthands: "embed", "act/{i}", "act/{last}_normed".
                     See all_shorthands(n_layers) for the complete list.
    pool           : "last" | "mean" | "all"

    Returns
    -------
    Flat dict mapping "{stat}.{shorthand}" → float16 Tensor on CPU.

    Key examples (36-layer model, pool="all")
    ------------------------------------------
        "last.embed"             embedding, last output token
        "mean.act/0"             after block 0, mean over output tokens
        "last.act/35"            after block 35 (raw residual), last output token
        "mean.act/35_normed"     after block 35 + final RMSNorm, mean
    """
    was_training = model.training
    model.eval()

    n_layers  = model.config.num_hidden_layers
    valid     = set(all_shorthands(n_layers))
    normed_sh = f"act/{n_layers - 1}_normed"   # e.g. "act/35_normed"

    # ── Resolve wanted shorthands ─────────────────────────────────────────────
    if layers == "all":
        wanted: set[str] = valid
    else:
        bad = [sh for sh in layers if sh not in valid]
        if bad:
            raise ValueError(
                f"Unknown shorthand(s): {bad}. "
                f"Valid options for this model: {sorted(valid)}"
            )
        wanted = set(layers)

    # The normed shorthand shares a tuple index with act/{N-1} but needs
    # a separate pass through the final norm, so track it apart.
    need_normed = normed_sh in wanted
    raw_wanted  = wanted - {normed_sh}

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states   # tuple of length n_layers + 1

    # ── Pool and collect raw residuals ────────────────────────────────────────
    result: dict[str, Tensor] = {}

    for shorthand in raw_wanted:
        tuple_idx = shorthand_to_layer(shorthand)
        h = hidden_states[tuple_idx][0].float()   # (seq_len, hidden_dim)
        for stat, vec in _apply_pool(h, ctx_len, pool).items():
            result[f"{stat}.{shorthand}"] = vec

    # ── Normed variant ────────────────────────────────────────────────────────
    if need_normed:
        final_norm = _get_final_norm(model)
        tuple_idx  = shorthand_to_layer(normed_sh)   # == n_layers
        with torch.no_grad():
            raw      = hidden_states[tuple_idx][0]               # (seq_len, hidden_dim)
            h_normed = final_norm(raw.unsqueeze(0))[0].float()
        for stat, vec in _apply_pool(h_normed, ctx_len, pool).items():
            result[f"{stat}.{normed_sh}"] = vec

    if was_training:
        model.train()

    return result