from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .trajectory import Trajectory

from transformers import PreTrainedTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Context selection  ←  PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def select_context(history: list[dict], step_idx: int) -> list[int]:
    """Return the indices of history turns to use as context for step `step_idx`.

    **Default**: every turn strictly before step_idx, i.e. range(step_idx).

    This function is called inside :func:`build_context`.  
    Replace or monkey-patch it to implement.

    Parameters
    ----------
    history  : full trajectory history list.
    step_idx : the step being scored (0-indexed; never 0 itself).

    Returns
    -------
    list[int]
        Ordered indices into `history` to include as context.
        All indices must satisfy idx < step_idx.
    """
    return list(range(step_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_turns(history: list[dict], indices: list[int]) -> str:
    """Flatten selected turns into a single plain-text string.

    Format per turn:
        [<role>]: <content>

    Turns are separated by a blank line.  Roles are kept verbatim (e.g.
    "Orchestrator (thought)", "WebSurfer") so the model sees the full
    multi-agent structure.
    """
    parts: list[str] = []
    for i in indices:
        turn    = history[i]
        role    = turn.get("role", f"turn_{i}")
        content = turn.get("content", "").strip()
        parts.append(f"[{role}] - Step {i}: {content}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Context builders
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    traj:            Trajectory,
    step_idx:        int,
    tokenizer:       PreTrainedTokenizer,
    max_tokens:      int | None = None,
    template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Tokenise one (context, step) pair for GradNorm scoring.
 
    Layout fed into apply_chat_template
    ------------------------------------
 
        <user>
          [role_0]: content_0
 
          [role_1]: content_1
          ...                         ← context turns from select_context()
        </user>
        <assistant>
          content of history[step_idx] ← NTP loss is computed over these tokens
        </assistant>
 
    The context turns are serialised as plain text and placed in the user
    slot; the step content is placed verbatim in the assistant slot.
    apply_chat_template wraps both with model-specific special tokens.
 
    Parameters
    ----------
    history   : full trajectory history.
    step_idx  : step to score.  Must be ≥ 1 (step 0 is the human question).
    tokenizer : HuggingFace tokeniser with a chat template.
 
    Returns
    -------
    dict with:
        "input_ids" : LongTensor shape (1, seq_len)
        "ctx_len"   : int
            Number of tokens *before* the first step-content token.
            Used in :func:`gradnorm._ntp_loss` to mask context positions.
 
    Notes
    -----
    ctx_len is computed as the length of the user-turn prefix with
    ``add_generation_prompt=True``, which appends the assistant header tokens
    (e.g. ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` for Llama 3).
    This correctly accounts for any template-injected tokens surrounding the
    assistant response.
 
    Qwen3 note: Qwen3's chat template may prepend <think> tokens by default.
    Disable this by calling
        tokenizer.apply_chat_template(..., enable_thinking=False)
    or by patching the template variable before calling build_context.
    """
    history      = traj.history
    ctx_indices  = select_context(history, step_idx)
    step_content = _serialize_turns(history, [step_idx])
    assistant_msg = {"role": "assistant", "content": step_content}
    tk = template_kwargs or {}
 
    def _apply(indices: list[int]) -> tuple:
        """Tokenise [user_msg, assistant_msg] and the user-only prefix."""
        user_msg = {"role": "user", "content": _serialize_turns(history, indices)}
        full_ids = tokenizer.apply_chat_template(
            [user_msg, assistant_msg],
            tokenize              = True,
            add_generation_prompt = False,
            return_tensors        = "pt",
            **tk,
        )
        prefix_ids = tokenizer.apply_chat_template(
            [user_msg],
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
            **tk,
        )
        return full_ids, prefix_ids
 
    full_ids, prefix_ids = _apply(ctx_indices)
 
    # ── Truncate context if full sequence exceeds max_tokens ─────────────
    # Drop the oldest context turns one by one until the total fits.
    # The step content is always preserved; only ctx_indices shrinks.
    if max_tokens is not None:
        while (
            full_ids["input_ids"].shape[1] > max_tokens
            and len(ctx_indices) > 0
        ):
            ctx_indices = ctx_indices[1:]   # drop oldest turn
            full_ids, prefix_ids = _apply(ctx_indices)

        if full_ids["input_ids"].shape[1] > max_tokens:
            step_len = full_ids["input_ids"].shape[1] - prefix_ids["input_ids"].shape[1]
            full_ids["input_ids"] = full_ids["input_ids"][:, -max_tokens:]
            ctx_len = max(0, max_tokens - step_len)
            return {"input_ids": full_ids["input_ids"], "ctx_len": ctx_len}
 
    ctx_len = prefix_ids["input_ids"].shape[1]
    return {"input_ids": full_ids["input_ids"], "ctx_len": ctx_len}


def separate_steps(
    traj:            Trajectory,
    step_idx:        int,
    tokenizer:       PreTrainedTokenizer,
    max_tokens:      int | None = None,
    template_kwargs: dict | None = None,
) -> dict[str, Any]:

    history      = traj.history
    ctx_indices  = select_context(history, step_idx)
    step_content = _serialize_turns(history, [step_idx])
    assistant_msg = {"role": "assistant", "content": step_content}
    tk = template_kwargs or {}

    def _apply(indices: list[int]) -> tuple:
        """Tokenise [user_msg, assistant_msg] and the user-only prefix."""
        user_msg = {"role": "user", "content": _serialize_turns(history, indices)}
        full_ids = tokenizer.apply_chat_template(
            [user_msg, assistant_msg],
            tokenize              = True,
            add_generation_prompt = False,
            return_tensors        = "pt",
            **tk,
        )
        prefix_ids = tokenizer.apply_chat_template(
            [user_msg],
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
            **tk,
        )
        return full_ids, prefix_ids

    def _build_step_tokens(
        ctx_indices: list[int],
        ctx_len:     int,
        seq_len:     int,
    ) -> dict[int, list[int]]:
        """Map each step index to its token positions in full_ids.

        Context steps: found by progressive tokenization — the prefix grows
        one step at a time and the length delta gives the token span of each
        added step.

        Scored step (step_idx): always occupies [ctx_len, seq_len).
        """
        step_tokens: dict[int, list[int]] = {}

        # Compute prefix length after each cumulative prefix of ctx_indices.
        # prefix_lengths[k] = number of tokens in the prefix that contains
        # exactly the first k context steps.
        prefix_lengths = []
        for k in range(len(ctx_indices) + 1):
            _, partial_prefix = _apply(ctx_indices[:k])
            prefix_lengths.append(partial_prefix["input_ids"].shape[1])

        for k, idx in enumerate(ctx_indices):
            start = prefix_lengths[k]
            end   = prefix_lengths[k + 1]
            step_tokens[idx] = list(range(start, end))

        # The scored step always sits right after the context prefix.
        step_tokens[step_idx] = list(range(ctx_len, seq_len))

        return step_tokens

    full_ids, prefix_ids = _apply(ctx_indices)

    # ── Truncate context if full sequence exceeds max_tokens ─────────────
    if max_tokens is not None:
        while (
            full_ids["input_ids"].shape[1] > max_tokens
            and len(ctx_indices) > 0
        ):
            ctx_indices = ctx_indices[1:]   # drop oldest turn
            full_ids, prefix_ids = _apply(ctx_indices)

        if full_ids["input_ids"].shape[1] > max_tokens:
            # Hard truncation: step alone exceeds budget; slice from the front.
            # All ctx_indices have already been dropped, so step_tokens only
            # contains the scored step (no context step entries).
            step_len = full_ids["input_ids"].shape[1] - prefix_ids["input_ids"].shape[1]
            full_ids["input_ids"] = full_ids["input_ids"][:, -max_tokens:]
            ctx_len     = max(0, max_tokens - step_len)
            seq_len     = full_ids["input_ids"].shape[1]
            step_tokens = _build_step_tokens([], ctx_len, seq_len)
            return {
                "input_ids":   full_ids["input_ids"],
                "ctx_len":     ctx_len,
                "step_tokens": step_tokens,
            }

    ctx_len     = prefix_ids["input_ids"].shape[1]
    seq_len     = full_ids["input_ids"].shape[1]
    step_tokens = _build_step_tokens(ctx_indices, ctx_len, seq_len)

    return {
        "input_ids":   full_ids["input_ids"],
        "ctx_len":     ctx_len,
        "step_tokens": step_tokens,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def iter_scoreable_steps(trajectory: Trajectory) -> list[int]:
    """Return step indices that should receive a GradNorm score.

    Step 0 is the human question and is never a mistake step, so it is
    excluded.  Returns [1, 2, ..., T-1].
    """
    is_handcrafted = trajectory.history[0]['role'] == 'human'
    if is_handcrafted: return list(range(1, len(trajectory.history)))
    else:              return list(range(len(trajectory.history)))