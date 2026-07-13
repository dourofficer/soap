"""src/data_v2/context.py — corrected chat-template concatenation + step spans.

Rewrite of :mod:`src.data.context` that fixes two boundary bugs found by
``verify_context`` (both from deriving boundaries via
``apply_chat_template(add_generation_prompt=True)``, whose assistant scaffolding
and reasoning ``<think>`` don't map to real content positions):

1. context-step spans shifted right by the assistant-scaffold length (all models);
2. DeepSeek-R1 ``ctx_len`` overshoot by a ``<think>`` that exists only in the
   generation prompt.

Design — assemble the sequence from independently-tokenised pieces so **every token
is assigned to exactly one step with no overlap**, for any tokenizer:

    head = open_ids ++ chunk_0 ++ sep ++ chunk_1 ++ … ++ close_ids
    input_ids = head ++ content_ids            # scored step appended, no trailing EOS
    ctx_len   = len(head)

``open_ids`` / ``close_ids`` are the template's own text before/after the user
content (found with a sentinel), so the render stays on-distribution: the generation
prompt puts reasoning models in "assistant responding" mode, and ``_ensure_empty_think``
gives DeepSeek an empty ``<think></think>`` block (Qwen3.5 emits one already). We tokenise
pieces separately rather than trust ``offset_mapping`` — which is unreliable on some fast
tokenizers (DeepSeek's is compressed/misaligned). ``src/data`` is left untouched.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizer

from .trajectory import Trajectory

# ─────────────────────────────────────────────────────────────────────────────
# Thinking-block handling (hackable)
# ─────────────────────────────────────────────────────────────────────────────

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Appended to close an unbalanced generation-prompt <think> as an empty block.
THINK_CLOSE_BLOCK = "\n" + THINK_CLOSE + "\n\n"

_SEP = "\n\n"                 # separator between serialized turns (matches _serialize_turns)
_SENTINEL = ""   # private-use marker to split the template's user scaffolding


def _ensure_empty_think(head_text: str) -> str:
    """Close an unbalanced generation-prompt ``<think>`` with an empty block.

    DeepSeek-R1 opens ``<think>`` in the generation prompt but never closes it, and its
    template *strips* ``<think>…</think>`` from assistant messages (so it can't be passed
    as content) — we close it empty here. Balanced (Qwen3.5 ``enable_thinking=False``) or
    think-less (Llama) prompts are returned unchanged.
    """
    if THINK_OPEN in head_text and THINK_CLOSE not in head_text:
        return head_text + THINK_CLOSE_BLOCK
    return head_text


# ─────────────────────────────────────────────────────────────────────────────
# Context selection / serialisation (copied from src.data.context)
# ─────────────────────────────────────────────────────────────────────────────

def select_context(history: list[dict], step_idx: int) -> list[int]:
    """Indices of history turns used as context for ``step_idx`` (default: all earlier)."""
    return list(range(step_idx))


def _serialize_turns(history: list[dict], indices: list[int]) -> str:
    """Flatten turns into ``[role] - Step i: content`` blocks joined by a blank line."""
    parts: list[str] = []
    for i in indices:
        turn = history[i]
        role = turn.get("role", f"turn_{i}")
        content = turn.get("content", "").strip()
        parts.append(f"[{role}] - Step {i}: {content}")
    return _SEP.join(parts)


def iter_scoreable_steps(trajectory: Trajectory) -> list[int]:
    """Scoreable step indices (skips the initial ``human`` question for hand-crafted)."""
    if trajectory.history[0]["role"] == "human":
        return list(range(1, len(trajectory.history)))
    return list(range(len(trajectory.history)))


# ─────────────────────────────────────────────────────────────────────────────
# Template scaffolding
# ─────────────────────────────────────────────────────────────────────────────

def _scaffold(tokenizer: PreTrainedTokenizer, tk: dict) -> tuple[str, str]:
    """Return ``(opening_text, closing_text)`` — the template text before / after the
    user content, with an empty-think block ensured in the closing for reasoning models.

    Rendered with a sentinel as the user content, then split on it: everything before is
    the user-turn opening; everything after is the user close + assistant opener (+ empty
    think). Both are pure template text (no content), so tokenising them is stable.
    """
    tmpl = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        tokenize=False, add_generation_prompt=True, **tk,
    )
    tmpl = _ensure_empty_think(tmpl)
    parts = tmpl.split(_SENTINEL)
    if len(parts) != 2:
        raise RuntimeError(
            f"sentinel did not round-trip through the chat template (got {len(parts)} parts) "
            f"— pick a different _SENTINEL for {tokenizer.name_or_path!r}"
        )
    return parts[0], parts[1]


def _ids(tokenizer: PreTrainedTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# Core build
# ─────────────────────────────────────────────────────────────────────────────

def _build(
    traj: Trajectory,
    step_idx: int,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int | None,
    template_kwargs: dict | None,
) -> dict[str, Any]:
    """Assemble ``input_ids`` from tokenised pieces. Returns
    ``{input_ids: list[int], ctx_len, step_tokens, hard_truncated}``."""
    history = traj.history
    tk = template_kwargs or {}
    opening_text, closing_text = _scaffold(tokenizer, tk)
    open_ids = _ids(tokenizer, opening_text)
    close_ids = _ids(tokenizer, closing_text)
    sep_ids = _ids(tokenizer, _SEP)
    content_ids = _ids(tokenizer, _serialize_turns(history, [step_idx]))

    def assemble(indices: list[int]) -> tuple[list[int], dict[int, list[int]]]:
        head = list(open_ids)
        spans: dict[int, list[int]] = {}
        for k, i in enumerate(indices):
            if k > 0:
                head += sep_ids
            chunk = _ids(tokenizer, _serialize_turns(history, [i]))
            spans[i] = list(range(len(head), len(head) + len(chunk)))
            head += chunk
        head += close_ids
        return head, spans

    ctx_indices = select_context(history, step_idx)
    head_ids, step_tokens = assemble(ctx_indices)

    # ── Truncation: drop oldest context turns until it fits ──────────────────
    if max_tokens is not None:
        while len(head_ids) + len(content_ids) > max_tokens and ctx_indices:
            ctx_indices = ctx_indices[1:]
            head_ids, step_tokens = assemble(ctx_indices)
        if len(head_ids) + len(content_ids) > max_tokens:
            # content alone overflows: hard front-slice, keep the last max_tokens
            input_ids = (head_ids + content_ids)[-max_tokens:]
            ctx_len = max(0, len(input_ids) - len(content_ids))
            return {
                "input_ids": input_ids,
                "ctx_len": ctx_len,
                "step_tokens": {step_idx: list(range(ctx_len, len(input_ids)))},
                "hard_truncated": True,
            }

    ctx_len = len(head_ids)
    input_ids = head_ids + content_ids
    step_tokens[step_idx] = list(range(ctx_len, len(input_ids)))
    return {
        "input_ids": input_ids,
        "ctx_len": ctx_len,
        "step_tokens": step_tokens,
        "hard_truncated": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public builders (same interface as src.data.context)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    traj: Trajectory,
    step_idx: int,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int | None = None,
    template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Tokenise one (context, scored-step) pair for activation/GradNorm scoring.

    Returns ``{"input_ids": (1, L) LongTensor, "ctx_len": int}``. ``[ctx_len:]`` is exactly
    the scored step's content (no trailing EOS)."""
    b = _build(traj, step_idx, tokenizer, max_tokens, template_kwargs)
    ids = torch.tensor(b["input_ids"], dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "ctx_len": b["ctx_len"]}


def separate_steps(
    traj: Trajectory,
    step_idx: int,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int | None = None,
    template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Like :func:`build_context` but also returns ``step_tokens``: each history step index
    → its token positions in ``input_ids`` (scored step = ``range(ctx_len, L)``; context
    steps exact and non-overlapping by construction)."""
    b = _build(traj, step_idx, tokenizer, max_tokens, template_kwargs)
    ids = torch.tensor(b["input_ids"], dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "ctx_len": b["ctx_len"], "step_tokens": b["step_tokens"]}
