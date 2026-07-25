"""Chat-template concatenation with exact per-step token spans.

Assembles the sequence from independently-tokenised pieces so **every token is
assigned to exactly one step with no overlap**, for any tokenizer:

    head = open_ids ++ chunk_0 ++ sep ++ chunk_1 ++ ... ++ close_ids
    input_ids = head ++ content_ids            # scored step appended, no trailing EOS
    ctx_len   = len(head)

open_ids / close_ids are the template's own text before/after the user content
(found via a sentinel), so the render stays on-distribution; ``_ensure_empty_think``
gives DeepSeek an empty <think></think> block, so its ctx_len is not inflated by a
generation-prompt <think> that occupies no real content position.

This module is data-only; import and call from an extractor:
    from src.data.context import separate_steps, iter_scoreable_steps
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizer

from .trajectory import Trajectory

# ── thinking-block handling ─────────────────────────────────────────────────
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_CLOSE_BLOCK = "\n" + THINK_CLOSE + "\n\n"

_SEP = "\n\n"                 # separator between serialized turns
_SENTINEL = ""         # private-use marker to split the user scaffolding


def _ensure_empty_think(head_text: str) -> str:
    """Close an unbalanced generation-prompt <think> with an empty block."""
    if THINK_OPEN in head_text and THINK_CLOSE not in head_text:
        return head_text + THINK_CLOSE_BLOCK
    return head_text


# ── context selection / serialisation ───────────────────────────────────────
def select_context(history: list[dict], step_idx: int) -> list[int]:
    """Indices of history turns used as context for step_idx (default: all earlier)."""
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


# ── template scaffolding ────────────────────────────────────────────────────
def _scaffold(tokenizer: PreTrainedTokenizer, tk: dict) -> tuple[str, str]:
    """Return (opening_text, closing_text): template text before / after user content."""
    tmpl = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        tokenize=False, add_generation_prompt=True, **tk,
    )
    tmpl = _ensure_empty_think(tmpl)
    parts = tmpl.split(_SENTINEL)
    if len(parts) != 2:
        raise RuntimeError(
            f"sentinel did not round-trip through the chat template (got {len(parts)} parts) "
            f"- pick a different _SENTINEL for {tokenizer.name_or_path!r}"
        )
    return parts[0], parts[1]


def _ids(tokenizer: PreTrainedTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ── core build ──────────────────────────────────────────────────────────────
def _build(traj, step_idx, tokenizer, max_tokens, template_kwargs) -> dict[str, Any]:
    """Assemble input_ids from independently-tokenised pieces.

    Returns ``{input_ids, ctx_len, step_tokens, hard_truncated}``.

    WHY ASSEMBLE RATHER THAN TOKENISE THE WHOLE STRING
    --------------------------------------------------
    Two downstream stages need to know exactly which token positions belong to which
    step: activation pooling averages over the SCORED step's tokens, and attention
    extraction sums query attention into per-PREDECESSOR-step buckets. If those spans
    are off by even a few tokens the representation blends neighbouring steps and the
    attention mass is attributed to the wrong predecessor.

    Deriving spans by re-rendering growing prefixes of the chat template (the obvious
    approach) is wrong in two ways that are easy to miss: the template's assistant
    scaffolding shifts every context span to the right, and reasoning models inject a
    generation-prompt ``<think>`` that occupies no real content position. Trusting
    ``offset_mapping`` is also unsafe — some fast tokenizers report compressed or
    misaligned offsets.

    So we tokenise each piece ON ITS OWN and concatenate ids:

        head      = open_ids ++ chunk_0 ++ sep ++ chunk_1 ++ ... ++ close_ids
        input_ids = head ++ content_ids
        ctx_len   = len(head)

    Every token then belongs to exactly one step by construction, with no overlap, for
    any tokenizer. ``open_ids``/``close_ids`` are the template's own text either side of
    the user content (recovered with a sentinel), so the sequence still renders
    on-distribution — the generation prompt puts the model in "assistant is responding"
    mode, which is the state under which we want to read its internals.

    Note the scored step's content is appended WITHOUT a trailing EOS: we are reading
    the model's state while producing the step, not scoring an end-of-turn decision.
    """
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

    # ── Truncation ───────────────────────────────────────────────────────────
    # Long trajectories (hand-crafted runs reach ~130 steps) blow past the context
    # window. We drop the OLDEST context turns one at a time and re-assemble, which
    # keeps the scored step and its immediate predecessors — the steps the attention
    # weights care about — intact. Re-assembling each time (rather than slicing ids)
    # is what keeps step_tokens exact after every drop.
    #
    # Caveat worth knowing when interpreting long-trajectory results: the task question
    # is turn 0, so it is the FIRST thing dropped. `select_context` is the hook to
    # change that policy (e.g. pin turn 0 and drop from the middle instead).
    if max_tokens is not None:
        while len(head_ids) + len(content_ids) > max_tokens and ctx_indices:
            ctx_indices = ctx_indices[1:]
            head_ids, step_tokens = assemble(ctx_indices)
        if len(head_ids) + len(content_ids) > max_tokens:
            # Degenerate case: the step's own content exceeds the budget with no context
            # left to drop. Keep the tail so the step's END (which last-token pooling
            # reads) survives; ctx_len collapses to whatever prefix remains.
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


# ── public builders ─────────────────────────────────────────────────────────
def build_context(traj, step_idx, tokenizer, max_tokens=None, template_kwargs=None):
    """Tokenise one (context, scored-step) pair. Returns {input_ids (1,L), ctx_len}."""
    b = _build(traj, step_idx, tokenizer, max_tokens, template_kwargs)
    ids = torch.tensor(b["input_ids"], dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "ctx_len": b["ctx_len"]}


def separate_steps(traj, step_idx, tokenizer, max_tokens=None, template_kwargs=None):
    """Like build_context but also returns step_tokens: history step -> token positions."""
    b = _build(traj, step_idx, tokenizer, max_tokens, template_kwargs)
    ids = torch.tensor(b["input_ids"], dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "ctx_len": b["ctx_len"], "step_tokens": b["step_tokens"]}
