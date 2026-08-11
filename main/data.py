"""Trajectories, and the per-step model input with exact token spans.

Two downstream stages need to know precisely which token positions belong to which
step: activation pooling averages over the SCORED step's tokens, and attention
extraction sums query attention into per-PREDECESSOR-step buckets. If those spans are
off by even a few tokens, the representation blends neighbouring steps and attention
mass is attributed to the wrong predecessor.

So the sequence is assembled from INDEPENDENTLY TOKENISED pieces and the ids are
concatenated:

    head      = open_ids ++ chunk_0 ++ sep ++ chunk_1 ++ ... ++ close_ids
    input_ids = head ++ content_ids            (the scored step, no trailing EOS)
    ctx_len   = len(head)

Every token then belongs to exactly one step by construction, for any tokenizer. The
obvious alternative — re-rendering growing prefixes of the chat template — is wrong in
two ways that are easy to miss: the assistant scaffolding shifts every context span to
the right, and reasoning models inject a generation-prompt ``<think>`` occupying no real
content position. ``offset_mapping`` is also unsafe; some fast tokenizers report
compressed or misaligned offsets.

``open_ids``/``close_ids`` are the template's own text either side of the user content
(recovered with a sentinel), so the sequence still renders on-distribution: the
generation prompt puts the model in "assistant is responding" mode, which is the state
we want to read. The scored step's content is appended WITHOUT a trailing EOS — we are
reading the model while it produces the step, not scoring an end-of-turn decision.

    from main.data import load_dataset, iter_scoreable_steps, build_step_input
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# ── thinking-block handling ─────────────────────────────────────────────────
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_CLOSE_BLOCK = "\n" + THINK_CLOSE + "\n\n"

_SEP = "\n\n"                 # separator between serialized turns

# Sentinel context-bucket id for the pinned GT block in with-GT mode. It labels the
# block's token span in ``step_tokens`` so attention buckets it as a predecessor, but it
# is never a scoreable step: no activation entry is written under it, and
# ``main.rescore.build_W`` drops it (no keeper row) exactly like the unscored ``human``
# question turn of hand-crafted trajectories.
GT_STEP = -1
_SENTINEL = ""          # private-use marker to split the user scaffolding


# ── trajectories ────────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    """One failure instance (a Who&When / CORRECT / TraceElephant-style trace)."""
    filename:      str
    question_id:   str
    history:       list[dict]     # ordered turns; step t == history[t], 0-indexed
    mistake_agent: str
    mistake_step:  int            # 0-indexed
    level:         int
    subset:        str
    question:      str
    system:        str | None
    ground_truth:  str = ""       # gold task answer; empty on corpora without one


def _sorted_json_files(directory: Path) -> list[str]:
    """JSON filenames sorted numerically by their digits (1.json, 2.json, ...)."""
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    return sorted(files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))


def load_dataset(path: str | Path, subset: str) -> list[Trajectory]:
    """Load ``<path>/<subset>/*.json`` into Trajectories, numerically sorted."""
    root = Path(path) / subset
    trajectories: list[Trajectory] = []
    for filename in _sorted_json_files(root):
        item = json.loads((root / filename).read_text(encoding="utf-8"))
        system_description = None
        if subset == "algorithm-generated":
            prefix = "## Your role\n"
            system_description = "\n\n".join(
                f"{name}: {desc[len(prefix):].strip()}"
                for name, desc in item.get("system_prompt", {}).items()
            )
        trajectories.append(Trajectory(
            filename=filename,
            question_id=item["question_ID"],
            history=item["history"],
            mistake_agent=item["mistake_agent"],
            mistake_step=int(item["mistake_step"]),
            level=item.get("level", -1),
            subset=subset,
            question=item.get("question", ""),
            system=system_description,
            ground_truth=item.get("ground_truth") or item.get("groundtruth") or "",
        ))
    return trajectories


def extract_metadata(traj: Trajectory) -> dict:
    """Trajectory-metadata header written into every .safetensors payload."""
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
        "ground_truth":  traj.ground_truth,
    }


def iter_scoreable_steps(traj: Trajectory) -> list[int]:
    """Scoreable step indices (skips the initial ``human`` question for hand-crafted)."""
    if traj.history[0]["role"] == "human":
        return list(range(1, len(traj.history)))
    return list(range(len(traj.history)))


def select_context(history: list[dict], step_idx: int) -> list[int]:
    """Indices of history turns used as context for step_idx (default: all earlier)."""
    return list(range(step_idx))


# ── serialisation + template scaffolding ────────────────────────────────────
def _serialize_turns(history: list[dict], indices: list[int]) -> str:
    """Flatten turns into ``[role] - Step i: content`` blocks joined by a blank line."""
    parts: list[str] = []
    for i in indices:
        turn = history[i]
        role = turn.get("role", f"turn_{i}")
        content = turn.get("content", "").strip()
        parts.append(f"[{role}] - Step {i}: {content}")
    return _SEP.join(parts)


def _gt_block_text(traj: Trajectory) -> str:
    """The pinned with-GT prefix. Phrasing matches the prompting baselines so the proxy
    reads the answer in the same register; it contains no ``[role] - Step i:`` substring,
    so it cannot collide with a turn."""
    gt = (traj.ground_truth or "").strip()
    assert gt, f"GT mode requires a non-empty ground_truth ({traj.filename})"
    return f"The problem is: {traj.question}\nThe Answer for the problem is: {gt}"


def _ensure_empty_think(head_text: str) -> str:
    """Close an unbalanced generation-prompt ``<think>`` with an empty block.

    Without this, DeepSeek's ``ctx_len`` is inflated by a ``<think>`` that occupies no
    real content position.
    """
    if THINK_OPEN in head_text and THINK_CLOSE not in head_text:
        return head_text + THINK_CLOSE_BLOCK
    return head_text


def _scaffold(tokenizer, tk: dict) -> tuple[str, str]:
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
            f"- pick a different _SENTINEL for {tokenizer.name_or_path!r}")
    return parts[0], parts[1]


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ── the build ───────────────────────────────────────────────────────────────
def build_step_input(traj: Trajectory, step_idx: int, tokenizer, max_tokens: int | None = None,
                     template_kwargs: dict | None = None, with_gt: bool = False) -> dict[str, Any]:
    """Assemble one (context, scored-step) input.

    Returns ``{input_ids (1,L), ctx_len, step_tokens, hard_truncated}``, where
    ``step_tokens`` maps a history step index (plus ``GT_STEP`` in with-GT mode) to its
    token positions.

    With ``with_gt=True`` the context becomes ``[question, answer] + [s_1..s_t]``: the GT
    block sits right after ``open_ids``, is PINNED (the truncation loop only drops real
    context turns), and is recorded under ``GT_STEP`` so attention treats it as a
    predecessor bucket that is never scored. It is part of the head, so ``ctx_len`` covers
    it and pooling never reads its tokens.
    """
    history = traj.history
    tk = template_kwargs or {}
    opening_text, closing_text = _scaffold(tokenizer, tk)
    open_ids = _ids(tokenizer, opening_text)
    close_ids = _ids(tokenizer, closing_text)
    sep_ids = _ids(tokenizer, _SEP)
    content_ids = _ids(tokenizer, _serialize_turns(history, [step_idx]))
    gt_ids = _ids(tokenizer, _gt_block_text(traj)) if with_gt else []

    def assemble(indices: list[int]) -> tuple[list[int], dict[int, list[int]]]:
        head = list(open_ids)
        spans: dict[int, list[int]] = {}
        if gt_ids:
            spans[GT_STEP] = list(range(len(head), len(head) + len(gt_ids)))
            head += gt_ids
        for k, i in enumerate(indices):
            if k > 0 or gt_ids:
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
    # window. Drop the OLDEST context turns one at a time and RE-ASSEMBLE, which keeps
    # the scored step and its immediate predecessors — the ones the attention weights
    # care about — intact. Re-assembling (rather than slicing ids) is what keeps
    # step_tokens exact after every drop.
    #
    # Worth knowing when interpreting long-trajectory results: the task question is turn
    # 0, so it is the FIRST thing dropped. ``select_context`` is the hook to change that.
    # In with-GT mode the pinned block is not in ctx_indices, so it survives every drop;
    # only the degenerate tail-keep below can cut into it (and then step_tokens holds
    # just the scored step, so no phantom GT bucket is emitted).
    if max_tokens is not None:
        while len(head_ids) + len(content_ids) > max_tokens and ctx_indices:
            ctx_indices = ctx_indices[1:]
            head_ids, step_tokens = assemble(ctx_indices)
        if len(head_ids) + len(content_ids) > max_tokens:
            # Degenerate: the step's own content exceeds the budget with no context left
            # to drop. Keep the tail so the step's END survives; ctx_len collapses to
            # whatever prefix remains.
            ids = (head_ids + content_ids)[-max_tokens:]
            ctx_len = max(0, len(ids) - len(content_ids))
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                "ctx_len": ctx_len,
                "step_tokens": {step_idx: list(range(ctx_len, len(ids)))},
                "hard_truncated": True,
            }

    ctx_len = len(head_ids)
    ids = head_ids + content_ids
    step_tokens[step_idx] = list(range(ctx_len, len(ids)))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        "ctx_len": ctx_len,
        "step_tokens": step_tokens,
        "hard_truncated": False,
    }
