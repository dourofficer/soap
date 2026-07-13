"""src/data/verify_context.py — verify & visualise step-concatenation for extraction.

Both extraction pipelines concatenate a trajectory's prior turns into one context
and run a forward pass per scoreable step, but they use *different* builders in
``src/data/context.py``:

* activations → :func:`build_context`   → ``input_ids``, ``ctx_len``
* attention   → :func:`separate_steps`  → those **plus** per-step token spans
  ``step_tokens`` (the ``T_t`` that ``build_key_mask`` turns into ``w_{i,t}``).

They are near-duplicate functions, and ``separate_steps`` derives each step's span
from progressive-tokenisation length deltas, which can silently drift a token at
chat-template / ``\\n\\n``-join boundaries. This module checks, tokenizer-only (no
model weights, no GPU), that:

  A. both builders feed the two pipelines *identical* inputs,
  B. the recorded per-step spans really point at each step's text,
  C. the sequence is a faithful chat-template render for the model,

and, in ``show`` mode, renders the concatenated tokens with step boundaries in
colour (ANSI, or self-contained HTML via ``--html``) for both the activation
pooled region and the attention spans.

Run from the repo root:

    python -m src.data.verify_context check --model <hf_id> --input data/ww \\
        --subset hand-crafted --limit 5 [--max-tokens 8192] [--verbose]

    python -m src.data.verify_context show  --model <hf_id> --input data/ww \\
        --subset hand-crafted --index 0 --step 5 [--html /tmp/ctx.html]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from ..models import get_adapter
from .trajectory import Trajectory, load_dataset
from .context import (
    build_context,
    separate_steps,
    select_context,
    iter_scoreable_steps,
    _serialize_turns,
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers (shared by both subcommands)
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer_and_kwargs(model: str) -> tuple[Any, dict]:
    """Mirror the extractors' tokenisation without loading model weights.

    ``get_adapter`` only reads ``config.model_type`` and ``template_kwargs`` is
    the sole chat-template variation (e.g. ``enable_thinking=False`` for Qwen3),
    so this reproduces exactly what ``build_context`` / ``separate_steps`` see in
    production while staying CPU-only.
    """
    adapter = get_adapter(model)
    tok = AutoTokenizer.from_pretrained(model)
    return tok, adapter.template_kwargs()


def resolve_dataset(input_path: str, subset: str | None) -> list[Trajectory]:
    """Same ``--input`` / ``--subset`` resolution as ``activations/extract.py``."""
    p = Path(input_path)
    if subset:
        base, sub = str(p), subset
    else:
        base, sub = str(p.parent), p.name
    return load_dataset(base, subset=sub)


def _norm(s: str) -> str:
    """Whitespace-insensitive form for robust substring matching across the
    ``\\n\\n`` joins and boundary token-merges the chat template introduces."""
    return "".join(s.split())


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_step(
    traj: Trajectory,
    step_idx: int,
    tok: Any,
    tk: dict,
    max_tokens: int | None,
) -> tuple[list[str], bool]:
    """Run Checks A/B/C for one (trajectory, step).

    Returns ``(fails, trailing_special)`` — a list of failure strings (empty ⇒ all
    passed) and whether the last pooled/queried token is a template special token.
    """
    fails: list[str] = []
    history = traj.history

    bc = build_context(traj, step_idx, tok, max_tokens, tk)
    ss = separate_steps(traj, step_idx, tok, max_tokens, tk)

    ids_bc = bc["input_ids"]
    ids = ss["input_ids"]
    ctx_len = ss["ctx_len"]
    seq_len = ids.shape[1]
    st = ss["step_tokens"]

    # Did max_tokens actually truncate this step? Detected robustly by comparing
    # against the untruncated build length (covers both the drop-oldest branch and
    # the hard front-slice, including a step-0 slice with no context). Only pay the
    # extra build when a budget is set.
    ctx_indices_full = select_context(history, step_idx)
    was_truncated = False
    if max_tokens is not None:
        untrunc_len = separate_steps(traj, step_idx, tok, None, tk)["input_ids"].shape[1]
        was_truncated = untrunc_len > max_tokens

    # C4 (informational): is the last pooled/queried token a template special token?
    trailing_special = int(ids[0, -1]) in set(tok.all_special_ids)

    # ── Check A — builder agreement ──────────────────────────────────────────
    if not torch.equal(ids_bc, ids):
        fails.append(
            f"A: build_context vs separate_steps input_ids differ "
            f"({tuple(ids_bc.shape)} vs {tuple(ids.shape)})"
        )
    if bc["ctx_len"] != ctx_len:
        fails.append(f"A: ctx_len differs ({bc['ctx_len']} vs {ctx_len})")

    # ── Check B — span alignment ─────────────────────────────────────────────
    # B1: scored step is exactly the post-context region.
    if st.get(step_idx) != list(range(ctx_len, seq_len)):
        fails.append(
            f"B1: step_tokens[{step_idx}] != range({ctx_len}, {seq_len})"
        )

    # B2: structural — contiguous, ordered, non-overlapping, ends at ctx_len.
    ctx_ids = [i for i in st if i != step_idx]
    ctx_ids_sorted = sorted(ctx_ids, key=lambda i: st[i][0] if st[i] else -1)
    prev_end: int | None = None
    for i in ctx_ids_sorted:
        span = st[i]
        if span != list(range(span[0], span[-1] + 1)):
            fails.append(f"B2: step {i} span not contiguous: {span[:3]}…")
            continue
        if prev_end is not None and span[0] != prev_end:
            fails.append(
                f"B2: gap/overlap before step {i} (starts {span[0]}, prev end {prev_end})"
            )
        prev_end = span[-1] + 1
    # Surviving context spans (a suffix of ctx_indices under drop-oldest) always
    # end at ctx_len and stay in select_context order, so these hold regardless of
    # truncation; only the hard-slice case (no context spans) skips via the guard.
    if ctx_ids_sorted:
        if prev_end != ctx_len:
            fails.append(f"B2: context spans end at {prev_end}, ctx_len={ctx_len}")
        if ctx_ids_sorted != [i for i in ctx_indices_full if i in st]:
            fails.append("B2: context span order != select_context order")

    # B3 + span-drift: each surviving context span must decode to that step's FULL
    # serialized turn ("[role] - Step i: content"), not just its raw content — the
    # latter would miss a span shifted by only the "[role] - Step i: " label (e.g.
    # the +5 assistant-scaffold shift on Llama). Survivors are whole turns, so this
    # holds even under truncation.
    for i in ctx_ids_sorted:
        decoded = tok.decode([ids[0, p] for p in st[i]])
        serial_i = _serialize_turns(history, [i])
        if serial_i and _norm(serial_i) not in _norm(decoded):
            fails.append(
                f"B3: step {i} span text mismatch — span drift?\n"
                f"      decoded span : {decoded[:120]!r}\n"
                f"      expected     : {serial_i[:120]!r}"
            )
    step_decoded = tok.decode(ids[0, ctx_len:], skip_special_tokens=True)
    step_serialized = _serialize_turns(history, [step_idx])   # "[role] - Step i: content"
    if step_serialized:
        ns, nd = _norm(step_serialized), _norm(step_decoded)
        # Full step present (untruncated, or context dropped but step fits): ns ⊆ nd.
        # Hard front-slice: the region holds only the suffix of the step: nd ⊆ ns.
        if not (ns in nd or nd in ns):
            fails.append(
                f"B3: scored step {step_idx} text mismatch in [ctx_len:] region\n"
                f"      decoded : {step_decoded[:120]!r}\n"
                f"      expected: {step_serialized[:120]!r}"
            )

    # ── Check C — chat-template faithfulness ─────────────────────────────────
    # Only meaningful for the untruncated render (truncation intentionally drops
    # context / slices the front, so the sequence is no longer the full template).
    if not was_truncated:
        user_msg = {"role": "user", "content": _serialize_turns(history, ctx_indices_full)}
        assistant_msg = {"role": "assistant", "content": _serialize_turns(history, [step_idx])}

        # C1: input_ids ARE the canonical template tokenization. Compared at the
        # token-id level (not via decode) — some fast tokenizers have a lossy
        # decode (e.g. DeepSeek renders ▁/U+2581 as a space and drops spaces), and
        # extraction never decodes anyway.
        ref_ids = tok.apply_chat_template(
            [user_msg, assistant_msg], tokenize=True,
            add_generation_prompt=False, return_tensors="pt", **tk,
        )["input_ids"]
        if not torch.equal(ids, ref_ids):
            fails.append(
                "C1: input_ids != canonical apply_chat_template(tokenize=True)\n"
                f"      got shape {tuple(ids.shape)}, ref shape {tuple(ref_ids.shape)}"
            )

        # C2: exactly one BOS token id, no double-BOS (id-level, decode-independent).
        if tok.bos_token_id is not None:
            n_bos = int((ids[0] == tok.bos_token_id).sum())
            if n_bos != 1:
                fails.append(
                    f"C2: expected exactly one BOS id ({tok.bos_token_id}), found {n_bos}"
                )

        # C3: the user-only generation-prompt prefix must be an exact prefix of the
        # full sequence, so ctx_len lands on the scored step's first content token.
        # Reasoning-model templates (DeepSeek-R1, etc.) inject a <think> token after
        # the assistant header on add_generation_prompt=True that is ABSENT from a
        # rendered assistant turn — this pushes ctx_len one token past the real
        # boundary, so pool_mean/pool_last and the attention query_idx silently drop
        # the step's first content token.
        prefix = tok.apply_chat_template(
            [user_msg], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", **tk,
        )["input_ids"]
        if prefix.shape[1] != ctx_len:
            fails.append(f"C3: prefix len {prefix.shape[1]} != ctx_len {ctx_len}")
        elif not torch.equal(ids[0, :ctx_len], prefix[0]):
            d = 0
            while d < ctx_len and int(ids[0, d]) == int(prefix[0, d]):
                d += 1
            injected = tok.convert_ids_to_tokens(prefix[0, d:].tolist())
            actual = tok.convert_ids_to_tokens(ids[0, d:d + 4].tolist())
            fails.append(
                "C3: ctx_len boundary misaligned — generation prompt injects "
                "token(s) absent from the scored assistant turn, so [ctx_len:] drops "
                "the step's first content token(s):\n"
                f"      prefix injects at {d}: {injected}\n"
                f"      full sequence has  : {actual}"
            )

    return fails, trailing_special


def run_check(args: argparse.Namespace) -> int:
    tok, tk = load_tokenizer_and_kwargs(args.model)
    trajs = resolve_dataset(args.input, args.subset)
    if args.limit is not None:
        trajs = trajs[: args.limit]

    n_steps = 0
    n_fail_steps = 0
    trailing_special = 0  # C4 informational flag
    print(
        f"Checking {len(trajs)} trajectories with tokenizer for {args.model!r} "
        f"(max_tokens={args.max_tokens})\n"
    )
    for traj in trajs:
        for step_idx in iter_scoreable_steps(traj):
            n_steps += 1
            fails, is_trailing_special = check_step(traj, step_idx, tok, tk, args.max_tokens)
            if is_trailing_special:
                trailing_special += 1

            if fails:
                n_fail_steps += 1
                print(f"✗ {traj.filename}  step {step_idx}")
                for f in fails:
                    print(f"    {f}")
            elif args.verbose:
                print(f"✓ {traj.filename}  step {step_idx}")

    print("\n" + "─" * 60)
    print(f"steps checked        : {n_steps}")
    print(f"steps with failures  : {n_fail_steps}")
    print(
        f"trailing special tok : {trailing_special}/{n_steps}  "
        f"(pool_last / query_idx include a template token when >0 — "
        f"decide whether to trim before pooling)"
    )
    if n_fail_steps == 0:
        print("RESULT: all checks passed ✓")
        return 0
    print("RESULT: FAILURES present ✗")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (show mode)
# ─────────────────────────────────────────────────────────────────────────────

# Rotating 256-colour backgrounds for step spans (readable on dark & light).
_STEP_BG = [39, 208, 34, 170, 214, 45, 205, 118, 220, 141, 44, 202]
_SCAFFOLD_BG = 240   # dim grey
_STEP_FG = 16        # near-black text for contrast on the bright backgrounds


def _region_map(step_tokens: dict[int, list[int]], step_idx: int, seq_len: int):
    """pos → ("ctx", i) | ("step", step_idx) | ("scaffold", None)."""
    region: list[tuple[str, int | None]] = [("scaffold", None)] * seq_len
    for i, positions in step_tokens.items():
        tag = "step" if i == step_idx else "ctx"
        for p in positions:
            if 0 <= p < seq_len:
                region[p] = (tag, i)
    return region


def _tok_text(tok: Any, tid: int) -> str:
    """Human-readable rendering of a single token, whitespace made visible."""
    s = tok.decode([tid])
    if s == "":
        s = tok.convert_ids_to_tokens(tid)  # special / whitespace-only tokens
    return s.replace("\n", "⏎").replace("\t", "→")


def render_ansi(traj: Trajectory, step_idx: int, tok: Any, ss: dict) -> str:
    ids = ss["input_ids"][0]
    ctx_len = ss["ctx_len"]
    seq_len = ids.shape[0]
    region = _region_map(ss["step_tokens"], step_idx, seq_len)
    special = set(tok.all_special_ids)

    # stable colour per step index
    step_order = [i for i in ss["step_tokens"] if i != step_idx]
    color_of = {i: _STEP_BG[k % len(_STEP_BG)] for k, i in enumerate(sorted(step_order))}

    out: list[str] = []
    for pos in range(seq_len):
        tid = int(ids[pos])
        tag, i = region[pos]
        text = _tok_text(tok, tid)
        if tag == "step":
            bg = 231  # bright white → the scored step / pooled / query region
        elif tag == "ctx":
            bg = color_of[i]
        else:
            bg = _SCAFFOLD_BG
        cell = f"\033[48;5;{bg}m\033[38;5;{_STEP_FG}m{text}\033[0m"
        if tid in special:
            cell = f"\033[4m{cell}\033[0m"        # underline template tokens
        if pos == seq_len - 1:
            cell = f"\033[7m{cell}\033[0m"          # reverse-video: pool_last token
        out.append(cell)

    legend = [
        "",
        f"trajectory : {traj.filename}   scored step : {step_idx} "
        f"[{traj.history[step_idx].get('role','?')}]",
        f"seq_len={seq_len}  ctx_len={ctx_len}  "
        f"context steps={len(step_order)}",
        "",
        "  \033[48;5;231m\033[38;5;16m scored step \033[0m  "
        "= [ctx_len:seq_len)  → ACTIVATION pools v_t here (mean over region);  "
        "ATTENTION query_idx",
        "  " + "  ".join(
            f"\033[48;5;{color_of[i]}m\033[38;5;16m step {i} \033[0m"
            for i in sorted(step_order)[:8]
        ) + ("  …" if len(step_order) > 8 else "")
        + "   = ATTENTION key spans (w_{i,t})",
        f"  \033[48;5;{_SCAFFOLD_BG}m\033[38;5;16m scaffold \033[0m "
        "= chat-template tokens (unassigned)   "
        "\033[4munderline\033[0m = special token   "
        "\033[7mreverse\033[0m = pool_last token (h[-1])",
        "",
    ]
    return "".join(out) + "\n" + "\n".join(legend)


def render_html(traj: Trajectory, step_idx: int, tok: Any, ss: dict) -> str:
    ids = ss["input_ids"][0]
    ctx_len = ss["ctx_len"]
    seq_len = ids.shape[0]
    region = _region_map(ss["step_tokens"], step_idx, seq_len)
    special = set(tok.all_special_ids)
    step_order = sorted(i for i in ss["step_tokens"] if i != step_idx)

    # hue-rotated palette for context steps
    hues = [(30 + int(330 * k / max(1, len(step_order)))) for k in range(len(step_order))]
    color_of = {i: f"hsl({hues[k]},70%,78%)" for k, i in enumerate(step_order)}

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace("\n", "⏎"))

    spans: list[str] = []
    for pos in range(seq_len):
        tid = int(ids[pos])
        tag, i = region[pos]
        if tag == "step":
            bg = "#ffffff"
        elif tag == "ctx":
            bg = color_of[i]
        else:
            bg = "#d0d0d0"
        border = "outline:2px solid #e11;" if pos == seq_len - 1 else ""
        weight = "font-weight:700;" if tid in special else ""
        title = f"pos={pos} id={tid} step={'scaffold' if i is None else i}"
        spans.append(
            f'<span style="background:{bg};{border}{weight}" title="{esc(title)}">'
            f'{esc(_tok_text(tok, tid))}</span>'
        )

    legend = "".join(
        f'<span style="background:{color_of[i]};padding:2px 6px;margin:2px;'
        f'border-radius:3px">step {i} [{esc(traj.history[i].get("role","?"))}]</span>'
        for i in step_order
    )
    return f"""<!doctype html><meta charset="utf-8">
<title>context: {esc(traj.filename)} step {step_idx}</title>
<body style="font-family:system-ui;margin:24px;line-height:2.1">
<h2>{esc(traj.filename)} — scored step {step_idx}
 [{esc(traj.history[step_idx].get('role','?'))}]</h2>
<p>seq_len={seq_len} · ctx_len={ctx_len} · context steps={len(step_order)}</p>
<p><b style="background:#fff;outline:1px solid #ccc;padding:2px 6px">scored step</b>
 = [ctx_len:seq_len): activation pools <i>v_t</i> here, attention <i>query_idx</i>.
 Coloured spans below = attention key steps (<i>w<sub>i,t</sub></i>).
 <span style="background:#d0d0d0;padding:2px 6px">scaffold</span> = chat-template tokens.
 <b>bold</b> = special token · red outline = pool_last token.</p>
<p>{legend}</p>
<hr>
<div style="font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap;
 word-break:break-all">{''.join(spans)}</div>
</body>"""


def run_show(args: argparse.Namespace) -> int:
    tok, tk = load_tokenizer_and_kwargs(args.model)
    trajs = resolve_dataset(args.input, args.subset)
    if args.index >= len(trajs):
        print(f"--index {args.index} out of range (have {len(trajs)})", file=sys.stderr)
        return 1
    traj = trajs[args.index]
    scoreable = iter_scoreable_steps(traj)
    if args.step not in scoreable:
        print(
            f"--step {args.step} is not scoreable for {traj.filename}; "
            f"choices: {scoreable[:20]}{'…' if len(scoreable) > 20 else ''}",
            file=sys.stderr,
        )
        return 1

    ss = separate_steps(traj, args.step, tok, args.max_tokens, tk)
    print(render_ansi(traj, args.step, tok, ss))
    if args.html:
        Path(args.html).write_text(render_html(traj, args.step, tok, ss))
        print(f"\nwrote HTML → {args.html}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="HF id / path (tokenizer only).")
    common.add_argument("--input", default="data/ww", help="Dataset directory.")
    common.add_argument("--subset", default=None, help="e.g. hand-crafted | algorithm-generated")
    common.add_argument("--max-tokens", type=int, default=None,
                        help="Context budget; set to exercise the truncation path.")

    c = sub.add_parser("check", parents=[common], help="Run Checks A/B/C.")
    c.add_argument("--limit", type=int, default=None, help="Max trajectories to check.")
    c.add_argument("--verbose", action="store_true", help="Print passing steps too.")

    s = sub.add_parser("show", parents=[common], help="Colour step boundaries for one step.")
    s.add_argument("--index", type=int, default=0, help="Trajectory index in the subset.")
    s.add_argument("--step", type=int, required=True, help="Scored step index.")
    s.add_argument("--html", default=None, help="Write a self-contained HTML file here.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "check":
        return run_check(args)
    return run_show(args)


if __name__ == "__main__":
    raise SystemExit(main())
