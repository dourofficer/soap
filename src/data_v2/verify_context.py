"""src/data_v2/verify_context.py — verify & visualise the data_v2 construction.

Mirrors ``src/data/verify_context.py`` but checks the head+content construction of
:mod:`src.data_v2.context`:

  A. ``build_context`` and ``separate_steps`` feed identical inputs.
  B. per-step spans partition the sequence — each context span decodes to its
     serialized turn, the scored span decodes to the scored step, no overlap.
  C. construction faithfulness — ``input_ids[:ctx_len]`` is exactly the
     on-distribution prompt (``apply_chat_template(add_generation_prompt=True)`` +
     empty-think close), ``input_ids[ctx_len:]`` is exactly the scored content,
     one BOS, and the empty ``<think></think>`` block is present for reasoning models.

Tokenizer-only, no GPU. Run from the repo root:

    python -m src.data_v2.verify_context check --model <hf_id> --input data/ww --subset <s> [--limit N] [--verbose]
    python -m src.data_v2.verify_context show  --model <hf_id> --input data/ww --subset <s> --index i --step t [--html out.html]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.models import get_adapter
from .trajectory import Trajectory, load_dataset
from .context import (
    build_context,
    separate_steps,
    select_context,
    iter_scoreable_steps,
    _serialize_turns,
    _scaffold,
    THINK_OPEN,
    THINK_CLOSE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer_and_kwargs(model: str) -> tuple[Any, dict]:
    """Tokenizer + template_kwargs, no model weights (mirrors the extractors)."""
    adapter = get_adapter(model)
    tok = AutoTokenizer.from_pretrained(model)
    return tok, adapter.template_kwargs()


def resolve_dataset(input_path: str, subset: str | None) -> list[Trajectory]:
    p = Path(input_path)
    if subset:
        base, sub = str(p), subset
    else:
        base, sub = str(p.parent), p.name
    return load_dataset(base, subset=sub)


def _norm(s: str) -> str:
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
    """Run Checks A/B/C for one (trajectory, step). Returns ``(fails, trailing_special)``."""
    fails: list[str] = []
    history = traj.history

    bc = build_context(traj, step_idx, tok, max_tokens, tk)
    ss = separate_steps(traj, step_idx, tok, max_tokens, tk)
    ids = ss["input_ids"]
    ctx_len = ss["ctx_len"]
    seq_len = ids.shape[1]
    st = ss["step_tokens"]
    ids0 = ids[0].tolist()

    ctx_indices_full = select_context(history, step_idx)
    was_truncated = False
    if max_tokens is not None:
        untrunc_len = separate_steps(traj, step_idx, tok, None, tk)["input_ids"].shape[1]
        was_truncated = untrunc_len > max_tokens

    trailing_special = ids0[-1] in set(tok.all_special_ids)

    # ── Check A — builder agreement ──────────────────────────────────────────
    if bc["input_ids"].shape != ids.shape or bc["input_ids"].tolist() != ids.tolist():
        fails.append(
            f"A: build_context vs separate_steps input_ids differ "
            f"({tuple(bc['input_ids'].shape)} vs {tuple(ids.shape)})"
        )
    if bc["ctx_len"] != ctx_len:
        fails.append(f"A: ctx_len differs ({bc['ctx_len']} vs {ctx_len})")

    # ── Check B — span alignment ─────────────────────────────────────────────
    # B1: scored step is the post-head region.
    if st.get(step_idx) != list(range(ctx_len, seq_len)):
        fails.append(f"B1: step_tokens[{step_idx}] != range({ctx_len}, {seq_len})")

    # B2: context spans — contiguous, ordered, non-overlapping, all inside [0, ctx_len).
    ctx_ids_sorted = sorted(
        (i for i in st if i != step_idx), key=lambda i: st[i][0] if st[i] else -1
    )
    seen: set[int] = set()
    prev_last = -1
    for i in ctx_ids_sorted:
        span = st[i]
        if not span:
            fails.append(f"B2: step {i} has an EMPTY span (no tokens assigned)")
            continue
        if span != list(range(span[0], span[-1] + 1)):
            fails.append(f"B2: step {i} span not contiguous: {span[:3]}…")
        if span[-1] >= ctx_len:
            fails.append(f"B2: step {i} span {(span[0], span[-1])} not inside [0, {ctx_len})")
        if span[0] <= prev_last:
            fails.append(f"B2: step {i} overlaps/precedes previous (start {span[0]} <= {prev_last})")
        prev_last = span[-1]
        if set(span) & seen:
            fails.append(f"B2: step {i} span overlaps an earlier step")
        seen |= set(span)
    if not was_truncated and ctx_ids_sorted != [i for i in ctx_indices_full if i in st]:
        fails.append("B2: context span order != select_context order")

    # B3: each context span's ids equal that turn's tokens; scored region equals the
    # scored step's tokens. Compared at the id level — decode is lossy on some fast
    # tokenizers (DeepSeek strips spaces / alters chars), so decode-based matching gives
    # false positives even when the spans are exactly right.
    for i in ctx_ids_sorted:
        span_ids = [ids0[p] for p in st[i]]
        chunk_ids = tok(_serialize_turns(history, [i]), add_special_tokens=False)["input_ids"]
        if span_ids != chunk_ids:
            fails.append(
                f"B3: step {i} span ids != tokenized turn\n"
                f"      span decodes : {tok.decode(span_ids)[:100]!r}\n"
                f"      expected turn: {_serialize_turns(history, [i])[:100]!r}"
            )
    step_serialized = _serialize_turns(history, [step_idx])
    scored_ids = ids0[ctx_len:]
    content_ids = tok(step_serialized, add_special_tokens=False)["input_ids"]
    if was_truncated:
        ok = (not scored_ids) or scored_ids == content_ids[-len(scored_ids):]  # front-sliced suffix
    else:
        ok = scored_ids == content_ids
    if not ok:
        fails.append(
            f"B3: scored step {step_idx} content ids mismatch in [ctx_len:]\n"
            f"      decodes : {tok.decode(scored_ids)[:100]!r}"
        )

    # ── Check C — construction faithfulness ──────────────────────────────────
    if not was_truncated:
        _opening, closing = _scaffold(tok, tk)
        close_ids = tok(closing, add_special_tokens=False)["input_ids"]

        # C1: the assistant opener (+ empty think) sits immediately before ctx_len, so the
        # scored content (checked id-exact in B3) begins on the content edge.
        if ctx_len < len(close_ids) or ids0[ctx_len - len(close_ids):ctx_len] != close_ids:
            fails.append("C1: closing scaffold (assistant opener/think) not right before ctx_len")

        # C2: exactly one BOS.
        if tok.bos_token_id is not None:
            n_bos = ids0.count(tok.bos_token_id)
            if n_bos != 1:
                fails.append(f"C2: expected exactly one BOS id ({tok.bos_token_id}), found {n_bos}")

        # C3: empty <think></think> block present for reasoning models.
        if THINK_OPEN in closing:
            if THINK_CLOSE not in closing:
                fails.append("C3: <think> opened in closing scaffold but never closed")
            else:
                inner = closing[closing.index(THINK_OPEN) + len(THINK_OPEN):closing.index(THINK_CLOSE)]
                if inner.strip() != "":
                    fails.append(f"C3: <think> block is not empty: {inner!r}")

    return fails, trailing_special


def run_check(args: argparse.Namespace) -> int:
    tok, tk = load_tokenizer_and_kwargs(args.model)
    trajs = resolve_dataset(args.input, args.subset)
    if args.limit is not None:
        trajs = trajs[: args.limit]

    n_steps = n_fail_steps = trailing_special = 0
    print(
        f"[data_v2] Checking {len(trajs)} trajectories with tokenizer for {args.model!r} "
        f"(max_tokens={args.max_tokens})\n"
    )
    for traj in trajs:
        for step_idx in iter_scoreable_steps(traj):
            n_steps += 1
            fails, is_trailing = check_step(traj, step_idx, tok, tk, args.max_tokens)
            trailing_special += int(is_trailing)
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
        f"(should be 0 — data_v2 appends no EOS, so pool_last is the last content token)"
    )
    if n_fail_steps == 0:
        print("RESULT: all checks passed ✓")
        return 0
    print("RESULT: FAILURES present ✗")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (show mode)
# ─────────────────────────────────────────────────────────────────────────────

_STEP_BG = [39, 208, 34, 170, 214, 45, 205, 118, 220, 141, 44, 202]
_SCAFFOLD_BG = 240
_STEP_FG = 16


def _region_map(step_tokens: dict[int, list[int]], step_idx: int, seq_len: int):
    region: list[tuple[str, int | None]] = [("scaffold", None)] * seq_len
    for i, positions in step_tokens.items():
        tag = "step" if i == step_idx else "ctx"
        for p in positions:
            if 0 <= p < seq_len:
                region[p] = (tag, i)
    return region


def _tok_text(tok: Any, tid: int) -> str:
    s = tok.decode([tid])
    if s == "":
        s = tok.convert_ids_to_tokens(tid)
    return s.replace("\n", "⏎").replace("\t", "→")


def render_ansi(traj: Trajectory, step_idx: int, tok: Any, ss: dict) -> str:
    ids = ss["input_ids"][0]
    ctx_len = ss["ctx_len"]
    seq_len = ids.shape[0]
    region = _region_map(ss["step_tokens"], step_idx, seq_len)
    special = set(tok.all_special_ids)
    step_order = [i for i in ss["step_tokens"] if i != step_idx]
    color_of = {i: _STEP_BG[k % len(_STEP_BG)] for k, i in enumerate(sorted(step_order))}

    out: list[str] = []
    for pos in range(seq_len):
        tid = int(ids[pos])
        tag, i = region[pos]
        text = _tok_text(tok, tid)
        bg = 231 if tag == "step" else (color_of[i] if tag == "ctx" else _SCAFFOLD_BG)
        cell = f"\033[48;5;{bg}m\033[38;5;{_STEP_FG}m{text}\033[0m"
        if tid in special:
            cell = f"\033[4m{cell}\033[0m"
        if pos == seq_len - 1:
            cell = f"\033[7m{cell}\033[0m"
        out.append(cell)

    legend = [
        "",
        f"trajectory : {traj.filename}   scored step : {step_idx} "
        f"[{traj.history[step_idx].get('role','?')}]",
        f"seq_len={seq_len}  ctx_len={ctx_len}  context steps={len(step_order)}",
        "",
        "  \033[48;5;231m\033[38;5;16m scored step \033[0m  "
        "= [ctx_len:seq_len)  → ACTIVATION pools v_t here;  ATTENTION query_idx",
        "  " + "  ".join(
            f"\033[48;5;{color_of[i]}m\033[38;5;16m step {i} \033[0m"
            for i in sorted(step_order)[:8]
        ) + ("  …" if len(step_order) > 8 else "") + "   = ATTENTION key spans (w_{i,t})",
        f"  \033[48;5;{_SCAFFOLD_BG}m\033[38;5;16m scaffold \033[0m "
        "= template tokens incl. empty <think></think> (unassigned)   "
        "\033[4munderline\033[0m = special token   "
        "\033[7mreverse\033[0m = pool_last token (h[-1], now the last content token)",
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
    hues = [(30 + int(330 * k / max(1, len(step_order)))) for k in range(len(step_order))]
    color_of = {i: f"hsl({hues[k]},70%,78%)" for k, i in enumerate(step_order)}

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace("\n", "⏎"))

    spans: list[str] = []
    for pos in range(seq_len):
        tid = int(ids[pos])
        tag, i = region[pos]
        bg = "#ffffff" if tag == "step" else (color_of[i] if tag == "ctx" else "#d0d0d0")
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
<title>data_v2 context: {esc(traj.filename)} step {step_idx}</title>
<body style="font-family:system-ui;margin:24px;line-height:2.1">
<h2>{esc(traj.filename)} — scored step {step_idx}
 [{esc(traj.history[step_idx].get('role','?'))}]</h2>
<p>seq_len={seq_len} · ctx_len={ctx_len} · context steps={len(step_order)}</p>
<p><b style="background:#fff;outline:1px solid #ccc;padding:2px 6px">scored step</b>
 = [ctx_len:seq_len): activation pools <i>v_t</i>, attention <i>query_idx</i>.
 Coloured spans = attention key steps. <span style="background:#d0d0d0;padding:2px 6px">scaffold</span>
 = template tokens incl. empty &lt;think&gt;&lt;/think&gt;. <b>bold</b> = special · red outline = pool_last.</p>
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
            f"--step {args.step} not scoreable for {traj.filename}; choices: "
            f"{scoreable[:20]}{'…' if len(scoreable) > 20 else ''}",
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
