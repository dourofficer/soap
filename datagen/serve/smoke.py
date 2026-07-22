#!/usr/bin/env python3
"""Verify a served endpoint can do everything the harnesses need.

The Magentic-One orchestrator drives its ledger through JSON-mode completions,
and its tool-using agents (FileSurfer, the coder's executor) need function
calling. This script checks each capability separately so a failure names the
exact thing that is broken, rather than surfacing later as a mid-trajectory
crash.

    python datagen/serve/smoke.py --model qwen3.5-9b --wait
    python datagen/serve/smoke.py --all

Exit status is non-zero if any REQUIRED check fails. Tool calling is required
only for models declaring `--enable-auto-tool-choice` in serve.yaml; for the
others (deepseek-8b) it is reported as "n/a" and not counted as a failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def _client(ep: dict):
    from openai import OpenAI

    return OpenAI(base_url=ep["base_url"], api_key=ep["api_key"], timeout=120.0)


def wait_ready(ep: dict, timeout_s: int) -> bool:
    """Poll /v1/models until the server answers or the deadline passes."""
    import openai

    deadline = time.time() + timeout_s
    client = _client(ep)
    while time.time() < deadline:
        try:
            client.models.list()
            return True
        except (openai.APIConnectionError, openai.APIStatusError):
            time.sleep(5)
    return False


def check_models(ep: dict) -> tuple[bool, str]:
    served = [m.id for m in _client(ep).models.list().data]
    if ep["model"] not in served:
        return False, f"served ids {served} do not include {ep['model']!r}"
    return True, f"serving {served}"


def check_chat(ep: dict) -> tuple[bool, str]:
    # Ask for multi-line output: a broken ByteLevel decoder only shows up on
    # whitespace, so a single-word reply would pass a corrupt tokenizer.
    r = _client(ep).chat.completions.create(
        model=ep["model"],
        messages=[{"role": "user", "content":
                   "Reply with exactly two lines: 'PONG' then 'SECOND LINE'."}],
        max_tokens=2048, temperature=0.0,
    )
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning_content", None)
    # A reasoning parser is configured for both models; confirm the thinking is
    # split out of `content`, since trajectory steps are logged from `content`.
    split = "reasoning split out" if reasoning else "no reasoning_content"

    # Raw ByteLevel markers mean the tokenizer never applied its decoder —
    # see datagen/serve/fix_tokenizer.py. Every trajectory step would be
    # unreadable, so this is a hard failure, not a warning.
    stray = [m for m in ("Ċ", "Ġ", "âĢ") if m in content]
    if stray:
        return False, (f"undecoded byte tokens {stray} in content "
                       f"({content[:80]!r}) — patch the tokenizer")
    if "<think>" in content:
        return False, f"<think> leaked into content: {content[:120]!r}"
    if "PONG" not in content.upper():
        return False, f"unexpected reply {content[:120]!r} ({split})"
    if "\n" not in content:
        return False, f"no newline survived in a two-line reply: {content[:120]!r}"
    return True, f"{content[:40]!r}; {split}"


def check_json(ep: dict) -> tuple[bool, str]:
    r = _client(ep).chat.completions.create(
        model=ep["model"],
        messages=[
            {"role": "system", "content": "You reply with JSON only."},
            {"role": "user", "content":
             'Return a JSON object with keys "is_request_satisfied" (boolean) '
             'and "next_speaker" (string, one of Coder/Executor). Any values.'},
        ],
        response_format={"type": "json_object"},
        max_tokens=2048, temperature=0.0,
    )
    raw = (r.choices[0].message.content or "").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON ({e}): {raw[:120]!r}"
    return True, f"parsed keys {sorted(obj)}"


def check_tools(ep: dict) -> tuple[bool | None, str]:
    r = _client(ep).chat.completions.create(
        model=ep["model"],
        messages=[{"role": "user", "content": "What is the weather in Hanoi? Use the tool."}],
        tools=[WEATHER_TOOL], tool_choice="auto",
        max_tokens=2048, temperature=0.0,
    )
    calls = r.choices[0].message.tool_calls
    if not calls:
        return False, f"no tool_calls; content={(r.choices[0].message.content or '')[:120]!r}"
    fn = calls[0].function
    try:
        args = json.loads(fn.arguments)
    except json.JSONDecodeError as e:
        return False, f"tool args not JSON ({e}): {fn.arguments[:120]!r}"
    return True, f"{fn.name}({args})"


def run_checks(name: str, cfg: dict, wait_s: int) -> bool:
    ep = common.resolve_endpoint(name, cfg)
    spec = cfg["models"][name]
    tools_expected = "--enable-auto-tool-choice" in [str(a) for a in spec.get("extra_args", [])]

    print(f"\n=== {name}  ({ep['base_url']}) ===")
    if wait_s:
        print(f"  waiting up to {wait_s}s for readiness...", flush=True)
        if not wait_ready(ep, wait_s):
            print(f"  FAIL  endpoint not answering after {wait_s}s")
            return False

    checks = [("models", check_models), ("chat", check_chat), ("json", check_json)]
    ok_all = True
    for label, fn in checks:
        try:
            ok, detail = fn(ep)
        except Exception as e:  # noqa: BLE001 — report, keep checking
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"  {'PASS' if ok else 'FAIL'}  {label:6s} {detail}")
        ok_all &= ok

    try:
        ok, detail = check_tools(ep)
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    if tools_expected:
        print(f"  {'PASS' if ok else 'FAIL'}  tools  {detail}")
        ok_all &= ok
    else:
        # Not configured for tool use by design (R1-distill is not tool-trained);
        # a failure here is expected and must not fail the smoke run.
        print(f"  n/a   tools  not enabled in serve.yaml — {detail}")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", dest="models",
                    help="model key from serve.yaml (repeatable)")
    ap.add_argument("--all", action="store_true", help="check every model in serve.yaml")
    ap.add_argument("--wait", nargs="?", type=int, const=900, default=0,
                    metavar="SECONDS",
                    help="block until the endpoint answers (default 900s)")
    args = ap.parse_args()

    cfg = common.load_cfg("serve")
    names = list(cfg["models"]) if args.all else (args.models or [])
    if not names:
        ap.error("pass --model <key> (repeatable) or --all")

    results = {n: run_checks(n, cfg, args.wait) for n in names}
    print("\n" + "=" * 50)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
