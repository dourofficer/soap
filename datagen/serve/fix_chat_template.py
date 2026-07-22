#!/usr/bin/env python3
"""Generate patched chat templates for models whose bundled template rejects
the message shapes the harnesses actually send.

Why this exists
---------------
Qwen3.5's `chat_template.jinja` scans backwards for the last user-role message
to locate `last_query_index`, and hard-fails when it finds none:

    {%- if ns.multi_step_tool %}
        {{- raise_exception('No user query found in messages.') }}
    {%- endif %}

autogen 0.2's group chat (Captain-Agent) routinely sends system+assistant-only
message lists once an expert has spoken, so every expert turn after the first
returns HTTP 400 `No user query found in messages.`, retries three times, and
yields empty content. The observable symptom is a Captain run that builds a
real expert team and then produces a 1-step trajectory with empty turns.

The fix is to drop the raise. `last_query_index` is already initialised to
`messages|length - 1`, so removing the guard falls back to that sentinel and
leaves normal user-containing conversations byte-identical.

The original checkpoint is never modified — vLLM is pointed at the patched copy
with `--chat-template` (see `chat_template:` in configs/serve.yaml).

    python datagen/serve/fix_chat_template.py --check-all   # audit, no writes
    python datagen/serve/fix_chat_template.py --model qwen3.5-9b
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

OUT_ROOT = common.DATAGEN_DIR / "serve" / "chat_templates"

# The guard to remove, tolerant of whitespace/`-` trim markers.
GUARD = re.compile(
    r"\{%-?\s*if\s+ns\.multi_step_tool\s*-?%\}\s*"
    r"\{\{-?\s*raise_exception\(\s*'No user query found in messages\.'\s*\)\s*-?\}\}\s*"
    r"\{%-?\s*endif\s*-?%\}",
    re.DOTALL,
)

# Message shapes the harnesses send. Captain hits the middle two constantly.
PROBES = {
    "system+user": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    "system+assistant": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
    "assistant-only": [{"role": "assistant", "content": "a"}],
}


def _render(template: str, messages: list[dict]) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}

    def raise_exception(msg):
        raise ValueError(msg)

    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True,
        raise_exception=raise_exception, tools=None)


def probe(template: str) -> dict[str, str]:
    """Render each probe shape; return {shape: 'ok' | error text}."""
    out = {}
    for name, msgs in PROBES.items():
        try:
            _render(template, msgs)
            out[name] = "ok"
        except Exception as e:  # noqa: BLE001 — the failure text is the result
            out[name] = f"{type(e).__name__}: {e}"
    return out


def template_path(model_dir: Path) -> Path | None:
    """Where the model keeps its chat template (file, or inside tokenizer_config)."""
    p = model_dir / "chat_template.jinja"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", dest="models",
                    help="model key from serve.yaml (repeatable)")
    ap.add_argument("--check-all", action="store_true",
                    help="report probe results for every model; write nothing")
    args = ap.parse_args()

    cfg = common.load_cfg("serve")
    if args.check_all:
        for name, spec in cfg["models"].items():
            src = (common.REPO_ROOT / spec["path"]).resolve()
            tpl = template_path(src)
            if tpl is None:
                print(f"  --   {name:18s} no chat_template.jinja (template lives in "
                      f"tokenizer_config or none)")
                continue
            r = probe(tpl.read_text())
            bad = [k for k, v in r.items() if v != "ok"]
            print(f"  {'OK  ' if not bad else 'BAD '} {name:18s} " +
                  ("all message shapes render" if not bad
                   else f"rejects {bad}: {r[bad[0]][:60]}"))
        return 0

    if not args.models:
        ap.error("pass --model <key> (repeatable) or --check-all")

    for name in args.models:
        spec = cfg["models"][name]
        src = (common.REPO_ROOT / spec["path"]).resolve()
        tpl = template_path(src)
        if tpl is None:
            print(f"[{name}] no chat_template.jinja at {src} — nothing to patch")
            continue

        original = tpl.read_text()
        before = probe(original)
        print(f"[{name}] original: " +
              ", ".join(f"{k}={'ok' if v == 'ok' else 'FAIL'}" for k, v in before.items()))
        if all(v == "ok" for v in before.values()):
            print("  already permissive; drop the `chat_template:` key from serve.yaml")
            continue

        patched, n = GUARD.subn("", original, count=1)
        if not n:
            print("  FAILED: the `No user query found` guard was not found — the "
                  "template changed upstream; re-inspect before patching.")
            return 1

        after = probe(patched)
        print(f"  patched : " +
              ", ".join(f"{k}={'ok' if v == 'ok' else 'FAIL'}" for k, v in after.items()))
        if any(v != "ok" for v in after.values()):
            print("  FAILED: patch did not make every shape render.")
            return 1

        # A user-containing conversation must render exactly as before, or the
        # patch would silently change normal generation.
        if _render(original, PROBES["system+user"]) != _render(patched, PROBES["system+user"]):
            print("  FAILED: patch altered rendering of a normal system+user chat.")
            return 1

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        dst = OUT_ROOT / f"{name}.jinja"
        dst.write_text(patched)
        print(f"  wrote {dst}  ({len(original) - len(patched)} chars removed; "
              f"normal chats render identically)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
