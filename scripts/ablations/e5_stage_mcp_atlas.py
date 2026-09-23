"""E5 staging — MCP-Atlas runs as a reference corpus, successes and failures alike.

Writes `data/synthetic/mcp-atlas/<n>.json`, one file per run of the corpus OAT ships
(`../attrib-prompting/vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B/`). Outcome follows
OAT's own rule (`data_pipeline.load_mcp_atlas_trajectories`): a run is a success when
its `errors` list is empty; a failure whose errors name no valid step is dropped, as OAT
drops it. One step is one message of `raw_conversation_history`, with OAT's body —
reasoning, then content, then tool calls; non-model content cut at 4,096 characters —
so SOAP and OAT read the same text. The files carry the minimal schema `main.data`
requires plus `outcome` ("success" | "fail", the field E4's `failed_files` reads) and
`source_id`. `mistake_step = -1`: the error annotations are never read.

    python scripts/ablations/e5_stage_mcp_atlas.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO.parent / "attrib-prompting/vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B"
OUT = REPO / "data" / "synthetic" / "mcp-atlas"
MAX_TOOL_CONTENT_LENGTH = 4096          # OAT's config.MAX_TOOL_CONTENT_LENGTH


def serialize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item["text"]) if isinstance(item, dict) and "text" in item
                         else str(item) for item in content)
    return str(content)


def step_body(entry: dict) -> str:
    is_model = entry.get("role") == "assistant"
    parts = []
    if is_model:
        reasoning = (entry.get("original_message") or {}).get("reasoning_content", "")
        if reasoning:
            parts.append(f"[REASONING]\n{reasoning}")
    content = serialize_content(entry.get("content"))
    if not is_model and len(content) > MAX_TOOL_CONTENT_LENGTH:
        content = content[:MAX_TOOL_CONTENT_LENGTH] + "\n...[truncated]"
    if content:
        parts.append(content)
    for tc in entry.get("tool_calls") or []:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            parts.append(f"[TOOL_CALL: {func.get('name', 'unknown')}"
                         f"({func.get('arguments', '')})]")
    return "\n".join(parts)


def has_valid_error_step(errors: list, n_steps: int) -> bool:
    for err in errors:
        scope = err.get("scope")                      # OAT's `_parse_scope`
        if isinstance(scope, str) and scope.lstrip("-").isdigit():
            scope = int(scope)
        if isinstance(scope, int) and 0 <= scope - 1 < n_steps:
            return True
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"success": 0, "fail": 0, "dropped": 0}
    n = 0
    for fp in sorted(SRC.glob("*.json")):
        data = json.loads(fp.read_text())
        errors, history = data.get("errors"), data.get("raw_conversation_history") or []
        if not isinstance(errors, list) or not history:
            counts["dropped"] += 1
            continue
        if errors and not has_valid_error_step(errors, len(history)):
            counts["dropped"] += 1
            continue
        outcome = "fail" if errors else "success"
        counts[outcome] += 1
        record = {"question_ID": f"mcp-atlas-{fp.stem}", "source_id": fp.stem,
                  "question": str(data.get("PROMPT") or ""), "ground_truth": "",
                  "history": [{"role": str(e.get("role", "unknown")),
                               "content": step_body(e)} for e in history],
                  "mistake_agent": "", "mistake_step": -1, "mistake_reason": "",
                  "level": -1, "subset": "mcp-atlas", "outcome": outcome}
        (OUT / f"{n}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
        n += 1
    print(f"wrote {n} files to {OUT}  {counts}")


if __name__ == "__main__":
    main()
