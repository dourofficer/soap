#!/usr/bin/env python3
"""Convert the TraceElephant dataset into the repo's trajectory JSON schema.

TraceElephant ships as a single zip (`data/elephant.zip`) whose tasks live under
5 run-group directories that map onto 3 agent systems:

    captain-runs-gaia, captain-runs-assistantbench          -> captain
    magentic-runs-gaia, magentic-runs-assistant-bench       -> magentic
    swe-agent-runs-swe-bench                                 -> swe

Every task dir is in the "new format": `trace_metadata.json` + `step_records.json`.
`step_records.json` is a list of steps in execution order; each step's assistant
text lives at `output.choices[0].message.content` (already a parsed dict — no
`ChatCompletion` string parsing needed). `mistake_step` is 1-based (== `step_id`),
so the 0-based index is `int(mistake_step) - 1`.

Output: `data/traceelephant/{captain,magentic,swe}/<N>.json`, one trajectory per
file, matching the schema `src.data.trajectory.load_dataset` expects.

Run from repo root:
    python scripts/convert_traceelephant.py --zip data/elephant.zip --out data/traceelephant
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# run-group dir prefix -> output system/subset name
GROUP_TO_SYSTEM = {
    "captain-runs-": "captain",
    "magentic-runs-": "magentic",
    "swe-agent-runs-": "swe",
}
# iteration order for reproducibility (captain, then magentic, then swe)
SYSTEM_ORDER = ["captain", "magentic", "swe"]

_ARG_TRUNC = 2000  # cap serialized tool-call arguments so files stay reasonable


def system_of(group_dir: str) -> str | None:
    for prefix, system in GROUP_TO_SYSTEM.items():
        if group_dir.startswith(prefix):
            return system
    return None


def _list_task_dirs(z: zipfile.ZipFile) -> dict[str, list[str]]:
    """Return {system: [task_dir_path, ...]} sorted by task-dir name.

    task_dir_path is the zip-internal prefix, e.g.
    "data/captain-runs-gaia/gaia_task_0_.../".
    """
    seen: set[str] = set()
    by_system: dict[str, list[str]] = defaultdict(list)
    for name in z.namelist():
        m = re.match(r"(data/([^/]+)/([^/]+))/", name)
        if not m:
            continue
        task_dir, group_dir = m.group(1), m.group(2)
        if task_dir in seen:
            continue
        system = system_of(group_dir)
        if system is None:
            continue
        seen.add(task_dir)
        by_system[system].append(task_dir)
    for system in by_system:
        by_system[system].sort()
    return by_system


def _load_json(z: zipfile.ZipFile, path: str):
    with z.open(path) as f:
        return json.load(f)


def _serialize_tool_calls(agent_name: str, tool_calls: list) -> str:
    """Render tool calls as readable text so tool-only steps aren't blank."""
    blocks = []
    for tc in tool_calls:
        fn = (tc or {}).get("function", {}) or {}
        name = fn.get("name", "<unknown>")
        args = fn.get("arguments", "")
        if isinstance(args, str):
            # arguments is usually a JSON string; pretty-print if it parses
            try:
                args = json.dumps(json.loads(args), indent=2, ensure_ascii=False)
            except (ValueError, TypeError):
                pass
        else:
            args = json.dumps(args, indent=2, ensure_ascii=False)
        if len(args) > _ARG_TRUNC:
            args = args[:_ARG_TRUNC] + "\n... [truncated]"
        blocks.append(f"[{agent_name}] tool_call {name}(\n{args}\n)")
    return "\n\n".join(blocks)


def extract_content(record: dict, stats: dict) -> str:
    """Clean assistant text for one step.

    1. output.choices[0].message.content
    2. else serialize message.tool_calls
    3. else str(message)  (counted as a fallback)
    """
    agent_name = record.get("agent_name", "")
    try:
        message = record["output"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        stats["content_fallback"] += 1
        return str(record.get("output", ""))

    content = message.get("content")
    if content:
        return content

    # tool-call steps carry no assistant text; the invocation lives either in
    # message.tool_calls (magentic) or the step's tool_logs (swe). Both share the
    # {"function": {"name", "arguments"}} shape.
    tool_calls = message.get("tool_calls") or record.get("tool_logs")
    if tool_calls:
        return _serialize_tool_calls(agent_name, tool_calls)

    stats["content_fallback"] += 1
    return str(message)


def parse_mistake_step(raw, malformed: list, task_id: str) -> int:
    """1-based `mistake_step` -> 0-based index; -1 (kept) if unparseable."""
    if isinstance(raw, int):
        return raw - 1
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return int(raw.strip()) - 1
    malformed.append((task_id, raw))
    return -1


def convert_trace(z: zipfile.ZipFile, task_dir: str, system: str, stats: dict):
    """Return a converted trajectory dict, or None if it must be skipped."""
    task_id = task_dir.rsplit("/", 1)[-1]
    meta = _load_json(z, f"{task_dir}/trace_metadata.json")
    records = _load_json(z, f"{task_dir}/step_records.json")

    history = [
        {"role": r.get("agent_name", ""), "content": extract_content(r, stats)}
        for r in records
    ]

    mistake_agent = meta.get("mistake_agent", "")
    mistake_step = parse_mistake_step(
        meta.get("mistake_step"), stats["malformed_step"], task_id
    )

    # ── validation ──
    if not history or any(not turn["role"] for turn in history):
        stats["skipped_empty"].append(task_id)
        return None
    if mistake_step != -1:
        if not (0 <= mistake_step < len(history)):
            stats["validation_failed"].append((task_id, f"step {mistake_step} out of range 0..{len(history)-1}"))
            return None
        if history[mistake_step]["role"] != mistake_agent:
            stats["validation_failed"].append(
                (task_id, f"role {history[mistake_step]['role']!r} != mistake_agent {mistake_agent!r}")
            )
            return None

    out = {
        "question_ID": task_id,
        "question": meta.get("task_instruction", ""),
        "history": history,
        "mistake_agent": mistake_agent,
        "mistake_step": mistake_step,
        "level": -1,
        "system": meta.get("agent_system_intro"),
        "subset": system,
    }
    if "ground_truth" in meta:
        out["ground_truth"] = meta["ground_truth"]
    if "mistake_reason" in meta:
        out["mistake_reason"] = meta["mistake_reason"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="data/elephant.zip", help="path to the TraceElephant zip")
    ap.add_argument("--out", default="data/traceelephant", help="output root directory")
    ap.add_argument("--systems", default="captain,magentic,swe",
                    help="comma-separated systems to convert")
    ap.add_argument("--limit", type=int, default=None, help="cap traces per system (debug)")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
    out_root = Path(args.out)

    z = zipfile.ZipFile(args.zip)
    by_system = _list_task_dirs(z)

    grand = {}
    for system in SYSTEM_ORDER:
        if system not in wanted:
            continue
        task_dirs = by_system.get(system, [])
        if args.limit is not None:
            task_dirs = task_dirs[: args.limit]
        stats = {
            "content_fallback": 0,
            "malformed_step": [],
            "skipped_empty": [],
            "validation_failed": [],
            "converted": 0,
        }
        out_dir = out_root / system
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = 0
        for task_dir in task_dirs:
            try:
                traj = convert_trace(z, task_dir, system, stats)
            except Exception as e:  # noqa: BLE001 - report, don't crash the batch
                stats["validation_failed"].append((task_dir.rsplit("/", 1)[-1], f"exception: {e}"))
                continue
            if traj is None:
                continue
            (out_dir / f"{idx}.json").write_text(
                json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            idx += 1
            stats["converted"] += 1
        grand[system] = stats

    # ── summary ──
    print("\n" + "=" * 60)
    print("TraceElephant conversion summary")
    print("=" * 60)
    for system in SYSTEM_ORDER:
        if system not in grand:
            continue
        s = grand[system]
        print(f"\n[{system}] -> {out_root / system}")
        print(f"  converted            : {s['converted']}")
        print(f"  content fallbacks    : {s['content_fallback']}")
        print(f"  mistake_step = -1    : {len(s['malformed_step'])}")
        for tid, raw in s["malformed_step"]:
            print(f"      - {tid}  (raw={raw!r})")
        print(f"  skipped (empty)      : {len(s['skipped_empty'])}")
        for tid in s["skipped_empty"]:
            print(f"      - {tid}")
        print(f"  validation failed    : {len(s['validation_failed'])}")
        for tid, reason in s["validation_failed"]:
            print(f"      - {tid}: {reason}")
    print()


if __name__ == "__main__":
    main()
