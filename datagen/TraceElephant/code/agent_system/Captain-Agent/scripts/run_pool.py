#!/usr/bin/env python3
"""Run ONE task from a prepared datagen pool through Captain-Agent.

Adapted from scripts/run_assistantbench.py. Differences that matter:

* **Pool-agnostic.** Reads the uniform task schema written by
  datagen/pools/prepare.py (`question`/`answer`/`id`/`pool`) instead of
  AssistantBench's `task`/`answer`/`id`.
* **Orchestrator-owned output.** `--output-dir` is the run directory; nothing
  is written to timestamped paths under the harness repo, so the batch
  orchestrator can treat `summary.json` as the done-marker.
* **`summary.json` matches the datagen raw-run layout** so one converter
  handles both harnesses. Captain history entries are `{content, name}` —
  the converter maps `name` → `role`.
* **Build reuse.** `--build-state` loads a previously autobuilt team, which
  amortizes the (expensive) build phase across a pool.

Captain talks to its agents through markdown code blocks and a retrieved tool
library rather than native function calling, which makes it the friendlier
harness for backbones served without a tool-call parser.

Usage (normally via datagen/collect/run_batch.py):
    python scripts/run_pool.py --pool gsm8k --task-index 0 \
        --output-dir /path/to/run --model qwen3.5-9b \
        --config-list /path/to/oai_config_list.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

import autogen
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent

load_dotenv(override=False)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Captain-Agent → agent_system → code → TraceElephant → datagen → repo root
ATTRIBSCOPE_ROOT = Path(__file__).resolve().parents[6]

SYSTEM_TEMPLATE = """Today's date is {today}.

# Task
You need to solve the below question given by a user.

# Question
{question}
{attachment}
# Important Constraint
You MUST solve this problem within {max_round} rounds of conversation. Plan your approach efficiently and focus on the most direct path to the answer.

# Output format (MANDATORY)
Please respond in the following structure:
## ANSWER
[concise final answer]
## REASON
[brief reasoning or evidence used to reach the answer]

If you obtain and output the final answer, please output 'terminate' in **uppercase** format.
"""

DEFAULT_BUILDING_TASK = """- Web search specialist skilled at finding and analyzing online information
- Python coder who can process data and perform calculations
- Checker that verifies answers and enforces grounding in cited evidence"""


# ── config ────────────────────────────────────────────────────────────────────

def build_llm_configs(args, work_dir: Path):
    """LLM + autobuild config. Mirrors the upstream driver's structure."""
    config_list = autogen.config_list_from_json(
        args.config_list, filter_dict={"model": [args.model]})
    if not config_list:
        raise SystemExit(f"no entry for model {args.model!r} in {args.config_list}")

    general = {"temperature": 0.1, "top_p": 0.95,
               "timeout": args.request_timeout, "config_list": config_list}
    tool_root = REPO_ROOT / "tools"
    nested = {
        "autobuild_init_config": {
            "config_file_or_env": args.config_list,
            "builder_model": [args.model],
            "agent_model": [args.model],
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 0.1, "top_p": 0.95,
                                   "max_tokens": 4096, "cache_seed": None,
                                   "timeout": args.request_timeout},
            "code_execution_config": {"timeout": 300, "work_dir": str(work_dir),
                                      "last_n_messages": 2, "use_docker": False},
            "coding": True,
            "library_path_or_json": str(REPO_ROOT / "agent_library.json"),
        },
        "autobuild_tool_config": {
            "tool_corpus": str(tool_root / "tool_description.tsv"),
            "tool_root": str(tool_root),
            "retriever": "all-MiniLM-L6-v2",
        },
        "group_chat_config": {"max_round": args.max_round},
        "group_chat_llm_config": general.copy(),
    }
    return general, nested


# ── history extraction ────────────────────────────────────────────────────────

def collect_history(meta_user_proxy) -> tuple[List[Dict], Dict]:
    """Pull the nested group-chat transcript out of the agent objects.

    autogen 0.2 exposes no transcript API, so this walks the same caches the
    upstream driver does, most reliable source first.
    """
    history: List[Dict] = []
    system_prompts: Dict[str, str] = {}

    try:
        cache = getattr(meta_user_proxy, "_agent_list_cache", None) or {}
        for group_name, agent_list in cache.items():
            for agent in agent_list:
                if hasattr(agent, "name") and hasattr(agent, "system_message"):
                    system_prompts[agent.name] = agent.system_message

            gc_cache = getattr(meta_user_proxy, "_groupchat_cache", None) or {}
            gc = gc_cache.get(group_name)
            if gc is not None and getattr(gc, "messages", None):
                history = gc.messages
            if not history and agent_list and hasattr(agent_list[0], "chat_messages"):
                for _, messages in agent_list[0].chat_messages.items():
                    history = messages
                    break
            if history:
                break

        if not history and getattr(meta_user_proxy, "chat_messages", None):
            for _, messages in meta_user_proxy.chat_messages.items():
                history = messages
                break
    except Exception as e:  # noqa: BLE001 — a partial transcript beats none
        print(f"[warn] history extraction: {type(e).__name__}: {e}", flush=True)

    cleaned = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not content:
            continue
        name = msg.get("name") or {
            "assistant": "assistant", "user": "user",
            "system": "system", "tool": "tool"}.get(msg.get("role"), "unknown")
        cleaned.append({"content": content, "name": name})
    return cleaned, system_prompts


# The task prompt itself contains a "## ANSWER\n[concise final answer]" block,
# and the captain echoes the instructions to its experts — so a naive scan
# happily "extracts" the template placeholder. Reject those.
PLACEHOLDERS = {"[concise final answer]", "[brief reasoning or evidence used to reach the answer]"}


def extract_answer(history: List[Dict]) -> str:
    """Latest genuine `## ANSWER` block in the transcript."""
    for msg in reversed(history):
        text = msg.get("content")
        if not isinstance(text, str) or "## answer" not in text.lower():
            continue
        _, remainder = re.split(r"##\s*ANSWER", text, maxsplit=1, flags=re.IGNORECASE)
        for line in remainder.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("## reason"):
                break
            if stripped.lower() in PLACEHOLDERS:
                break          # this message is a prompt, not an answer
            return stripped
    return ""


# ── run ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="pool name; resolves to the prepared jsonl")
    src.add_argument("--tasks", help="explicit path to a task jsonl")
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", required=True, help="model key in the config list")
    ap.add_argument("--config-list",
                    default=str(ATTRIBSCOPE_ROOT / "datagen" / "configs" / "oai_config_list.json"))
    ap.add_argument("--build-name", default="datagen_captain")
    ap.add_argument("--build-state", default=None,
                    help="reuse a saved team; built fresh into the run dir if omitted")
    ap.add_argument("--building-task", default=DEFAULT_BUILDING_TASK)
    ap.add_argument("--max-round", type=int, default=20)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    args = ap.parse_args()

    if args.pool:
        args.tasks = str(ATTRIBSCOPE_ROOT / "datagen" / "pools" / "data" / f"{args.pool}.jsonl")
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        raise SystemExit(f"task file not found: {tasks_path}")
    tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    if not 0 <= args.task_index < len(tasks):
        raise SystemExit(f"--task-index {args.task_index} out of range 0..{len(tasks)-1}")
    task = tasks[args.task_index]

    run_dir = Path(args.output_dir)
    # Start from a clean slate: a run killed midway leaves llm_steps and
    # workspace files behind, and step numbering restarts, so a re-run would
    # otherwise interleave two attempts' provenance.
    for stale in (run_dir / "llm_steps", run_dir / "workspace"):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    work_dir = run_dir / "workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "llm_steps").mkdir(parents=True, exist_ok=True)
    # The vendored autogen writes per-call logs here.
    os.environ["LLM_STEP_DIR"] = str(run_dir / "llm_steps")

    general_llm_config, nested_mode_config = build_llm_configs(args, work_dir)

    def _is_terminate(msg: Dict) -> bool:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        return isinstance(content, str) and "terminate" in content.lower()

    meta_agent = MetaAgent(name="captain_agent", llm_config=general_llm_config,
                           nested_mode="autobuild")
    meta_user_proxy = MetaUserProxyAgent(
        name="captain_user_proxy",
        nested_mode_config=nested_mode_config,
        code_execution_config={"use_docker": False, "work_dir": str(work_dir),
                               "last_n_messages": 2},
        default_group_name=args.build_name,
        is_termination_msg=_is_terminate,
        max_consecutive_auto_reply=1,
    )

    print(f"=== {task['id']} ({task['pool']}) via {args.model} ===", flush=True)

    error = None
    started = datetime.now()
    build_state_path = Path(args.build_state) if args.build_state else run_dir / "build_state.json"
    try:
        if args.build_state and build_state_path.exists():
            meta_user_proxy.load_build_state(str(build_state_path))
            print(f"[info] reused team from {build_state_path}", flush=True)
        else:
            meta_user_proxy.build_team(args.build_name, args.building_task)
            build_state_path.parent.mkdir(parents=True, exist_ok=True)
            meta_user_proxy.save_build_state(str(build_state_path), args.building_task)
            print(f"[info] built team -> {build_state_path}", flush=True)

        attachment = ""
        if task.get("file_path"):
            attachment = (f"\n# Attached file\n"
                          f"The question refers to this local file: {task['file_path']}\n")
        prompt = SYSTEM_TEMPLATE.format(
            today=datetime.now().date().isoformat(),
            question=task["question"].strip(),
            attachment=attachment,
            max_round=args.max_round)
        meta_user_proxy.initiate_chat(meta_agent, silent=False, message=prompt)
    except Exception as e:  # noqa: BLE001 — a crashed run is still a data point
        error = f"{type(e).__name__}: {e}"
        print(f"[error] {error}", flush=True)
        traceback.print_exc()

    history, system_prompts = collect_history(meta_user_proxy)
    extracted = extract_answer(history)

    summary = {
        "history": history,
        "question": task["question"],
        "ground_truth": task["answer"],
        "question_ID": task["id"],
        "pool": task["pool"],
        "extracted_answer": extracted,
        "elapsed_s": (datetime.now() - started).total_seconds(),
        "backbone": args.model,
        "agents": sorted(system_prompts),
        "system_prompts": system_prompts,
    }
    if error:
        summary["error"] = error

    # In-run guess only; judge/rejudge.py re-derives the authoritative verdict.
    gt = str(task["answer"]).lower().strip()
    guess = bool(extracted) and (extracted.lower().strip() == gt or gt in extracted.lower())
    (run_dir / "judge.json").write_text(json.dumps(
        {"is_correct": guess, "extracted_answer": extracted,
         "ground_truth": task["answer"], "method": "in-run substring (untrusted)"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # Written last: the orchestrator uses its presence as the done-marker.
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== done: {len(history)} steps, answer={extracted[:80]!r} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
