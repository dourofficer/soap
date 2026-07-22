#!/usr/bin/env python3
"""Run ONE task from a prepared datagen pool through Magentic-One.

Adapted from run_assistant_bench.py. Differences that matter:

* **One task per process.** The LLM-logging monkeypatch keeps its state in
  module globals, so a single process must own a single task — otherwise steps
  bleed between tasks and a timeout kill leaves a half-written run. The batch
  orchestrator launches one of these per task.
* **Configurable team.** `--agents` selects which agents join the group chat.
  `surfer` (MultimodalWebSurfer) is opt-in and runs in TEXT mode: with
  `model_info["vision"]=False` it reads pages from the DOM instead of
  screenshots. It attaches over CDP to a shared Chromium container
  (`--cdp-url`), because this host has no browser and no root to install one.
* **`model_info` override.** autogen refuses to build a client for a model
  name it does not recognise, which is every locally-served name.
* **Pool-agnostic.** Reads the uniform task schema written by
  datagen/pools/prepare.py; nothing here is AssistantBench-specific.

Output (one run directory, matching the datagen raw-run layout):
    summary.json          {history:[{role,content}], question, ground_truth, ...}
    judge.json            in-run correctness guess (untrusted; rejudge.py wins)
    llm_steps/step_N.json one file per real LLM call

Usage (normally via datagen/collect/run_batch.py):
    python run_pool.py --pool gsm8k --task-index 0 --output-dir /path/to/run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse, CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

# `.env` must not clobber the endpoint the orchestrator injects.
load_dotenv(override=False)

# Set by run(); the logging patch writes here.
llm_call_logs: list[dict] = []
current_steps_dir: Path | None = None
_llm_client_patched = False


PROMPT_TEMPLATE = """Today's date is {today}.
System: {system_info}. Do not use sudo if you need to run commands.

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


# ── LLM call logging ──────────────────────────────────────────────────────────

def patch_llm_client_for_logging() -> None:
    """Record every real LLM request/response to `current_steps_dir`.

    Unlike the upstream driver — which logs `repr(ChatCompletion(...))` — this
    writes plain JSON, so the logs stay machine-readable for later provenance
    work without a repr-parsing step.
    """
    global _llm_client_patched
    if _llm_client_patched:
        return

    original_create = OpenAIChatCompletionClient.create

    async def logged_create(self, messages, *args, **kwargs):
        response = await original_create(self, messages, *args, **kwargs)
        if current_steps_dir is None:
            return response

        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role, content = msg.get("role", "unknown"), msg.get("content", "")
            else:
                role = (getattr(msg, "role", None) or getattr(msg, "source", None)
                        or getattr(msg, "type", None) or "unknown")
                content = getattr(msg, "content", None)
                if content is None:
                    content = str(msg)
            formatted.append({"role": str(role),
                              "content": content if isinstance(content, str) else str(content)})

        content = getattr(response, "content", None)
        entry = {
            "timestamp": datetime.now().timestamp(),
            "request": {"model": getattr(self, "_raw_config", {}).get("model", ""),
                        "messages": formatted},
            "response": {
                "content": content if isinstance(content, str) else str(content),
                "finish_reason": getattr(response, "finish_reason", None),
                "usage": {
                    "prompt_tokens": getattr(getattr(response, "usage", None), "prompt_tokens", None),
                    "completion_tokens": getattr(getattr(response, "usage", None), "completion_tokens", None),
                },
                "thought": getattr(response, "thought", None),
            },
        }
        llm_call_logs.append(entry)
        step_file = current_steps_dir / f"step_{len(llm_call_logs)}.json"
        step_file.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        return response

    OpenAIChatCompletionClient.create = logged_create
    _llm_client_patched = True


# ── History formatting (kept faithful to the upstream schema) ─────────────────

LEDGER_MARKERS = ("GIVEN OR VERIFIED FACTS", "Here is the plan", "Updated Ledger",
                  "fact sheet", "Next speaker")


def content_to_text(content: Any) -> str:
    """Flatten a message body to text, replacing images with a stable marker.

    WebSurfer replies are `MultiModalMessage`s whose content is
    `[text, Image]` — and the screenshot is attached unconditionally, even in
    text mode (`_multimodal_web_surfer.py:817`). Plain `str()` on that list
    yields `<autogen_core._image.Image object at 0x7f19…>`, embedding a memory
    address that differs on every run. Substituting a fixed marker keeps the
    trajectory readable and deterministic.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif type(p).__name__ == "Image":
                # autogen_core.Image; size is not always cheaply available.
                parts.append("[screenshot omitted]")
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def format_history(messages: list[Any], task: dict) -> dict:
    """Messages → the `summary.json` schema the datagen converter consumes.

    Roles follow the Who&When conventions the attribscope loader expects:
    `human`, `Orchestrator (thought)`, `Orchestrator (-> Agent)`, the bare
    agent name, and `Orchestrator (termination condition)`.
    """
    history = []
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        content = content_to_text(content if content is not None else msg)

        # End-of-run object carrying the whole conversation + stop reason.
        if content.startswith(("messages=[", "[TextMessage(", "[MultiModalMessage(")):
            reason = getattr(msg, "stop_reason", None) or "Task terminated"
            history.append({"content": str(reason),
                            "role": "Orchestrator (termination condition)"})
            continue

        source = getattr(msg, "source", None)
        if source is None:
            role = "unknown"
        elif source == "user":
            role = "human"
        elif source == "MagenticOneOrchestrator":
            if any(m in content for m in LEDGER_MARKERS):
                role = "Orchestrator (thought)"
            else:
                # An instruction addressed to whoever speaks next.
                nxt = getattr(messages[i + 1], "source", None) if i + 1 < len(messages) else None
                role = (f"Orchestrator (-> {nxt})"
                        if nxt and nxt != "MagenticOneOrchestrator" else "Orchestrator (thought)")
        else:
            role = source
        history.append({"content": content, "role": role})

    return {
        "history": history,
        "question": task["question"],
        "ground_truth": task["answer"],
        "question_ID": task["id"],
        "pool": task["pool"],
        "is_corrected": False,
    }


# The task prompt itself contains "## ANSWER\n[concise final answer]", and the
# orchestrator echoes the task when planning — so a naive reverse scan happily
# "extracts" the template placeholder from step 0 or 1.
PLACEHOLDERS = {"[concise final answer]",
                "[brief reasoning or evidence used to reach the answer]"}


def extract_answer(messages: list[Any]) -> str:
    """Pull the `## ANSWER` block from the latest message that has a real one."""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        text = content_to_text(content if content is not None else msg)
        if "## ANSWER" not in text:
            continue
        lines, collecting, out = text.split("\n"), False, []
        for line in lines:
            if "## ANSWER" in line:
                collecting = True
                continue
            if collecting and line.startswith("##"):
                break
            if collecting and line.strip():
                if line.strip().lower() in PLACEHOLDERS:
                    out = []          # this message is a prompt, not an answer
                    break
                out.append(line.strip())
        if out:
            return " ".join(out).strip()
    return ""


# ── Team construction ─────────────────────────────────────────────────────────

def build_model_client(args) -> OpenAIChatCompletionClient:
    """Client for a locally served model.

    autogen has no built-in profile for these model names, so `model_info` is
    mandatory. `function_calling` is a CLI flag because it is a property of the
    served model, not of the harness: DeepSeek-R1-Distill is not tool-call
    trained and its endpoint runs without a tool parser.
    """
    return OpenAIChatCompletionClient(
        model=os.environ["M1_MODEL"],
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ["OPENAI_API_BASE"],
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        model_info={
            "vision": False,
            "function_calling": args.function_calling,
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )


def approval_func(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(approved=True, reason="Auto-approved for benchmark execution")


async def connect_browser(cdp_url: str):
    """Attach to the shared Chromium container and open an isolated context.

    This box has no browser and no root to install one, so Chromium runs in the
    Playwright image and we attach over CDP. Passing both `playwright` and
    `context` into MultimodalWebSurfer makes its `_lazy_init` skip every local
    launch path (`_multimodal_web_surfer.py:318,322`) and just call
    `context.new_page()`.

    One container is shared across concurrent tasks; each task gets its own
    context, so cookies and storage stay isolated.
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(cdp_url)
    context = await browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
        locale="en-US",
    )
    return pw, browser, context


async def build_team(args, client, work_dir: Path):
    """Assemble the group chat from `--agents`.

    Returns (team, browser_resources) — the caller must close the resources.
    """
    wanted = [a.strip() for a in args.agents.split(",") if a.strip()]
    agents, browser_res = [], None

    for name in wanted:
        if name == "fs":
            agents.append(FileSurfer("FileSurfer", model_client=client))
        elif name == "coder":
            agents.append(MagenticOneCoderAgent("Coder", model_client=client))
        elif name == "executor":
            work_dir.mkdir(parents=True, exist_ok=True)
            agents.append(CodeExecutorAgent(
                "ComputerTerminal",
                code_executor=LocalCommandLineCodeExecutor(work_dir=str(work_dir)),
                approval_func=approval_func))
        elif name == "surfer":
            from autogen_ext.agents.web_surfer import MultimodalWebSurfer

            # Text mode. The surfer branches on model_info["vision"]: with it
            # False it drives WEB_SURFER_TOOL_PROMPT_TEXT off the DOM
            # (`get_visible_text` / `get_page_markdown`) and an aria-labelled
            # target list, and skips the 1105-token screenshot per step. Page
            # text comes from the DOM either way, so nothing is OCR'd back out
            # of pixels. Note the surfer hard-requires function calling
            # (`_multimodal_web_surfer.py:235`).
            pw, browser, context = await connect_browser(args.cdp_url)
            browser_res = (pw, browser, context)
            agents.append(MultimodalWebSurfer(
                "WebSurfer", model_client=client,
                playwright=pw, context=context,
                headless=True, animate_actions=False,
                to_save_screenshots=False,
                start_page=args.start_page,
            ))
        else:
            raise SystemExit(
                f"unknown agent {name!r}; choose from fs,coder,executor,surfer")

    if not agents:
        raise SystemExit("--agents selected no agents")
    return (MagenticOneGroupChat(agents, model_client=client,
                                 max_turns=args.max_round), browser_res)


async def close_browser(browser_res) -> None:
    """Best-effort teardown; a leaked context would pile up in the container."""
    if not browser_res:
        return
    pw, browser, context = browser_res
    for closer in (context.close, browser.close, pw.stop):
        try:
            await closer()
        except Exception:  # noqa: BLE001 — teardown must not mask a run result
            pass


# ── Run ───────────────────────────────────────────────────────────────────────

def build_prompt(task: dict, max_round: int) -> str:
    attachment = ""
    if task.get("file_path"):
        attachment = (f"\n# Attached file\n"
                      f"The question refers to this local file: {task['file_path']}\n")
    return PROMPT_TEMPLATE.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        system_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        question=task["question"].strip(),
        attachment=attachment,
        max_round=max_round,
    )


async def run(args) -> int:
    global llm_call_logs, current_steps_dir

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if not 0 <= args.task_index < len(tasks):
        raise SystemExit(f"--task-index {args.task_index} out of range 0..{len(tasks)-1}")
    task = tasks[args.task_index]

    run_dir = Path(args.output_dir)
    # Start from a clean slate. A run killed midway leaves llm_steps/step_N and
    # workspace files behind; step numbering restarts at 1, so without this the
    # directory would interleave two attempts' provenance.
    for stale in (run_dir / "llm_steps", run_dir / "workspace"):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    steps_dir = run_dir / "llm_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    llm_call_logs = []
    current_steps_dir = steps_dir

    patch_llm_client_for_logging()
    client = build_model_client(args)
    team, browser_res = await build_team(args, client, run_dir / "workspace")

    print(f"=== {task['id']} ({task['pool']}) via {os.environ['M1_MODEL']} "
          f"agents={args.agents} ===", flush=True)

    messages: list[Any] = []
    started = datetime.now()
    error = None
    try:
        async for message in team.run_stream(task=build_prompt(task, args.max_round)):
            messages.append(message)
            print(f"[step {len(messages)}] {str(message)[:400]}", flush=True)
    except Exception as e:  # noqa: BLE001 — a crashed run is still a data point
        error = f"{type(e).__name__}: {e}"
        print(f"[error] {error}", flush=True)
        traceback.print_exc()
    finally:
        await close_browser(browser_res)
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass

    summary = format_history(messages, task)
    extracted = extract_answer(messages)
    summary["extracted_answer"] = extracted
    summary["elapsed_s"] = (datetime.now() - started).total_seconds()
    summary["n_llm_calls"] = len(llm_call_logs)
    summary["backbone"] = os.environ["M1_MODEL"]
    summary["agents"] = args.agents
    if error:
        summary["error"] = error

    # In-run guess only. rejudge.py re-derives the authoritative verdict; this
    # substring rule is far too permissive for short answers.
    gt = str(task["answer"]).lower().strip()
    guess = bool(extracted) and (extracted.lower().strip() == gt or gt in extracted.lower())
    (run_dir / "judge.json").write_text(json.dumps(
        {"is_correct": guess, "extracted_answer": extracted,
         "ground_truth": task["answer"], "method": "in-run substring (untrusted)"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # summary.json is written last: the orchestrator treats its presence as the
    # done-marker, so a partial run must never look complete.
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== done: {len(summary['history'])} steps, {len(llm_call_logs)} llm calls, "
          f"answer={extracted[:80]!r} ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="pool name; resolves to the prepared jsonl")
    src.add_argument("--tasks", help="explicit path to a task jsonl")
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--agents", default="fs,coder,executor",
                    help="comma list from fs,coder,executor,surfer (default: %(default)s)")
    ap.add_argument("--cdp-url",
                    default=os.getenv("DATAGEN_CDP_URL", "http://127.0.0.1:9222"),
                    help="CDP endpoint of the shared browser container "
                         "(only used when `surfer` is in --agents)")
    ap.add_argument("--start-page", default="https://duckduckgo.com/",
                    help="WebSurfer landing page (default: %(default)s)")
    ap.add_argument("--max-round", type=int, default=20)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--function-calling", dest="function_calling",
                    action="store_true", default=True)
    ap.add_argument("--no-function-calling", dest="function_calling",
                    action="store_false",
                    help="for backbones served without a tool-call parser")
    args = ap.parse_args()

    if args.pool:
        # Magentic-One → agent_system → code → TraceElephant → datagen → repo root
        repo_root = Path(__file__).resolve().parents[5]
        args.tasks = str(repo_root / "datagen" / "pools" / "data" / f"{args.pool}.jsonl")
    if not Path(args.tasks).exists():
        raise SystemExit(f"task file not found: {args.tasks}")
    for var in ("M1_MODEL", "OPENAI_API_BASE"):
        if not os.getenv(var):
            raise SystemExit(f"{var} must be set (the orchestrator injects it)")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
