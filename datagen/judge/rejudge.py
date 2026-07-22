#!/usr/bin/env python3
"""Derive the authoritative success/failure verdict for each collected run.

The harness drivers write their own `judge.json`, but those are permissive
substring matches — Magentic-One's counts "18" as correct inside "1183". Every
downstream decision (which trajectories become the unlabeled fit corpus, which
become Phase-2 injection candidates) depends on this verdict, so it is
recomputed here from `summary.json` and written to `verdict.json`.

Dispatch is by the pool's `answer_type`: numeric/mcq/exact are checked
programmatically; `open` (gaia, assistantbench) and borderline exact-match
cases go to an LLM judge on a configurable endpoint.

    python datagen/judge/rejudge.py
    python datagen/judge/rejudge.py --pool gaia --force
    python datagen/judge/rejudge.py --no-llm --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402
from datagen.judge import checkers  # noqa: E402

JUDGE_PROMPT = """You are a strict judge. Given a ground truth answer and a model answer, decide if they match semantically.
Respond ONLY with JSON:
{{"is_correct": true/false, "reason": "brief justification"}}
Do not add extra text.
Question: {question}
Ground truth: {gold}
Model answer: {answer}"""

_client_lock = threading.Lock()
_client = None


def get_client(ep: dict):
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(base_url=ep["base_url"], api_key=ep["api_key"], timeout=180.0)
    return _client


def llm_judge(question: str, extracted: str, gold: str, ep: dict) -> tuple[bool | None, str]:
    if not extracted:
        return False, "no answer extracted"
    try:
        r = get_client(ep).chat.completions.create(
            model=ep["model"], temperature=0.0, max_tokens=2000,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question[:1500], gold=gold[:500], answer=extracted[:1500])}],
        )
        text = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None, f"judge returned no JSON: {text[:80]!r}"
        obj = json.loads(text[start:end + 1])
        return bool(obj.get("is_correct")), f"llm-judge: {str(obj.get('reason'))[:160]}"
    except Exception as e:  # noqa: BLE001 — an undecided run is better than a crash
        return None, f"llm-judge failed: {type(e).__name__}: {e}"


def judge_run(run_dir: Path, pool_index: dict, ep: dict | None,
              f1_threshold: float, force: bool) -> dict:
    out = run_dir / "verdict.json"
    if out.exists() and not force:
        return {"status": "skipped"}

    summary = json.loads((run_dir / "summary.json").read_text())
    pool = summary.get("pool") or run_dir.parent.name
    gold = str(summary.get("ground_truth", ""))
    extracted = str(summary.get("extracted_answer", ""))
    question = str(summary.get("question", ""))

    task = pool_index.get(pool, {}).get(summary.get("question_ID", ""))
    answer_type = (task or {}).get("answer_type", "open")

    checker = checkers.CHECKERS.get(answer_type)
    if checker is checkers.check_exact:
        is_correct, reason = checker(extracted, gold, f1_threshold)
    elif checker is not None:
        is_correct, reason = checker(extracted, gold)
    else:
        is_correct, reason = None, "open-ended: llm judge"

    method = f"checker:{answer_type}"
    if is_correct is None:
        if ep is None:
            # Undecidable without the judge: treat as failure, but say so, since
            # a wrong "correct" would pollute the Phase-2 injection candidates.
            is_correct, reason = False, f"{reason}; llm judge disabled"
            method = f"checker:{answer_type}+no-llm"
        else:
            is_correct, reason = llm_judge(question, extracted, gold, ep)
            method = f"checker:{answer_type}+llm"
            if is_correct is None:
                is_correct = False
                method += "(undecided->fail)"

    verdict = {"is_correct": bool(is_correct), "method": method,
               "extracted": extracted, "ground_truth": gold, "reason": reason}
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok", "is_correct": bool(is_correct), "pool": pool,
            "answer_type": answer_type, "method": method}


def build_pool_index(pools: list[str]) -> dict:
    """{pool: {task_id: task}} so a run can find its own answer_type."""
    index = {}
    for pool in pools:
        try:
            index[pool] = {t["id"]: t for t in common.load_pool(pool)}
        except FileNotFoundError:
            print(f"[warn] pool {pool!r} not prepared; runs default to answer_type=open")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Only `judge` and `out_root` are read, and both backbones share them, so
    # either per-model config serves; judging is a single pass over all runs.
    ap.add_argument("--config", default="collect-qwen")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--pool"); ap.add_argument("--harness"); ap.add_argument("--backbone")
    ap.add_argument("--force", action="store_true", help="re-judge runs that already have a verdict")
    ap.add_argument("--no-llm", action="store_true", help="programmatic checkers only")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = common.load_cfg(args.config)
    runs_root = Path(args.runs_root) if args.runs_root else common.REPO_ROOT / cfg["out_root"]
    if not runs_root.exists():
        raise SystemExit(f"no runs root at {runs_root}")

    runs = []
    for summary in sorted(runs_root.glob("*/*/*/*/summary.json")):
        d = summary.parent
        harness, backbone, pool = d.parent.parent.parent.name, d.parent.parent.name, d.parent.name
        if args.harness and harness != args.harness:
            continue
        if args.backbone and backbone != args.backbone:
            continue
        if args.pool and pool != args.pool:
            continue
        runs.append(d)

    if not runs:
        print("no runs found")
        return 1

    pools = sorted({d.parent.name for d in runs})
    pool_index = build_pool_index(pools)
    ep = None if args.no_llm else common.resolve_endpoint(cfg["judge"]["endpoint"])
    f1_threshold = cfg["judge"].get("f1_threshold", 0.6)

    judge_note = "llm judge disabled" if ep is None else f"llm judge = {ep['model']}"
    print(f"{len(runs)} runs across pools {pools}; {judge_note}")
    if args.dry_run:
        return 0

    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
        futures = [pool_exec.submit(judge_run, d, pool_index, ep, f1_threshold, args.force)
                   for d in runs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["status"] == "skipped":
                counts["skipped"] += 1
            else:
                counts["correct" if r["is_correct"] else "incorrect"] += 1
                counts[f"via:{r['method']}"] += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(runs)}]")

    total = counts["correct"] + counts["incorrect"]
    print("\nverdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if total:
        print(f"failure rate: {counts['incorrect']}/{total} "
              f"({100*counts['incorrect']/total:.1f}%) -> the unlabeled corpus")
    print("next: python datagen/convert.py --outcome fail --mixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
