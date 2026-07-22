#!/usr/bin/env python3
"""Prepare question pools into one uniform jsonl per pool.

Every downstream stage (harness drivers, judging, injection) reads this one
schema, so adding a benchmark means adding a loader here and nothing else:

    {"id": "gsm8k-test-0017", "question": "...", "answer": "72",
     "pool": "gsm8k", "level": null, "answer_type": "numeric",
     "file_path": null}

`answer_type` drives automatic success/failure judging downstream:
    numeric  exact after numeric normalization (gsm8k, math500)
    mcq      option-letter match (arc, mmlu-pro)
    exact    short-answer EM / token-F1 (hotpotqa, 2wikimqa, musique)
    open     LLM judge (gaia, assistantbench)

Run from the repo root:
    python datagen/pools/prepare.py --pool gsm8k
    python datagen/pools/prepare.py --all --limit 500
    python datagen/pools/prepare.py --all --stats     # summarize what exists
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

OUT_DIR = common.POOLS_DIR
# GAIA attachments (spreadsheets, images, audio) live next to the pool jsonl;
# the harness drivers pass `file_path` through to the agents.
GAIA_FILES = OUT_DIR / "gaia_files"


# ── helpers ───────────────────────────────────────────────────────────────────

def _task(pool, idx, question, answer, answer_type, level=None,
          file_path=None, split="test", raw_id=None):
    return {
        "id": f"{pool}-{split}-{idx:05d}",
        "question": (question or "").strip(),
        "answer": str(answer).strip() if answer is not None else "",
        "pool": pool,
        "level": level,
        "answer_type": answer_type,
        "file_path": file_path,
        "raw_id": raw_id,
    }


def _hf(path, name=None, split=None):
    from datasets import load_dataset

    return load_dataset(path, name, split=split)


def _mcq_question(stem: str, options: list[str]) -> tuple[str, list[str]]:
    """Render an MCQ as text plus the letter list, so answers stay comparable."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return (f"{stem.strip()}\n\n{body}\n\n"
            f"Answer with the letter of the correct option."), letters


# ── loaders (one per pool) ────────────────────────────────────────────────────

def load_gsm8k():
    ds = _hf("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        # Gold answers are "<reasoning>\n#### <final>"; keep only the final.
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(_task("gsm8k", i, r["question"], final, "numeric"))
    return out


def load_math500():
    ds = _hf("HuggingFaceH4/MATH-500", split="test")
    return [_task("math500", i, r["problem"], r["answer"], "numeric",
                  level=r.get("level"))
            for i, r in enumerate(ds)]


def load_arc():
    ds = _hf("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i, r in enumerate(ds):
        choices, labels = r["choices"]["text"], r["choices"]["label"]
        key = r["answerKey"]
        if key not in labels:
            continue
        # Some rows label choices 1-4 rather than A-D; re-letter uniformly.
        question, letters = _mcq_question(r["question"], choices)
        answer = letters[labels.index(key)]
        out.append(_task("arc", i, question, answer, "mcq", raw_id=r["id"]))
    return out


def load_mmlu_pro():
    ds = _hf("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i, r in enumerate(ds):
        question, letters = _mcq_question(r["question"], r["options"])
        idx = r.get("answer_index")
        answer = letters[idx] if idx is not None and idx < len(letters) else r["answer"]
        out.append(_task("mmlu-pro", i, question, answer, "mcq",
                         level=r.get("category"), raw_id=str(r.get("question_id"))))
    return out


def load_hotpotqa():
    ds = _hf("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [_task("hotpotqa", i, r["question"], r["answer"], "exact",
                  level=r.get("level"), raw_id=r.get("id"))
            for i, r in enumerate(ds)]


def load_2wikimqa():
    """2WikiMultihopQA dev split.

    The canonical `xanhho/2WikiMultihopQA` repo ships a loading script, which
    `datasets` 4.x refuses to run, so read the raw dev.json from a mirror that
    stores plain files.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("voidful/2WikiMultihopQA", "dev.json", repo_type="dataset")
    rows = json.load(open(path))
    return [_task("2wikimqa", i, r["question"], r["answer"], "exact",
                  level=r.get("type"), raw_id=str(r.get("_id")), split="dev")
            for i, r in enumerate(rows)]


def load_musique():
    ds = _hf("dgslibisey/MuSiQue", split="validation")
    return [_task("musique", i, r["question"], r["answer"], "exact",
                  raw_id=str(r.get("id")))
            for i, r in enumerate(ds)]


def load_assistantbench():
    ds = _hf("AssistantBench/AssistantBench", split="validation")
    return [_task("assistantbench", i, r["task"], r.get("answer", ""), "open",
                  level=r.get("difficulty"), raw_id=r.get("id"), split="dev")
            for i, r in enumerate(ds)]


def load_gaia():
    """GAIA validation, from the ungated smolagents mirror.

    The official `gaia-benchmark/GAIA` repo is gated; the mirror carries the
    same 2023 validation split plus the attachment files, which is what the
    harnesses need (`file_path` is handed to the agents).
    """
    from huggingface_hub import hf_hub_download

    repo = "smolagents/GAIA-annotated"
    meta = hf_hub_download(repo, "2023/validation/metadata.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(meta) if line.strip()]

    GAIA_FILES.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        file_path = None
        if r.get("file_name"):
            try:
                src = hf_hub_download(repo, f"2023/validation/{r['file_name']}",
                                      repo_type="dataset")
                dst = GAIA_FILES / r["file_name"]
                if not dst.exists():
                    dst.write_bytes(Path(src).read_bytes())
                file_path = str(dst.resolve())
            except Exception as e:  # noqa: BLE001 — keep the task, drop the file
                print(f"  [warn] gaia {r['task_id']}: attachment "
                      f"{r['file_name']} unavailable ({type(e).__name__})")
        out.append(_task("gaia", i, r["Question"], r.get("Final answer", ""),
                         "open", level=r.get("Level"), file_path=file_path,
                         raw_id=r["task_id"], split="val"))
    return out


LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "arc": load_arc,
    "mmlu-pro": load_mmlu_pro,
    "hotpotqa": load_hotpotqa,
    "2wikimqa": load_2wikimqa,
    "musique": load_musique,
    "gaia": load_gaia,
    "assistantbench": load_assistantbench,
}


# ── validation + IO ───────────────────────────────────────────────────────────

def validate(tasks: list[dict], pool: str) -> list[dict]:
    """Drop unusable rows; a task with no question or no answer cannot be judged."""
    kept, dropped = [], 0
    seen: set[str] = set()
    for t in tasks:
        if not t["question"] or not t["answer"]:
            dropped += 1
            continue
        if t["question"] in seen:      # exact-duplicate prompts
            dropped += 1
            continue
        seen.add(t["question"])
        kept.append(t)
    if dropped:
        print(f"  dropped {dropped} rows (empty question/answer or duplicate)")
    assert kept, f"pool {pool} produced no usable tasks"
    return kept


def write_pool(pool: str, tasks: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = common.pool_path(pool)
    with path.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    return path


def summarize(pool: str, tasks: list[dict]) -> str:
    lens = sorted(len(t["question"]) for t in tasks)
    med = lens[len(lens) // 2]
    files = sum(1 for t in tasks if t["file_path"])
    extra = f", {files} with attachments" if files else ""
    return (f"{pool:15s} n={len(tasks):5d}  answer_type={tasks[0]['answer_type']:8s}"
            f"  median question {med} chars{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", dest="pools",
                    choices=sorted(LOADERS), help="pool to prepare (repeatable)")
    ap.add_argument("--all", action="store_true", help="prepare every pool")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap tasks per pool (applied after validation)")
    ap.add_argument("--stats", action="store_true",
                    help="only summarize already-prepared pools")
    ap.add_argument("--force", action="store_true",
                    help="re-prepare pools whose jsonl already exists")
    args = ap.parse_args()

    if args.stats:
        for pool in sorted(LOADERS):
            path = common.pool_path(pool)
            if path.exists():
                print(summarize(pool, common.load_pool(pool)))
            else:
                print(f"{pool:15s} (not prepared)")
        return 0

    pools = sorted(LOADERS) if args.all else (args.pools or [])
    if not pools:
        ap.error("pass --pool <name> (repeatable), --all, or --stats")

    failures = []
    for pool in pools:
        path = common.pool_path(pool)
        if path.exists() and not args.force:
            print(f"[skip] {pool}: {path} exists (--force to redo)")
            continue
        print(f"[prep] {pool} ...")
        try:
            tasks = validate(LOADERS[pool](), pool)
        except Exception as e:  # noqa: BLE001 — one bad pool must not sink the batch
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures.append(pool)
            continue
        if args.limit:
            tasks = tasks[: args.limit]
        write_pool(pool, tasks)
        print("  " + summarize(pool, tasks))

    if failures:
        print(f"\nfailed pools: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
