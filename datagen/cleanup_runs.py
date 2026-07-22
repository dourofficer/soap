#!/usr/bin/env python3
"""Triage collected runs and delete the ones that can never be useful.

Resume keys on `summary.json` existing, so a run that finished *badly* looks
done and is skipped forever. This finds those and (with --delete) removes them
so the next collect pass retries the task.

Categories:

  partial        no summary.json — killed midway. NEVER touched: resume already
                 re-runs these, and the drivers wipe stale llm_steps/workspace.
  startup-crash  top-level `error` and almost no transcript (e.g. the chromadb
                 thread exhaustion).
  ended-on-error the transcript's last turn is an `Error: …` and no answer was
                 extracted — the run died on an infrastructure fault rather
                 than reasoning badly (e.g. the seek_experts_help tool-call bug).
  too-short      fewer than --min-steps turns; no attribution signal.
  keep           everything else, including runs that hit errors mid-flight but
                 kept going. Those are genuine failure trajectories and are
                 exactly what the corpus wants.

    python datagen/cleanup_runs.py                     # report only
    python datagen/cleanup_runs.py --delete
    python datagen/cleanup_runs.py --harness captain --delete
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen import common  # noqa: E402

DELETABLE = ("startup-crash", "ended-on-error", "too-short")


def classify(run_dir: Path, min_steps: int) -> tuple[str, str]:
    """Return (category, one-line reason)."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        return "partial", "no summary.json — resume will re-run it"

    try:
        d = json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return "startup-crash", f"unreadable summary.json ({type(e).__name__})"

    history = d.get("history") or []
    answer = (d.get("extracted_answer") or "").strip()

    if d.get("error") and len(history) < min_steps:
        return "startup-crash", str(d["error"]).splitlines()[0][:90]

    if history:
        last = history[-1].get("content")
        if isinstance(last, str) and last.lstrip().startswith("Error:") and not answer:
            return "ended-on-error", last.strip().splitlines()[0][:90]

    if len(history) < min_steps:
        return "too-short", f"{len(history)} turns < min-steps {min_steps}"

    return "keep", f"{len(history)} turns"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=None,
                    help="default: out_root from configs/collect-qwen.yaml")
    ap.add_argument("--harness"); ap.add_argument("--backbone"); ap.add_argument("--pool")
    ap.add_argument("--min-steps", type=int, default=2,
                    help="transcripts shorter than this carry no attribution "
                         "signal (default: %(default)s)")
    ap.add_argument("--delete", action="store_true",
                    help="actually remove the broken run directories")
    args = ap.parse_args()

    root = Path(args.runs_root) if args.runs_root else \
        common.REPO_ROOT / common.load_cfg("collect-qwen")["out_root"]
    if not root.exists():
        raise SystemExit(f"no runs root at {root}")

    counts: Counter = Counter()
    by_cell: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    doomed: list[Path] = []

    for run_dir in sorted(p for p in root.glob("*/*/*/*") if p.is_dir()):
        harness, backbone, pool = run_dir.parts[-4], run_dir.parts[-3], run_dir.parts[-2]
        if args.harness and harness != args.harness:
            continue
        if args.backbone and backbone != args.backbone:
            continue
        if args.pool and pool != args.pool:
            continue

        cat, why = classify(run_dir, args.min_steps)
        counts[cat] += 1
        by_cell[f"{harness}/{pool}"][cat] += 1
        if cat in DELETABLE:
            reasons[cat][why] += 1
            doomed.append(run_dir)

    total = sum(counts.values())
    print(f"{total} run dirs under {root}\n")
    for cat in ("keep", "partial", *DELETABLE):
        if counts[cat]:
            note = "  (left alone)" if cat == "partial" else \
                   "  (deletable)" if cat in DELETABLE else ""
            print(f"  {counts[cat]:6d}  {cat}{note}")

    if reasons:
        print("\nwhy runs are deletable:")
        for cat in DELETABLE:
            for why, n in reasons[cat].most_common(5):
                print(f"  {n:6d}  [{cat}] {why}")

    print("\nper cell:")
    for cell, c in sorted(by_cell.items()):
        bad = sum(c[k] for k in DELETABLE)
        print(f"  {cell:28s} keep={c['keep']:5d} partial={c['partial']:5d} deletable={bad:5d}")

    if not doomed:
        print("\nnothing to delete.")
        return 0

    if not args.delete:
        print(f"\n{len(doomed)} run dirs would be removed. Re-run with --delete, "
              f"then re-run collection to retry those tasks.")
        return 0

    for run_dir in doomed:
        shutil.rmtree(run_dir, ignore_errors=True)
    print(f"\nremoved {len(doomed)} run dirs. Now re-run collection to retry them:")
    print("  python datagen/collect/run_batch.py --config collect-qwen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
