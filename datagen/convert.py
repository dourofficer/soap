#!/usr/bin/env python3
"""Convert raw datagen run directories into the attribscope trajectory schema.

Reads `datagen/runs/<harness>/<backbone>/<pool>/<task>/summary.json` and writes
`data/<dataset>/<subset>/<N>.json` in the schema `src.data.trajectory
.load_dataset` consumes — the same shape `data/convert_traceelephant.py`
produces for the TraceElephant traces, including its validation invariants.

Subsets encode `<harness>-<backbone>`, because the generator backbone is a
different axis from the manifest `models:` field (which lists the *proxy*
models used for representation extraction).

Unlabeled trajectories carry `mistake_step = -1` and an empty `mistake_agent`.
Those are FIT-ONLY data: `compute_metrics` skips unlabeled trajectories when
counting hits but still counts them in the denominator, so they must never sit
in a split that reports metrics.

Run from the repo root:
    python datagen/convert.py --outcome fail
    python datagen/convert.py --runs-root datagen/runs --mixed --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen import common  # noqa: E402

# Subset-name aliases: the served model name is clumsy inside a path. Sizes are
# explicit — with two Qwen3.5 backbones in play, a bare "qwen" would be
# ambiguous. An unmapped backbone falls through to its served name unchanged.
BACKBONE_ALIAS = {
    "qwen3.5-9b": "qwen9b",
    "qwen3.5-35b-a3b": "qwen35b",
    "deepseek-8b": "dsr1",
}

# Serialized tool-call arguments can be enormous; cap them so files stay
# readable (same rationale and limit as data/convert_traceelephant.py).
ARG_TRUNC = 2000

SYSTEM_INTRO = {
    "magentic": (
        "Magentic-One: a multi-agent team led by an Orchestrator that maintains a "
        "task ledger, plans, and delegates to specialist agents.\n\n"
        "Orchestrator: plans, tracks progress, selects the next speaker, and "
        "reports the final answer.\n"
        "FileSurfer: opens and reads local files.\n"
        "Coder: writes code and reasons about the task.\n"
        "ComputerTerminal: executes code and returns its output."
    ),
    "captain": (
        "Captain-Agent: a Captain agent that assembles a task-specific expert team "
        "and delegates sub-tasks to it, with a user proxy executing tool calls."
    ),
}


# ── history normalization ─────────────────────────────────────────────────────

def _content_of(turn: dict) -> str:
    content = turn.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts
        parts = [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
                 for p in content]
        text = "\n".join(parts)
    else:
        text = json.dumps(content, ensure_ascii=False) if content is not None else ""
    return text[:ARG_TRUNC] + "\n... [truncated]" if len(text) > ARG_TRUNC else text


def normalize_history(raw: list[dict], harness: str) -> list[dict]:
    """Return `[{role, content}]`, whichever key the harness used for the agent.

    Magentic-One already emits `{role, content}`; Captain-Agent emits
    `{content, name}` (its driver strips `role` deliberately).
    """
    out = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("name") or ""
        content = _content_of(turn)
        if not role or not content.strip():
            continue
        out.append({"role": str(role), "content": content})
    return out


# ── run discovery ─────────────────────────────────────────────────────────────

def iter_runs(runs_root: Path):
    """Yield (harness, backbone, pool, run_dir) for every completed run.

    A run counts as complete only if `summary.json` exists — the drivers write
    it last, so a killed run never looks finished.
    """
    for summary in sorted(runs_root.glob("*/*/*/*/summary.json")):
        run_dir = summary.parent
        pool_dir = run_dir.parent
        yield (pool_dir.parent.parent.name, pool_dir.parent.name,
               pool_dir.name, run_dir)


def verdict_of(run_dir: Path) -> tuple[bool | None, str]:
    """Authoritative correctness for a run.

    `verdict.json` (written by judge/rejudge.py) wins; the driver's own
    `judge.json` is a permissive substring guess and is only a fallback.
    """
    v = run_dir / "verdict.json"
    if v.exists():
        d = json.loads(v.read_text())
        return bool(d.get("is_correct")), d.get("method", "verdict.json")
    j = run_dir / "judge.json"
    if j.exists():
        d = json.loads(j.read_text())
        return bool(d.get("is_correct")), "judge.json (untrusted fallback)"
    return None, "no verdict"


# ── conversion ────────────────────────────────────────────────────────────────

def convert_run(harness: str, backbone: str, pool: str, run_dir: Path,
                stats: Counter, outcome: str = "unknown") -> dict | None:
    summary = json.loads((run_dir / "summary.json").read_text())
    history = normalize_history(summary.get("history", []), harness)

    if not history:
        stats["skipped_empty_history"] += 1
        return None

    subset = f"{harness}-{BACKBONE_ALIAS.get(backbone, backbone)}"
    return {
        "question_ID": f"{pool}/{summary.get('question_ID', run_dir.name)}",
        "question": summary.get("question", ""),
        "ground_truth": summary.get("ground_truth", ""),
        "history": history,
        # Unlabeled: fit-only. Phase-2 injection fills these in.
        "mistake_agent": "",
        "mistake_step": -1,
        "mistake_reason": "",
        "level": summary.get("level", -1) if summary.get("level") is not None else -1,
        "system": SYSTEM_INTRO.get(harness, harness),
        "subset": subset,
        # Provenance — the loader ignores unknown keys, so a trajectory always
        # says where it came from: task pool, agent system, generator backbone,
        # and how the run ended ("success" / "fail" / "unknown" when unjudged).
        "pool": pool,
        "backbone": backbone,
        "harness": harness,
        "outcome": outcome,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=str(common.RUNS_DIR))
    ap.add_argument("--out-root", default="data/synthetic",
                    help="dataset root (default: %(default)s)")
    ap.add_argument("--outcome", choices=["fail", "success", "all"], default="fail",
                    help="which runs to convert. 'fail' matches the all-failure "
                         "distribution of the benchmarks; 'all' takes every valid "
                         "run regardless of verdict (SVD fitting is unsupervised) "
                         "and needs no rejudge pass (default: %(default)s)")
    ap.add_argument("--min-steps", type=int, default=2,
                    help="drop trajectories with fewer turns (default: %(default)s)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--exclude-pools", default="",
                    help="comma list; e.g. gaia, whose questions also underlie "
                         "benchmark test trajectories")
    ap.add_argument("--mixed", action="store_true",
                    help="also materialize a merged `mixed` subset for a "
                         "combined SVD fit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"no runs root at {runs_root}")
    excluded = {p.strip() for p in args.exclude_pools.split(",") if p.strip()}

    stats = Counter()
    by_subset: dict[str, list[tuple[dict, Path]]] = defaultdict(list)

    for harness, backbone, pool, run_dir in iter_runs(runs_root):
        stats["runs_found"] += 1
        if pool in excluded:
            stats["skipped_excluded_pool"] += 1
            continue

        is_correct, method = verdict_of(run_dir)
        if args.outcome == "all":
            # SVD fitting is unsupervised, so no verdict is required — every
            # structurally valid run converts. The outcome is still recorded
            # when one is available (verdict.json, else the in-run judge.json).
            if is_correct is not None:
                stats[f"verdict_via_{method.split()[0]}"] += 1
        else:
            if is_correct is None:
                stats["skipped_no_verdict"] += 1
                continue
            if args.outcome == "fail" and is_correct:
                stats["skipped_success"] += 1
                continue
            if args.outcome == "success" and not is_correct:
                stats["skipped_failure"] += 1
                continue
            stats[f"verdict_via_{method.split()[0]}"] += 1

        outcome = "unknown" if is_correct is None else ("success" if is_correct else "fail")
        stats[f"outcome_{outcome}"] += 1
        traj = convert_run(harness, backbone, pool, run_dir, stats, outcome)
        if traj is None:
            continue
        n = len(traj["history"])
        if n < args.min_steps:
            stats["skipped_too_short"] += 1
            continue
        if args.max_steps and n > args.max_steps:
            stats["skipped_too_long"] += 1
            continue

        by_subset[traj["subset"]].append((traj, run_dir))
        stats["converted"] += 1

    if args.mixed and by_subset:
        by_subset["mixed"] = [pair for s, pairs in sorted(by_subset.items())
                              for pair in pairs if s != "mixed"]

    out_root = Path(args.out_root)
    for subset, pairs in sorted(by_subset.items()):
        lens = sorted(len(t["history"]) for t, _ in pairs)
        print(f"{subset:18s} n={len(pairs):5d}  steps min/median/max="
              f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
        if args.dry_run:
            continue
        sub_dir = out_root / subset
        sub_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (traj, run_dir) in enumerate(pairs):
            (sub_dir / f"{i}.json").write_text(
                json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"file": f"{i}.json", "question_ID": traj["question_ID"],
                         "pool": traj["pool"], "backbone": traj["backbone"],
                         "harness": traj["harness"], "outcome": traj["outcome"],
                         "run_dir": str(run_dir)})
        with (sub_dir / "filename_map.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\nstats: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if args.dry_run:
        print("(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
