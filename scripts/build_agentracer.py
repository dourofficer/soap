"""Convert AgenTracer's TracerTraj code-test parquet into this repo's corpus layout.

AgenTracer ships one trajectory per parquet row; SOAP reads one JSON file per
trajectory under ``data/<ds>/<subset>/``. This script bridges the two. It downloads the
parquet (or reads a local copy), rewrites each row, and writes
``data/agentracer/code/<i>.json`` plus a ``_provenance.json`` recording what it did.

Three rewrites do the real work:

- **Agent names move into ``role``.** AgenTracer marks every turn ``role="assistant"``
  and puts the acting agent in ``name``. Every SOAP loader reads the agent from
  ``role`` -- ``stores.py`` takes ``history[t]["role"]`` as the agent for agent@k, and
  ``data.py`` serialises turns as ``[role] - Step i:``. So ``name`` becomes ``role``,
  and the redundant ``assistant`` is dropped.
- **The per-turn ``step`` counter is dropped.** It equals the list index in every row,
  and the corpus convention is turns of exactly ``{role, content}``.
- **Nothing is renumbered.** ``mistake_step`` is already 0-based and already agrees
  with the list index: ``history[mistake_step]["role"] == mistake_agent`` holds for all
  127 rows, which the script asserts.

Run it from the repo root:

    python scripts/build_agentracer.py [--parquet PATH] [--force]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/bingreeky/AgenTracer/main/data/"
    "tracertraj-code-test.parquet"
)
DATASET = "agentracer"
SUBSET = "code"

SYSTEM = (
    "**TracerTraj** trajectories come from a MetaGPT-style software company: a Team "
    "Leader routes the task to role-playing agents -- Product Manager, Architect, "
    "Engineer, Data Analyst -- who write a requirements document, a design, then the "
    "code.\n\n### Core mechanism\n\nThe Team Leader holds the plan and messages one "
    "agent at a time; each agent replies in role and hands control back. Every turn in "
    "the trajectory is one such agent message.\n\n### How the failure is labelled\n\n"
    "AgenTracer builds these traces by INJECTING a fault: a single agent turn is "
    "perturbed so the run fails, and that turn is the ground-truth decisive error. The "
    "error is therefore planted rather than naturally occurring, unlike Who&When or "
    "CORRECT-Error."
)


def fetch(parquet: Path | None, cache: Path) -> Path:
    if parquet is not None:
        return parquet
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, cache)
    return cache


def convert(row) -> dict:
    """One parquet row -> one corpus JSON object."""
    history = [
        {"role": str(turn["name"]), "content": str(turn["content"])}
        for turn in row["history"]
    ]
    mistake_step = int(str(row["mistake_step"]).strip())
    assert 0 <= mistake_step < len(history), (
        f"{row['question_ID']}: mistake_step {mistake_step} outside 0..{len(history) - 1}")
    assert history[mistake_step]["role"] == row["mistake_agent"], (
        f"{row['question_ID']}: step {mistake_step} is "
        f"{history[mistake_step]['role']!r}, labelled {row['mistake_agent']!r}")
    return {
        "question_ID":    str(row["question_ID"]),
        "question":       str(row["question"]),
        "history":        history,
        "mistake_agent":  str(row["mistake_agent"]),
        "mistake_step":   mistake_step,
        "level":          -1,
        "system":         SYSTEM,
        "subset":         SUBSET,
        "ground_truth":   str(row["ground_truth"]),
        "mistake_reason": str(row["mistake_reason"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=None,
                    help="local parquet; downloaded to data/agentracer/_source/ if omitted")
    ap.add_argument("--out", type=Path, default=Path("data") / DATASET)
    ap.add_argument("--force", action="store_true", help="overwrite an existing subset")
    args = ap.parse_args()

    src = fetch(args.parquet, args.out / "_source" / "tracertraj-code-test.parquet")
    df = pd.read_parquet(src)

    subset_dir = args.out / SUBSET
    if subset_dir.exists() and any(subset_dir.glob("*.json")) and not args.force:
        raise SystemExit(f"{subset_dir} already holds JSON files; pass --force to rebuild")
    subset_dir.mkdir(parents=True, exist_ok=True)

    lengths, agents = [], {}
    for i, (_, row) in enumerate(df.iterrows()):
        item = convert(row)
        (subset_dir / f"{i}.json").write_text(
            json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lengths.append(len(item["history"]))
        agents[item["mistake_agent"]] = agents.get(item["mistake_agent"], 0) + 1

    (args.out / "_provenance.json").write_text(json.dumps({
        "built_by": "scripts/build_agentracer.py",
        "source": {
            "paper": "AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?",
            "repo": "https://github.com/bingreeky/AgenTracer",
            "file": "data/tracertraj-code-test.parquet",
            "url": SOURCE_URL,
            "rows": int(len(df)),
        },
        "join": "parquet row order -> <i>.json, i = 0-based row index",
        "rewrites": [
            "turn 'name' becomes 'role' (SOAP reads the agent from 'role')",
            "turn 'role' (always 'assistant') dropped as redundant",
            "per-turn 'step' counter dropped; it equals the list index",
            "added 'level' (-1), 'subset', 'system'",
        ],
        "labels": {
            "mistake_step": "copied verbatim; already 0-based into history",
            "checked": "history[mistake_step]['role'] == mistake_agent for every row",
            "match_rate": 1.0,
        },
        "subsets": [{
            "subset": SUBSET,
            "n": len(lengths),
            "steps": int(sum(lengths)),
            "traj_len": {"min": min(lengths), "max": max(lengths),
                         "mean": round(sum(lengths) / len(lengths), 2)},
            "mistake_agent": dict(sorted(agents.items(), key=lambda kv: -kv[1])),
        }],
        "caveat": (
            "Errors are INJECTED, not naturally occurring, so detectability is not "
            "comparable to Who&When or CORRECT-Error. Only the test split is public: "
            "the repo ships no train parquet, so any SVD fit must come from a split of "
            "these 127 trajectories."
        ),
    }, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {len(lengths)} trajectories ({sum(lengths)} steps) to {subset_dir}")


if __name__ == "__main__":
    main()
