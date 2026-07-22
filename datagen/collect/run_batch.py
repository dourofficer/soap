#!/usr/bin/env python3
"""Batch-run harness tasks: expand the collection matrix and drive one
subprocess per task.

One process per task is deliberate. The Magentic-One driver's LLM-logging
patch keeps state in module globals, so sharing a process across tasks would
bleed steps between them; and a per-task process makes timeouts a clean kill
rather than a corrupted run.

Resumability is filesystem-based: a run is done iff its `summary.json` exists,
and the drivers write that file last. Re-running the batch therefore skips
completed work with no bookkeeping.

    python datagen/collect/run_batch.py --config collect-qwen --dry-run
    python datagen/collect/run_batch.py --config collect-deepseek
    python datagen/collect/run_batch.py --config collect-qwen --pool gsm8k --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

_ledger_lock = threading.Lock()


@dataclass
class Job:
    harness: str
    backbone: str
    pool: str
    task_index: int
    rep: int
    task_id: str

    @property
    def run_name(self) -> str:
        return f"{self.task_id}-r{self.rep}"


# ── matrix expansion ──────────────────────────────────────────────────────────

def expand(cfg: dict, filters: dict) -> list[Job]:
    """Turn the matrix into a concrete job list.

    One config = one backbone, so the matrix only crosses harness x pool. A
    harness a backbone cannot drive is simply absent from that config's
    `harnesses` (see collect-deepseek.yaml, which omits captain).
    """
    backbone = cfg.get("backbone")
    if not backbone:
        raise SystemExit("config sets no `backbone` — use a per-model config "
                         "(collect-qwen / collect-deepseek)")

    jobs: list[Job] = []
    seen: set[tuple] = set()

    # `--pools` REPLACES the matrix pool lists, so any prepared dataset can be
    # run against a config without editing it. `--pool` only narrows to a pool
    # the matrix already names.
    override = filters.get("pools")

    for block in cfg["matrix"]:
        for harness in block["harnesses"]:
            if harness not in cfg["harnesses"]:
                raise SystemExit(
                    f"matrix names harness {harness!r}, which this config does "
                    f"not define (has: {sorted(cfg['harnesses'])})")
            if filters["harness"] and harness != filters["harness"]:
                continue
            for pool in (override or block["pools"]):
                if filters["pool"] and pool != filters["pool"]:
                    continue

                tasks = common.load_pool(pool)
                limit = filters["limit"] or block.get("limit")
                if limit:
                    tasks = tasks[:limit]
                for idx, task in enumerate(tasks):
                    for rep in range(block.get("reps", 1)):
                        key = (harness, pool, idx, rep)
                        if key in seen:
                            continue
                        seen.add(key)
                        jobs.append(Job(harness, backbone, pool, idx, rep, task["id"]))
    return jobs


# ── command construction ──────────────────────────────────────────────────────

def cpu_slice(index: int, per_task: int) -> list[int]:
    """A rotating block of `per_task` CPUs for job `index`.

    Slices wrap, so oversubscription (concurrency * per_task > cores) just
    shares cores — fine, since harness runs are I/O bound on the vLLM endpoint.
    """
    cpus = sorted(os.sched_getaffinity(0))
    start = (index * per_task) % len(cpus)
    return [cpus[(start + k) % len(cpus)] for k in range(min(per_task, len(cpus)))]


def build_command(job: Job, cfg: dict, serve_cfg: dict, run_dir: Path,
                  index: int = 0) -> tuple[list[str], dict, Path]:
    """Return (argv, env, cwd) for one task."""
    hspec = cfg["harnesses"][job.harness]
    root = common.REPO_ROOT / hspec["root"]
    ep = common.resolve_endpoint(job.backbone, serve_cfg)

    env = dict(os.environ)
    env["OPENAI_API_BASE"] = ep["base_url"]
    env["OPENAI_API_KEY"] = ep["api_key"]
    env["M1_MODEL"] = ep["model"]
    # HF tokenizers forks a thread pool per process and warns; useless here.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if hspec.get("pythonpath"):
        env["PYTHONPATH"] = hspec["pythonpath"]
    for k, v in (hspec.get("extra_env") or {}).items():
        env[str(k)] = str(v)

    argv = [str(root / hspec["python"]), hspec["driver"],
            "--pool", job.pool,
            "--task-index", str(job.task_index),
            "--output-dir", str(run_dir),
            "--max-round", str(cfg["max_round"])]

    # Pin each task to a few cores. Several libraries in the Captain stack size
    # their thread pools from the visible CPU count — chromadb's tokio runtime
    # alone opened one worker per core — which on a 240-core box meant ~682
    # threads per process and EAGAIN from chromadb once enough ran at once.
    # Capping affinity constrains every such pool with one knob: 682 -> ~13
    # threads at 4 CPUs.
    per_task = cfg.get("cpus_per_task")
    if per_task and shutil.which("taskset"):
        cpus = ",".join(str(c) for c in cpu_slice(index, int(per_task)))
        argv = ["taskset", "-c", cpus] + argv

    if job.harness == "magentic":
        if hspec.get("agents"):
            argv += ["--agents", hspec["agents"]]
        if hspec.get("no_function_calling"):
            argv.append("--no-function-calling")
    elif job.harness == "captain":
        argv += ["--model", ep["model"],
                 "--config-list", str(common.CONFIGS_DIR / "oai_config_list.json")]

    return argv, env, root


# ── execution ─────────────────────────────────────────────────────────────────

def run_job(job: Job, cfg: dict, serve_cfg: dict, dry_run: bool,
            index: int = 0) -> dict:
    out_root = common.REPO_ROOT / cfg["out_root"]
    run_dir = out_root / job.harness / job.backbone / job.pool / job.run_name
    result = {"harness": job.harness, "backbone": job.backbone, "pool": job.pool,
              "task_id": job.task_id, "rep": job.rep, "run_dir": str(run_dir)}

    # Done-marker: drivers write summary.json last, so its presence means the
    # run completed (successfully or with a recorded in-run error).
    if (run_dir / "summary.json").exists():
        return {**result, "status": "skipped"}

    argv, env, cwd = build_command(job, cfg, serve_cfg, run_dir, index)
    if dry_run:
        print(f"  cd {cwd} && {' '.join(argv)}")
        return {**result, "status": "dry-run"}

    timeout = cfg["timeout_s"].get(job.pool, cfg["timeout_s"]["default"])
    attempts = cfg.get("retries", 0) + 1
    started = time.time()

    for attempt in range(1, attempts + 1):
        run_dir.mkdir(parents=True, exist_ok=True)
        log = run_dir / "driver.log"
        try:
            with log.open("w") as fh:
                proc = subprocess.run(argv, cwd=cwd, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT, timeout=timeout)
            if (run_dir / "summary.json").exists():
                return {**result, "status": "ok", "returncode": proc.returncode,
                        "elapsed_s": round(time.time() - started, 1),
                        "attempts": attempt}
            status = f"no-summary(rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            (run_dir / "timeout.marker").write_text(
                f"killed after {timeout}s (attempt {attempt})\n")
            status = "timeout"
        except Exception as e:  # noqa: BLE001 — record and move on
            status = f"error:{type(e).__name__}"

        if attempt == attempts:
            return {**result, "status": status,
                    "elapsed_s": round(time.time() - started, 1),
                    "attempts": attempt}
    return {**result, "status": "unreachable"}


def append_ledger(path: Path, row: dict) -> None:
    with _ledger_lock:
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="datagen/configs/<name>.yaml — one per backbone "
                         "(collect-qwen, collect-deepseek)")
    ap.add_argument("--set", action="append", dest="overrides", default=[],
                    metavar="KEY=VALUE", help="dot-path config override")
    ap.add_argument("--harness")
    ap.add_argument("--pool", help="narrow to one pool the matrix already names")
    ap.add_argument("--pools",
                    help="comma list that REPLACES the matrix pools, so a config "
                         "can be run against any prepared dataset without editing it "
                         "(e.g. --pools gsm8k,math500)")
    ap.add_argument("--limit", type=int, help="cap tasks per pool (overrides matrix)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = common.load_cfg(args.config, args.overrides)
    serve_cfg = common.load_cfg("serve")
    filters = {"harness": args.harness, "pool": args.pool, "limit": args.limit,
               "pools": [p.strip() for p in args.pools.split(",") if p.strip()]
               if args.pools else None}

    jobs = expand(cfg, filters)
    if not jobs:
        print("no jobs after filtering. Note --pool only narrows to a pool the "
              "matrix already names; use --pools to run a different dataset.")
        return 1

    out_root = common.REPO_ROOT / cfg["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    ledger = out_root / "manifest.jsonl"

    cells = sorted({(j.harness, j.backbone, j.pool) for j in jobs})
    print(f"{len(jobs)} jobs across {len(cells)} cells "
          f"(concurrency={cfg['concurrency']})")
    for h, b, p in cells:
        n = sum(1 for j in jobs if (j.harness, j.backbone, j.pool) == (h, b, p))
        print(f"  {h:9s} {b:12s} {p:15s} {n:5d}")

    if args.dry_run:
        for i, job in enumerate(jobs[:10]):
            run_job(job, cfg, serve_cfg, dry_run=True, index=i)
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        return 0

    counts: dict[str, int] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        futures = {pool.submit(run_job, j, cfg, serve_cfg, False, i): j
                   for i, j in enumerate(jobs)}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            row["finished_at"] = datetime.now().isoformat(timespec="seconds")
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row["status"] != "skipped":
                append_ledger(ledger, row)
            if i % 10 == 0 or i == len(jobs):
                rate = i / max(time.time() - started, 1e-9) * 3600
                print(f"[{i}/{len(jobs)}] " +
                      " ".join(f"{k}={v}" for k, v in sorted(counts.items())) +
                      f"  ({rate:.0f} runs/h)")

    print("\ndone: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"ledger: {ledger}")
    print("next:   python datagen/judge/rejudge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
