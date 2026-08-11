"""Sweep many seed triples through `main/` WITHOUT modifying `main/`.

`main/` takes exactly one 3-seed triple per subset, declared in its config. This drives it
from outside, once per triple, by injecting the seeds and a private output tree through
`--set`:

    python -m main sweep  --config configs-main/<ds>.yaml \\
        --set gt=<bool> --set results_base=results-sweep/<gt>/<triple> \\
        --set seeds.<subset>=[a,b,c]   (repeated for EVERY subset) \\
        --model M --subset S

THE ONE SUBTLETY. `--set results_base=X` redirects EVERYTHING under `REPO/X/<ds>/` —
including `activations/` and `attention/` — and it also overrides the gt/nogt tree choice.
So each triple tree gets RELATIVE symlinks pointing at the real extractions for its GT
setting. Relative, so `results-sweep/` survives a repo move.

Seeds only affect the train/val/test split, never extraction, so nothing here needs a GPU
forward pass — only the extractions already on disk.

Every triple run sets ALL subsets of a dataset to the same triple, so every cell is swept
at every triple and any subset can later pick its own best one.

    python scripts/main/sweep_triples.py --dry-run          # plan only
    python scripts/main/sweep_triples.py --triples s03-04-05,s17-18-19   # early gate
    WORKERS=8 python scripts/main/sweep_triples.py          # everything
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python")
SWEEP_ROOT = REPO / "results-sweep"
QUEUE = SWEEP_ROOT / "queue"
LOGS = SWEEP_ROOT / "logs"

DATASETS = ["ww", "traceelephant", "correct-error"]
# Measured on the production run; used only to order the queue longest-first so the
# 950 s correct-error units start early and the tail is short.
UNIT_COST = {"ww": 200, "traceelephant": 190, "correct-error": 950}

# Set once by main(); run_unit reads it to size BLAS thread pinning.
N_WORKERS = [8]

# The child `python -m main ...` currently running, so a signalled worker can take it
# down with it. Without this, killing the driver ORPHANS the child: it keeps running,
# keeps a GPU busy, and keeps writing into a tree the next driver will also touch.
_CHILD: list = [None]


def _terminate_child(signum, _frame):
    proc = _CHILD[0]
    if proc is not None and proc.poll() is None:
        proc.kill()
    sys.exit(128 + signum)


def _run_child(cmd, cwd, env, log) -> int:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    _CHILD[0] = proc
    try:
        return proc.wait()
    finally:
        _CHILD[0] = None


def triples(lo: int = 1, hi: int = 50, width: int = 3) -> list[list[int]]:
    """Consecutive windows: (1,2,3), (2,3,4), ... (48,49,50)."""
    return [list(range(s, s + width)) for s in range(lo, hi - width + 2)]


def tag(triple: list[int]) -> str:
    return "s" + "-".join(f"{x:02d}" for x in triple)


@dataclass(frozen=True)
class Unit:
    gt: bool
    triple: tuple[int, ...]
    dataset: str

    @property
    def gt_dir(self) -> str:
        return "gt" if self.gt else "nogt"

    @property
    def results_base(self) -> str:
        return f"results-sweep/{self.gt_dir}/{tag(list(self.triple))}"

    @property
    def root(self) -> Path:
        return REPO / self.results_base / self.dataset

    @property
    def source(self) -> Path:
        """The real extraction tree this unit symlinks to."""
        return REPO / ("results-gt" if self.gt else "results-nogt") / self.dataset

    def key(self) -> str:
        return f"{self.gt_dir}/{tag(list(self.triple))}/{self.dataset}"


def load_cfg(dataset: str) -> dict:
    return yaml.safe_load((REPO / "configs-main" / f"{dataset}.yaml").read_text())


def cells(dataset: str, cfg: dict) -> list[tuple[str, str]]:
    return [(m, s) for m in cfg["models"] for s in cfg["subsets"]]


def seed_overrides(cfg: dict, triple: tuple[int, ...]) -> list[str]:
    """One --set per subset: main/ reads seeds per subset, and check_stamp hashes the
    whole mapping, so setting them all keeps the stamp identical across a unit's cells."""
    return [f"seeds.{s}={list(triple)}" for s in cfg["subsets"]]


# ── preparation ─────────────────────────────────────────────────────────────
def prepare(unit: Unit) -> None:
    """Create the unit's tree and the two relative extraction symlinks. Idempotent.

    The realpath assertion is the guard against the quietest possible bug: linking the
    GT extractions into a no-GT unit produces numbers that are wrong and nothing errors.
    """
    unit.root.mkdir(parents=True, exist_ok=True)
    for kind in ("activations", "attention"):
        link = unit.root / kind
        target = unit.source / kind
        if not target.is_dir():
            raise SystemExit(f"missing extraction tree: {target}")
        rel = os.path.relpath(target, link.parent)
        if link.is_symlink():
            if os.readlink(link) != rel:
                link.unlink()
        elif link.exists():
            raise SystemExit(f"{link} exists and is not a symlink; refusing to touch it")
        if not link.is_symlink():
            link.symlink_to(rel, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise SystemExit(f"{link} resolves to {link.resolve()}, expected {target}")


def valid_sweep(path: Path) -> bool:
    """A sweep.tsv is trustworthy only if its row count is one main/ could have written.

    `main sweep` skips a cell whose file merely EXISTS, so a `to_csv` killed mid-write
    would be treated as done forever and the collector would silently read short. Every
    complete cell has exactly

        3 seeds x (210 bands x P positions + 4 ranges x 6 ws x 3 strategies x 7 gammas)
        = 3 x (210*P + 504)

    rows, so a valid file pins an integer P >= 1. Verified against the production run:
    8442 rows -> P=11 (qwen3.5-9b), 23562 -> P=35 (deepseek-8b).
    """
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            if f.readline().count(b"\t") < 10:          # header sanity
                return False
            n = sum(1 for _ in f)
            f.seek(max(0, path.stat().st_size - 1))
            if f.read(1) != b"\n":                      # truncated mid-row
                return False
    except OSError:
        return False
    if n % 3:
        return False
    per_seed = n // 3 - 504
    return per_seed > 0 and per_seed % 210 == 0


def valid_selection(path: Path, n_cells: int) -> bool:
    """`main select` prints [skip] for a missing cell and writes a SHORT file anyway —
    the failure mode that would silently poison the aggregation. A complete selection has
    one row per (model, subset) per row-label {svd, backprop, succ-strong, succ-near}."""
    if not path.exists():
        return False
    with open(path) as f:
        return sum(1 for _ in f) - 1 == 4 * n_cells


def is_done(unit: Unit, cfg: dict) -> bool:
    """Done iff select landed complete AND every cell's sweep.tsv is complete."""
    cs = cells(unit.dataset, cfg)
    if not valid_selection(unit.root / "select" / "selection.tsv", len(cs)):
        return False
    return all(valid_sweep(unit.root / "sweep" / m / s / "sweep.tsv") for m, s in cs)


# ── the shared work queue ───────────────────────────────────────────────────
def build_queue(units: list[Unit]) -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = [{"gt": u.gt, "triple": list(u.triple), "dataset": u.dataset} for u in units]
    (QUEUE / "jobs.json").write_text(json.dumps(rows, indent=1))
    (QUEUE / "cursor").write_text("0")


def pop() -> int | None:
    """Atomically take the next index. flock so 8 workers cannot claim the same unit."""
    path = QUEUE / "cursor"
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            i = int(f.read().strip() or 0)
            f.seek(0)
            f.write(str(i + 1))
            f.truncate()
            return i
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def record_failure(unit: Unit, what: str, code: int) -> None:
    with open(QUEUE / "failures.tsv", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{unit.key()}\t{what}\t{code}\n")
        fcntl.flock(f, fcntl.LOCK_UN)


# ── running one unit ────────────────────────────────────────────────────────
def run_unit(unit: Unit, gpu: int, log) -> bool:
    cfg = load_cfg(unit.dataset)
    cfg_path = str(REPO / "configs-main" / f"{unit.dataset}.yaml")
    base = ["--set", f"gt={str(unit.gt).lower()}",
            "--set", f"results_base={unit.results_base}"]
    for ov in seed_overrides(cfg, unit.triple):
        base += ["--set", ov]
    # Pin BLAS/OMP threads: torch would otherwise take all 240 cores PER worker, and 8
    # workers fighting over them is slower than 8 workers with a slice each.
    nthreads = str(max(1, min(16, (os.cpu_count() or 8) // max(1, N_WORKERS[0]))))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "OMP_NUM_THREADS": nthreads, "MKL_NUM_THREADS": nthreads,
           "OPENBLAS_NUM_THREADS": nthreads}

    prepare(unit)
    ok = True
    for model, subset in cells(unit.dataset, cfg):
        out = unit.root / "sweep" / model / subset / "sweep.tsv"
        if valid_sweep(out):
            continue
        cmd = [PY, "-m", "main", "sweep", "--config", cfg_path, *base,
               "--model", model, "--subset", subset]
        if out.exists():        # present but short -> main/ would skip it; force a redo
            cmd.append("--force")
            log.write(f"[{unit.key()}] {model}/{subset}: incomplete sweep.tsv, forcing\n".encode())
        rc = _run_child(cmd, REPO, env, log)
        if rc != 0 or not valid_sweep(out):
            record_failure(unit, f"sweep {model}/{subset}", rc)
            ok = False

    if not ok:
        # Never run select on an incomplete unit: main/select silently omits missing
        # cells, which would write a partial selection.tsv and poison the aggregation.
        log.write(f"[unit {unit.key()}] INCOMPLETE — skipping select\n".encode())
        return False

    cmd = [PY, "-m", "main", "select", "--config", cfg_path, *base]   # NO narrowing
    rc = _run_child(cmd, REPO, env, log)
    sel = unit.root / "select" / "selection.tsv"
    if rc != 0 or not valid_selection(sel, len(cells(unit.dataset, cfg))):
        record_failure(unit, "select", rc)
        return False
    return True


def worker(wid: int, gpus: list[int], units: list[Unit]) -> int:
    gpu = gpus[wid % len(gpus)]
    done = failed = 0
    with open(LOGS / f"worker-{wid}.log", "ab", buffering=0) as log:
        while True:
            i = pop()
            if i is None or i >= len(units):
                break
            unit = units[i]
            t0 = time.time()
            log.write(f"\n=== [{i}] {unit.key()} on gpu{gpu} ===\n".encode())
            if run_unit(unit, gpu, log):
                done += 1
            else:
                failed += 1
            log.write(f"=== [{i}] {unit.key()} took {time.time()-t0:.0f}s ===\n".encode())
    print(f"[worker {wid}] gpu{gpu}: {done} done, {failed} failed", flush=True)
    return failed


# ── entry point ─────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triples", default="all",
                   help="'all' or a comma list of tags, e.g. s03-04-05,s17-18-19")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--gt", default="both", choices=["both", "true", "false"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 8)))
    p.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3,4,5,6,7"))
    p.add_argument("--seed-lo", type=int, default=1)
    p.add_argument("--seed-hi", type=int, default=50)
    p.add_argument("--plan", default=None,
                   help="per-dataset seed ranges, e.g. "
                        "'ww:1-50,traceelephant:1-50,correct-error:1-20'. Overrides "
                        "--seed-lo/--seed-hi for the datasets it names.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    ds_list = [d for d in args.datasets.split(",") if d]
    gt_list = {"both": [False, True], "true": [True], "false": [False]}[args.gt]

    # Per-dataset triple ranges. correct-error costs 5x per unit, and 18 windows over
    # seeds 1-20 already matches the original protocol's span, so it does not need the
    # full 48 that ww/traceelephant get.
    ranges = {d: (args.seed_lo, args.seed_hi) for d in ds_list}
    for part in (args.plan or "").split(","):
        if not part.strip():
            continue
        ds, _, rng = part.partition(":")
        lo, _, hi = rng.partition("-")
        if ds not in ranges:
            raise SystemExit(f"--plan names unknown dataset {ds!r}")
        ranges[ds] = (int(lo), int(hi))

    want = None if args.triples == "all" else set(args.triples.split(","))
    per_ds = {}
    for ds in ds_list:
        ts = triples(*ranges[ds])
        if want is not None:
            ts = [t for t in ts if tag(t) in want]
        per_ds[ds] = ts
    if want is not None and not any(per_ds.values()):
        raise SystemExit(f"no triples matched {sorted(want)}")
    all_triples = sorted({tag(t) for ts in per_ds.values() for t in ts})

    units = [Unit(gt, tuple(t), ds)
             for gt in gt_list for ds in ds_list for t in per_ds[ds]]
    # longest-first: correct-error units start early so the tail is short
    units.sort(key=lambda u: -UNIT_COST[u.dataset])

    cfgs = {d: load_cfg(d) for d in ds_list}
    todo = [u for u in units if not is_done(u, cfgs[u.dataset])]
    n_cells = sum(len(cells(u.dataset, cfgs[u.dataset])) for u in todo)
    est = sum(UNIT_COST[u.dataset] for u in todo)

    print("plan: " + "  ".join(f"{d}={len(per_ds[d])}tri({ranges[d][0]}-{ranges[d][1]})"
                                for d in ds_list))
    print(f"distinct triples={len(all_triples)} gt={gt_list}")
    print(f"units: {len(todo)} to run / {len(units)} total   cells: {n_cells}")
    print(f"est ~{est/3600:.1f} GPU-h -> ~{est/3600/max(args.workers,1):.1f} h "
          f"on {args.workers} workers")
    if args.dry_run:
        for u in todo[:5]:
            print(f"  would run {u.key()}  ({len(cells(u.dataset, cfgs[u.dataset]))} cells)")
        print(f"  ... and {max(0, len(todo)-5)} more")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    N_WORKERS[0] = args.workers

    # Child mode: one worker process.
    if args.worker_id is not None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _terminate_child)
        return 1 if worker(args.worker_id, gpus, todo) else 0

    # Parent: build the queue once, then fan out children that share it.
    build_queue(todo)
    (QUEUE / "failures.tsv").write_text("")
    argv = [PY, __file__, "--triples", args.triples, "--datasets", args.datasets,
            "--gt", args.gt, "--gpus", args.gpus, "--workers", str(args.workers),
            "--seed-lo", str(args.seed_lo), "--seed-hi", str(args.seed_hi)]
    if args.plan:
        argv += ["--plan", args.plan]
    procs = [subprocess.Popen(argv + ["--worker-id", str(w)], cwd=REPO)
             for w in range(args.workers)]

    def _stop(signum, _frame):
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()          # workers forward it to their children
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)
    rc = max((pr.wait() for pr in procs), default=0)

    fails = [l for l in (QUEUE / "failures.tsv").read_text().splitlines() if l.strip()]
    remaining = [u for u in units if not is_done(u, cfgs[u.dataset])]
    print(f"\nunits complete: {len(units)-len(remaining)}/{len(units)}")
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for l in fails[:20]:
            print("  -", l)
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
