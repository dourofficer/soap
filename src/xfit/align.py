"""Question-aligned synthetic fit-set selection (paper setting).

For one (target subset, seed), pick synthetic trajectories that carry THE SAME
QUESTIONS as that seed's real train split, so the only thing that changes between the
Real row and a Synthetic row is who generated the reference trajectories. Each synthetic
cell holds exactly one trajectory per question, so alignment is 1:1; the handful of
train questions whose datagen run failed are filled to the exact train-split count with
a seeded same-pool draw that EXCLUDES every question appearing anywhere in the target
subset (train, val or test) — the fill can therefore never leak evaluation questions.

Join keys, per dataset:
* ``ww`` — the corpus ``question_ID`` IS the pool ``raw_id`` (GAIA uuid / AssistantBench
  sha256); joined through ``datagen/pools/data/<pool>.jsonl`` to ``<pool>/<pool-id>``.
* ``traceelephant`` — ``question_ID`` is a free slug; joined by whitespace-normalized
  question text against the pool questions (measured 100%).
* correct-* — NO join registered (36% synthetic coverage + corrupted questions);
  the paper setting refuses these targets until the corpus/datagen gap is closed.

Everything is deterministic: candidate lists are sorted before sampling and the RNG is
a local ``random.Random`` keyed on (dataset, subset, source, seed) — never the global
RNG, which ``split_files`` reseeds.

    from src.xfit.align import fit_files
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from random import Random

from .common import synth_data_dir

POOLS_DIR = Path("datagen/pools/data")


def _norm(q) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


@lru_cache(maxsize=None)
def _pool_rows(pool: str) -> tuple[dict, ...]:
    with (POOLS_DIR / f"{pool}.jsonl").open(encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


@lru_cache(maxsize=None)
def _raw_id_to_key(pools: tuple[str, ...]) -> dict[str, str]:
    return {str(r["raw_id"]): f"{pool}/{r['id']}"
            for pool in pools for r in _pool_rows(pool)}


@lru_cache(maxsize=None)
def _text_to_key(pools: tuple[str, ...]) -> dict[str, str]:
    return {_norm(r["question"]): f"{pool}/{r['id']}"
            for pool in pools for r in _pool_rows(pool)}


# ── per-dataset join: trajectory JSON -> canonical pool key ──────────────────
def _key_ww(doc: dict, pools: tuple[str, ...]) -> str | None:
    return _raw_id_to_key(pools).get(str(doc.get("question_ID")))


def _key_text(doc: dict, pools: tuple[str, ...]) -> str | None:
    return _text_to_key(pools).get(_norm(doc.get("question")))


JOINS = {"ww": _key_ww, "traceelephant": _key_text}


@lru_cache(maxsize=None)
def subset_keys(dataset: str, subset: str, pools: tuple[str, ...]) -> dict[str, str | None]:
    """stem -> pool key (or None) for EVERY trajectory of a real subset."""
    join = JOINS.get(dataset)
    if join is None:
        raise SystemExit(
            f"paper setting: no question-alignment join for dataset {dataset!r} "
            f"(correct-* deferred — see src/xfit/align.py docstring)")
    data_dir = Path("data") / dataset / subset
    out = {}
    for fp in sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem)):
        out[fp.stem] = join(json.loads(fp.read_text()), pools)
    return out


@lru_cache(maxsize=None)
def synth_map(source: str, pools: tuple[str, ...]) -> dict[str, str]:
    """pool key -> synthetic filename, restricted to the fit pools (1 traj/question)."""
    out = {}
    with (synth_data_dir(source) / "filename_map.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["pool"] in pools:
                assert row["question_ID"] not in out, \
                    f"duplicate synthetic question {row['question_ID']} in {source}"
                out[row["question_ID"]] = row["file"]
    return out


# ── the selection ────────────────────────────────────────────────────────────
def fit_files(dataset: str, subset: str, seed: int, source: str,
              train_stems: list[str], pools: list[str]) -> tuple[list[str], dict]:
    """Synthetic fit set for one (target, seed): question-matched + seeded fill.

    Returns (filenames sorted numerically, report). ``len(files) == len(train_stems)``
    always — the fit-set SIZE is part of the "only the reference source changed" claim.
    """
    pools_t = tuple(pools)
    keys = subset_keys(dataset, subset, pools_t)
    smap = synth_map(source, pools_t)

    matched: dict[str, str] = {}          # stem -> synthetic file
    unmatched: list[tuple[str, str | None]] = []
    taken: set[str] = set()
    for stem in sorted(train_stems, key=int):
        key = keys.get(stem)
        f = smap.get(key) if key else None
        if f is not None and f not in taken:
            matched[stem] = f
            taken.add(f)
        else:
            unmatched.append((stem, key))

    fill: list[str] = []
    if unmatched:
        # candidates share the unmatched questions' pools and are questions that appear
        # NOWHERE in this subset — filling can never touch a val/test question.
        want_pools = sorted({k.split("/")[0] for _, k in unmatched if k}) or sorted(pools_t)
        forbidden = {k for k in keys.values() if k}
        cands = sorted(f for k, f in smap.items()
                       if f not in taken and k not in forbidden
                       and k.split("/")[0] in want_pools)
        if len(cands) < len(unmatched):
            raise SystemExit(f"not enough fill candidates for {dataset}/{subset} "
                             f"seed={seed} source={source}")
        rng = Random(f"xfit-align:{dataset}:{subset}:{source}:{seed}")
        fill = sorted(rng.sample(cands, len(unmatched)))

    files = sorted([*matched.values(), *fill], key=lambda f: int(Path(f).stem))
    report = {
        "dataset": dataset, "subset": subset, "seed": int(seed), "source": source,
        "pools": sorted(pools_t), "n_train": len(train_stems),
        "n_matched": len(matched), "n_filled": len(fill),
        "matched": dict(sorted(matched.items(), key=lambda kv: int(kv[0]))),
        "filled_files": fill,
        "unmatched": [[stem, key] for stem, key in unmatched],
        "files": files,
    }
    assert len(files) == len(train_stems), "fit-set size must equal the train split"
    return files, report
