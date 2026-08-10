"""Import prompting-baseline predictions from ../attrib-prompting into this repo.

attrib-prompting stores one JSON per trajectory at
``<root>/<ds>/<subset>/<judge>/<method>/<id>.json`` with ``outputs/`` = with-GT and
``outputs-nogt/`` = without-GT. This converter writes the JSONL layout the reports
stage scores (``src/reports/baselines.py``):

    outputs/<ds>/baselines/prompting/<judge>/<subset>/predictions_method-<method>.jsonl
    outputs-gt/<ds>/...                                  (from attrib-prompting/outputs)

Only the closed-source judges (gpt-4o, gpt-5) are imported — the local-model dirs
over there are partly round-trips FROM this repo (their _run.json records
``imported_from: .../attribscope/...``) and the backbone JSONLs here stay untouched.
Kept fields: id, filename, question_id, predicted_agent, predicted_step, gold_agent,
gold_step (raw/call logs dropped — the scorer ignores them). ``_run.json`` and other
non-digit stems are skipped. Existing JSONLs are never overwritten without --force.
Line counts are asserted against the corpus trajectory counts.

    # from repo root
    python scripts/import_prompting.py [--src ../attrib-prompting] [--force]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEEP = ["id", "filename", "question_id", "predicted_agent", "predicted_step",
        "gold_agent", "gold_step"]
JUDGES = ["gpt-4o", "gpt-5"]
METHODS = ["all_at_once", "step_by_step", "binary_search"]
DATASETS = ["ww", "traceelephant", "correct-error"]
# (attrib-prompting source root, attribscope destination base)
SETTINGS = [("outputs-nogt", "outputs"), ("outputs", "outputs-gt")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(REPO.parent / "attrib-prompting"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    src_repo = Path(args.src)

    written = skipped = 0
    for src_root, dst_base in SETTINGS:
        for ds in DATASETS:
            for subset_dir in sorted((src_repo / src_root / ds).iterdir()):
                if not subset_dir.is_dir() or subset_dir.name == "reports":
                    continue
                subset = subset_dir.name
                n_traj = len(list((REPO / "data" / ds / subset).glob("*.json")))
                for judge in JUDGES:
                    for method in METHODS:
                        d = subset_dir / judge / method
                        if not d.is_dir():
                            print(f"[miss] {src_root}/{ds}/{subset}/{judge}/{method}")
                            continue
                        files = sorted((p for p in d.glob("*.json") if p.stem.isdigit()),
                                       key=lambda p: int(p.stem))
                        assert len(files) == n_traj, \
                            f"{d}: {len(files)} files != {n_traj} trajectories"
                        out = (REPO / dst_base / ds / "baselines" / "prompting" / judge
                               / subset / f"predictions_method-{method}.jsonl")
                        if out.exists() and not args.force:
                            skipped += 1
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with open(out, "w", encoding="utf-8") as fh:
                            for p in files:
                                row = json.loads(p.read_text(encoding="utf-8"))
                                fh.write(json.dumps({k: row.get(k) for k in KEEP},
                                                    ensure_ascii=False) + "\n")
                        written += 1
        print(f"[{src_root} -> {dst_base}] done")
    print(f"wrote {written} JSONLs, skipped {skipped} existing")


if __name__ == "__main__":
    main()
