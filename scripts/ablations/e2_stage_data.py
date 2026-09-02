"""E2 data staging — materialize the filtered synthetic reference corpora.

Each corpus in ../datagen/data/synthetic/ is filtered to the trajectories whose
question appears in the TARGET Who&When subset, so the fit set's question pool is
identical to WW's (experiments/todo.md, E2). Files are hardlinked under
data/synthetic/<target>-<generator>/ keeping their original numbering:

    ag-gpt4o   <- captain-gpt4o   ∩ WW-AG questions   (126)
    ag-qwen9b  <- captain-qwen9b  ∩ WW-AG questions   (124)
    hc-gpt4o   <- magentic-gpt4o  ∩ WW-HC questions   (55)
    hc-qwen9b  <- magentic-qwen9b ∩ WW-HC questions   (55)

Idempotent: an existing target directory with the expected file set is left alone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATAGEN = REPO.parent / "datagen" / "data" / "synthetic"
OUT = REPO / "data" / "synthetic"

# (target subset dir under data/ww, source corpus dir, staged name, expected count)
JOBS = [
    ("algorithm-generated", "captain-gpt4o", "ag-gpt4o", 126),
    ("algorithm-generated", "captain-qwen9b", "ag-qwen9b", 124),
    ("hand-crafted", "magentic-gpt4o", "hc-gpt4o", 55),
    ("hand-crafted", "magentic-qwen9b", "hc-qwen9b", 55),
]


def questions(directory: Path) -> set[str]:
    qs = set()
    for f in directory.glob("*.json"):
        qs.add(json.loads(f.read_text())["question"].strip())
    return qs


def main() -> int:
    for target, source, staged, expected in JOBS:
        ww_qs = questions(REPO / "data" / "ww" / target)
        src_dir = DATAGEN / source
        keep = [f for f in sorted(src_dir.glob("*.json"), key=lambda p: int(p.stem))
                if json.loads(f.read_text())["question"].strip() in ww_qs]
        assert len(keep) == expected, f"{staged}: kept {len(keep)}, expected {expected}"
        out_dir = OUT / staged
        have = {f.name for f in out_dir.glob("*.json")} if out_dir.exists() else set()
        want = {f.name for f in keep}
        if have == want:
            print(f"{staged}: {len(keep)} files already staged")
            continue
        assert not have, f"{staged}: existing files differ from the expected set"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in keep:
            os.link(f, out_dir / f.name)
        print(f"{staged}: staged {len(keep)} of {len(list(src_dir.glob('*.json')))} "
              f"from {source} (questions ∩ {target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
