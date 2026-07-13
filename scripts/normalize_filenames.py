#!/usr/bin/env python3
"""Normalize a dataset's trajectory filenames to the integer convention.

The scoring pipeline assumes integer trajectory filenames (`1.json`, `34.json`)
— it casts `int(<stem>)` in several places. Datasets with string filenames
(e.g. CORRECT-Error's `gpt-4o-mini_arc_task100_1.json`) crash the svd /
reps-loading stage. This tool renames each subset's trajectories to `1..N` and
re-maps any already-extracted `.safetensors` outputs with the *same* mapping, so
the `tensor_file.stem == traj_file.stem` join stays intact.

`trajectory_id` is preserved inside each JSON (and in the emitted mapping CSV),
so the rename is fully reversible/traceable.

Dry-run by default; pass --apply to actually rename.

    python scripts/normalize_filenames.py \
        --data-root    data/correct-error \
        --outputs-root outputs-correct-error \
        [--start-index 1] [--apply]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_maps(data_root: Path, start_index: int) -> dict[str, dict[str, int]]:
    """Per-subset {trajectory_id (current stem) -> integer index}, from the data.

    Deterministic: sort by current stem so the id<->int correspondence is stable
    across runs. Assignment order does not affect split correctness (splits
    seed-shuffle the integer list regardless).
    """
    maps: dict[str, dict[str, int]] = {}
    for subset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        stems = sorted(f.stem for f in subset_dir.glob("*.json"))
        maps[subset_dir.name] = {
            stem: i for i, stem in enumerate(stems, start=start_index)
        }
    return maps


def plan_renames(
    maps: dict[str, dict[str, int]],
    data_root: Path,
    outputs_root: Path | None,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return (renames, orphans).

    renames : list of (src, dst) for JSON data files and .safetensors outputs.
    orphans : output .safetensors whose stem is not in its subset's map.
    """
    renames: list[tuple[Path, Path]] = []
    orphans: list[Path] = []

    # Data JSONs.
    for subset, m in maps.items():
        for stem, idx in m.items():
            src = data_root / subset / f"{stem}.json"
            renames.append((src, src.with_name(f"{idx}.json")))

    # Extracted outputs: infer subset from the parent dir name
    # (outputs-root/<stage>/<model>/<subset>/<file>.safetensors).
    if outputs_root is not None and outputs_root.exists():
        for st in sorted(outputs_root.rglob("*.safetensors")):
            subset = st.parent.name
            idx = maps.get(subset, {}).get(st.stem)
            if idx is None:
                orphans.append(st)
            else:
                renames.append((st, st.with_name(f"{idx}.safetensors")))

    return renames, orphans


def summarize(maps, renames, orphans) -> None:
    n_json = sum(1 for s, _ in renames if s.suffix == ".json")
    n_st = sum(1 for s, _ in renames if s.suffix == ".safetensors")
    print("Per-subset trajectory counts (data):")
    for subset, m in maps.items():
        lo = min(m.values()) if m else 0
        hi = max(m.values()) if m else 0
        print(f"  {subset:10s} n={len(m):4d}  -> [{lo}..{hi}]")
    print(f"\nPlanned renames: {n_json} JSON data files, {n_st} .safetensors outputs")
    print(f"Orphan outputs (stem not in any data map, SKIPPED): {len(orphans)}")
    for o in orphans[:20]:
        print(f"  ORPHAN: {o}")
    if len(orphans) > 20:
        print(f"  ... and {len(orphans) - 20} more")


def apply_renames(renames: list[tuple[Path, Path]]) -> None:
    # Guard: no target may pre-exist (string sources vs integer targets are
    # disjoint on a fresh run; this catches accidental re-runs / collisions).
    for _, dst in renames:
        if dst.exists():
            raise SystemExit(f"Refusing to overwrite existing target: {dst}")
    for src, dst in renames:
        src.rename(dst)


def write_map_csv(maps: dict[str, dict[str, int]], data_root: Path) -> Path:
    # At the dataset ROOT, never inside a subset dir (load_dataset globs
    # <subset>/*.json and would mis-read a stray file there as a trajectory).
    csv_path = data_root / "filename_map.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "index", "trajectory_id"])
        for subset, m in maps.items():
            for stem, idx in sorted(m.items(), key=lambda kv: kv[1]):
                w.writerow([subset, idx, stem])
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="dataset root containing <subset>/*.json")
    p.add_argument("--outputs-root", type=Path, default=None,
                   help="root of extracted .safetensors to re-map with the same mapping")
    p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--apply", action="store_true",
                   help="perform the rename (default: dry-run)")
    args = p.parse_args()

    maps = build_maps(args.data_root, args.start_index)
    renames, orphans = plan_renames(maps, args.data_root, args.outputs_root)
    summarize(maps, renames, orphans)

    if not args.apply:
        print("\n[dry-run] nothing changed. Re-run with --apply to rename.")
        return

    apply_renames(renames)
    csv_path = write_map_csv(maps, args.data_root)
    print(f"\n[applied] renamed {len(renames)} files; wrote mapping -> {csv_path}")


if __name__ == "__main__":
    main()
