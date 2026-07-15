#!/usr/bin/env python
"""Merge all CORRECT-Error subsets into the single-subset dataset `correct-full`.

Who&When and TraceElephant name subsets by the *agentic system* that produced the
trajectories; every CORRECT-Error trajectory comes from Magentic-One, so the
distribution-based subsets (arc, gaia, ...) collapse into one subset (`magentic`).

Creates
    data/correct-full/<subset>/                      from data/correct-error/*/
    outputs-correct-full/activations/<model>/<subset>/
    outputs-correct-full/attention/<model>/<subset>/
    outputs-correct-full/prompting/<model>/<subset>/
    outputs-correct-full/chief/<model>/<subset>/     from outputs-correct-error/...

Trajectory filenames are per-subset sequential (1.json..N.json) and collide across
subsets, while the SVD/CRR pipeline splits and joins everything by the integer file
stem. The merge therefore renumbers globally (1..2226, alphabetical subset order,
numeric order within a subset) and applies the same map to data JSONs, both
safetensors stages (metadata header rewritten, tensor blob copied verbatim), and
the id/filename fields of the prompting/chief prediction rows.

Derived stages (weighted-projections, *-splits) are NOT migrated: the merged
subset gets a fresh global train/val/test split, so they must be recomputed.
"""

import argparse
import csv
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import NamedTuple

SUBSETS = ["arc", "gaia", "hotpot", "math500", "mmlu_pro", "musique", "wikimqa"]
PROMPT_METHODS = ["all_at_once", "step_by_step", "binary_search"]
EXPECTED_TOTAL = 2226

# Config keys allowed to differ across subsets; everything else must be identical.
VOLATILE_TENSOR_CFG = {"subset"}
VOLATILE_JSONL_CFG = {"subset", "n_trajectories"}


class Entry(NamedTuple):
    orig_subset: str
    orig_stem: int
    new_stem: int
    trajectory_id: str


# --------------------------------------------------------------------------- map


def build_renumber_map(src_data: Path) -> list[Entry]:
    entries: list[Entry] = []
    new_stem = 1
    for subset in SUBSETS:
        subset_dir = src_data / subset
        stems = sorted(int(p.stem) for p in subset_dir.glob("*.json"))
        assert stems == list(range(1, len(stems) + 1)), \
            f"{subset_dir}: stems are not contiguous 1..N"
        for stem in stems:
            traj = json.loads((subset_dir / f"{stem}.json").read_text())
            entries.append(Entry(subset, stem, new_stem, traj["trajectory_id"]))
            new_stem += 1
    assert len(entries) == EXPECTED_TOTAL, \
        f"expected {EXPECTED_TOTAL} trajectories, found {len(entries)}"

    map_csv = src_data / "filename_map.csv"
    if map_csv.exists():
        with open(map_csv) as f:
            ref = {(r["subset"], int(r["index"])): r["trajectory_id"]
                   for r in csv.DictReader(f)}
        for e in entries:
            assert ref.get((e.orig_subset, e.orig_stem)) == e.trajectory_id, \
                f"filename_map.csv mismatch for {e.orig_subset}/{e.orig_stem}"
    else:
        print(f"warning: {map_csv} not found, skipping cross-check")
    return entries


# ---------------------------------------------------------------------- preflight


def _stem_set(directory: Path, suffix: str) -> set[int]:
    return {int(p.stem) for p in directory.glob(f"*{suffix}")}


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            assert row["id"] not in rows, f"{path}: duplicate id {row['id']}"
            rows[row["id"]] = row
    return rows


def _assert_configs_consistent(paths: list[Path], volatile: set[str]) -> dict:
    configs = [json.loads(p.read_text()) for p in paths]
    residuals = [{k: v for k, v in c.items() if k not in volatile} for c in configs]
    for p, r in zip(paths[1:], residuals[1:]):
        assert r == residuals[0], f"config mismatch: {paths[0]} vs {p}"
    return configs[0]


def preflight(src_outputs: Path, models: list[str], entries: list[Entry]) -> None:
    by_subset: dict[str, list[Entry]] = {s: [] for s in SUBSETS}
    for e in entries:
        by_subset[e.orig_subset].append(e)

    for model in models:
        for stage in ("activations", "attention"):
            cfg_paths = []
            for subset in SUBSETS:
                d = src_outputs / stage / model / subset
                n = len(by_subset[subset])
                assert _stem_set(d, ".safetensors") == set(range(1, n + 1)), \
                    f"{d}: safetensors stems != 1..{n}"
                assert (d / "config.json").exists(), f"{d}/config.json missing"
                cfg_paths.append(d / "config.json")
            _assert_configs_consistent(cfg_paths, VOLATILE_TENSOR_CFG)

        for stage, methods in (("prompting", PROMPT_METHODS), ("chief", ["chief"])):
            for method in methods:
                cfg_paths = []
                for subset in SUBSETS:
                    d = src_outputs / stage / model / subset
                    rows = _load_jsonl(d / f"predictions_method-{method}.jsonl")
                    subset_entries = by_subset[subset]
                    assert set(rows) == {str(e.orig_stem) for e in subset_entries}, \
                        f"{d} [{method}]: row ids != 1..{len(subset_entries)}"
                    for e in subset_entries:
                        qid = rows[str(e.orig_stem)]["question_id"]
                        assert qid == e.trajectory_id, \
                            f"{d} [{method}] id {e.orig_stem}: question_id {qid!r} " \
                            f"!= trajectory_id {e.trajectory_id!r}"
                    cfg_paths.append(d / f"config_method-{method}.json")
                _assert_configs_consistent(cfg_paths, VOLATILE_JSONL_CFG)
    print("preflight: all source counts, ids and configs consistent")


# -------------------------------------------------------------------------- merge


def leaf_dirs(args) -> list[Path]:
    leaves = [args.dst_data / args.subset_name]
    for stage in ("activations", "attention", "prompting", "chief"):
        for model in args.models:
            leaves.append(args.dst_outputs / stage / model / args.subset_name)
    return leaves


def prepare_dst(args) -> None:
    src_roots = {args.src_data.resolve(), args.src_outputs.resolve()}
    for leaf in leaf_dirs(args):
        resolved = leaf.resolve()
        assert not any(root == resolved or root in resolved.parents
                       or resolved in root.parents
                       for root in src_roots), f"{leaf} overlaps a source root"
        if leaf.exists():
            if not args.force:
                sys.exit(f"error: {leaf} exists (use --force to overwrite)")
            shutil.rmtree(leaf)
        leaf.mkdir(parents=True)


def merge_data(entries: list[Entry], src_data: Path, dst_data: Path,
               subset_name: str) -> None:
    dst_dir = dst_data / subset_name
    for e in entries:
        shutil.copyfile(src_data / e.orig_subset / f"{e.orig_stem}.json",
                        dst_dir / f"{e.new_stem}.json")
    with open(dst_data / "filename_map.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "orig_subset", "orig_index", "trajectory_id"])
        for e in entries:
            writer.writerow([e.new_stem, e.orig_subset, e.orig_stem, e.trajectory_id])
    print(f"data: {len(entries)} trajectories -> {dst_dir}")


def read_header(path: Path) -> tuple[int, dict]:
    """Return (header_len, header_dict) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def rewrite_safetensors(src: Path, dst: Path, entry: Entry, subset_name: str) -> None:
    """Copy src -> dst updating only the metadata header; tensor blob is verbatim.

    Format: 8-byte LE u64 header length | header JSON (space-padded to 8-byte
    alignment) | tensor blob. data_offsets are relative to the blob start, so
    resizing the header never invalidates them.
    """
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = json.loads(header["__metadata__"]["payload_metadata"])
        assert meta["filename"] == f"{entry.orig_stem}.json", src
        assert meta["subset"] == entry.orig_subset, src
        meta["filename"] = f"{entry.new_stem}.json"
        meta["subset"] = subset_name
        meta["orig_subset"] = entry.orig_subset
        meta["orig_filename"] = f"{entry.orig_stem}.json"
        header["__metadata__"]["payload_metadata"] = json.dumps(meta)
        buf = json.dumps(header, separators=(",", ":")).encode("utf-8")
        buf += b" " * (-len(buf) % 8)
        with open(dst, "wb") as g:
            g.write(struct.pack("<Q", len(buf)))
            g.write(buf)
            shutil.copyfileobj(f, g, length=16 << 20)


def merge_config(src_dirs: list[Path], dst_path: Path, cfg_name: str,
                 subset_name: str, volatile: set[str],
                 n_trajectories: int | None = None) -> None:
    cfg = _assert_configs_consistent([d / cfg_name for d in src_dirs], volatile)
    cfg["subset"] = subset_name
    if n_trajectories is not None:
        cfg["n_trajectories"] = n_trajectories
    dst_path.write_text(json.dumps(cfg, indent=2))


def merge_tensor_stage(stage: str, model: str, entries: list[Entry],
                       src_outputs: Path, dst_outputs: Path,
                       subset_name: str) -> None:
    src_root = src_outputs / stage / model
    dst_dir = dst_outputs / stage / model / subset_name
    for i, e in enumerate(entries, 1):
        rewrite_safetensors(src_root / e.orig_subset / f"{e.orig_stem}.safetensors",
                            dst_dir / f"{e.new_stem}.safetensors", e, subset_name)
        if i % 500 == 0 or i == len(entries):
            print(f"{stage}/{model}: {i}/{len(entries)}")
    merge_config([src_root / s for s in SUBSETS], dst_dir / "config.json",
                 "config.json", subset_name, VOLATILE_TENSOR_CFG)


def merge_jsonl_stage(stage: str, methods: list[str], model: str,
                      entries: list[Entry], src_outputs: Path, dst_outputs: Path,
                      subset_name: str) -> None:
    src_root = src_outputs / stage / model
    dst_dir = dst_outputs / stage / model / subset_name
    for method in methods:
        rows_by_subset = {
            s: _load_jsonl(src_root / s / f"predictions_method-{method}.jsonl")
            for s in SUBSETS
        }
        with open(dst_dir / f"predictions_method-{method}.jsonl", "w") as f:
            for e in entries:
                row = rows_by_subset[e.orig_subset][str(e.orig_stem)]
                row["id"] = str(e.new_stem)
                row["filename"] = f"{e.new_stem}.json"
                f.write(json.dumps(row) + "\n")
        merge_config([src_root / s for s in SUBSETS],
                     dst_dir / f"config_method-{method}.json",
                     f"config_method-{method}.json", subset_name,
                     VOLATILE_JSONL_CFG, n_trajectories=len(entries))
        print(f"{stage}/{model} [{method}]: {len(entries)} rows")


# ------------------------------------------------------------------------- verify


def verify(args, entries: list[Entry]) -> None:
    from safetensors import safe_open

    subset_name = args.subset_name
    all_stems = set(range(1, len(entries) + 1))
    by_new = {e.new_stem: e for e in entries}

    data_dir = args.dst_data / subset_name
    assert _stem_set(data_dir, ".json") == all_stems, "data stems != 1..N"

    # Deterministic spot-check sample: evenly spaced across the merge order.
    k = min(args.spot_check, len(entries))
    sample = [entries[i * len(entries) // k] for i in range(k)]

    for model in args.models:
        for stage in ("activations", "attention"):
            d = args.dst_outputs / stage / model / subset_name
            assert _stem_set(d, ".safetensors") == all_stems, \
                f"{d}: safetensors stems != data stems"
            cfg = json.loads((d / "config.json").read_text())
            assert cfg["subset"] == subset_name

            for e in sample:
                dst_file = d / f"{e.new_stem}.safetensors"
                src_file = (args.src_outputs / stage / model / e.orig_subset
                            / f"{e.orig_stem}.safetensors")
                with safe_open(dst_file, framework="pt", device="cpu") as f:
                    meta = json.loads(f.metadata()["payload_metadata"])
                src_n, src_header = read_header(src_file)
                src_meta = json.loads(src_header["__metadata__"]["payload_metadata"])
                assert meta["filename"] == f"{e.new_stem}.json"
                assert meta["subset"] == subset_name
                assert meta["orig_subset"] == e.orig_subset
                assert meta["orig_filename"] == f"{e.orig_stem}.json"
                assert meta["question_id"] == e.trajectory_id
                assert meta["mistake_step"] == src_meta["mistake_step"]
                assert meta["mistake_agent"] == src_meta["mistake_agent"]
                dst_n, _ = read_header(dst_file)
                blob_src = src_file.read_bytes()[8 + src_n:]
                blob_dst = dst_file.read_bytes()[8 + dst_n:]
                assert blob_src == blob_dst, f"{dst_file}: tensor blob differs"

        for stage, methods in (("prompting", PROMPT_METHODS), ("chief", ["chief"])):
            d = args.dst_outputs / stage / model / subset_name
            for method in methods:
                rows = _load_jsonl(d / f"predictions_method-{method}.jsonl")
                assert set(rows) == {str(s) for s in all_stems}, \
                    f"{d} [{method}]: ids != 1..{len(entries)}"
                for id_str, row in rows.items():
                    e = by_new[int(id_str)]
                    assert row["filename"] == f"{e.new_stem}.json"
                    assert row["question_id"] == e.trajectory_id
                cfg = json.loads((d / f"config_method-{method}.json").read_text())
                assert cfg["subset"] == subset_name
                assert cfg["n_trajectories"] == len(entries)

    for e in sample:
        traj = json.loads((data_dir / f"{e.new_stem}.json").read_text())
        assert traj["trajectory_id"] == e.trajectory_id

    print(f"verify: all checks passed ({len(entries)} trajectories, "
          f"{len(args.models)} models, spot-check {k} files/stage)")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--subset-name", default="magentic")
    ap.add_argument("--src-data", type=Path, default=Path("data/correct-error"))
    ap.add_argument("--dst-data", type=Path, default=Path("data/correct-full"))
    ap.add_argument("--src-outputs", type=Path, default=Path("outputs-correct-error"))
    ap.add_argument("--dst-outputs", type=Path, default=Path("outputs-correct-full"))
    ap.add_argument("--models", nargs="+", default=["deepseek-8b", "qwen3.5-9b"])
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing destination leaf dirs")
    ap.add_argument("--dry-run", action="store_true",
                    help="build map + preflight only, write nothing")
    ap.add_argument("--spot-check", type=int, default=5,
                    help="safetensors per (stage, model) to deep-verify")
    args = ap.parse_args()

    entries = build_renumber_map(args.src_data)
    counts = {s: sum(1 for e in entries if e.orig_subset == s) for s in SUBSETS}
    print(f"renumber map: {len(entries)} trajectories "
          f"({', '.join(f'{s}={n}' for s, n in counts.items())})")

    preflight(args.src_outputs, args.models, entries)

    if args.dry_run:
        print("dry-run: would create")
        for leaf in leaf_dirs(args):
            print(f"  {leaf}")
        return

    prepare_dst(args)
    merge_data(entries, args.src_data, args.dst_data, args.subset_name)
    for model in args.models:
        for stage in ("activations", "attention"):
            merge_tensor_stage(stage, model, entries, args.src_outputs,
                               args.dst_outputs, args.subset_name)
        for stage, methods in (("prompting", PROMPT_METHODS), ("chief", ["chief"])):
            merge_jsonl_stage(stage, methods, model, entries, args.src_outputs,
                              args.dst_outputs, args.subset_name)
    verify(args, entries)


if __name__ == "__main__":
    main()
