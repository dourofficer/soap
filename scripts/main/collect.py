"""Collect the triple sweep into two artifacts, so re-picking never touches the TSVs.

    selections_all.tsv   every per-triple selection.tsv, concatenated (~8.4k rows)
                         -> what pick_triple.py reads; a protocol change costs seconds
    grid_all.parquet     EVERY config, seed-averaged per triple (~11.3M rows)
                         -> lets you change how the CONFIG is chosen (val-selected,
                            top-k, a different tiebreak) without re-reading 10.6 GB

Both are built in one pass. The parquet is streamed cell by cell into a ParquetWriter and
flushed in ~256k-row groups, so peak memory is one cell (~24k rows), never 11.3M.

TWO DTYPE HAZARDS, both handled by reading the config columns as STRINGS and grouping on
them:
  * `w` mixes 1..5, "all" and "-" in one column;
  * pandas' int->float round-trip turns c_begin 9 into 9.0, which is exactly the
    `norm_val` hazard main/sweep.py already works around.
Metrics stay float64: they are rationals like 26/63 and the picker compares them at 12
decimal places, which float32 could not represent.

    python scripts/main/collect.py
    python scripts/main/collect.py --force        # rebuild even if outputs exist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "main"))

from sweep_triples import Unit, cells, load_cfg, tag, triples, valid_selection, valid_sweep  # noqa: E402

OUT = REPO / "results-sweep"
# NOTE the column is `with_gt`, not `gt`: pandas already has DataFrame.gt / Series.gt
# (greater-than), so `df.gt` silently returns a METHOD instead of the column and every
# comparison against it comes out False. Renaming removes the trap for everyone
# downstream rather than requiring df["gt"] discipline forever.
CONFIG_COLS = ["position", "c_begin", "c_end", "strategy", "layer_range", "gamma", "w"]
ID_COLS = ["with_gt", "triple", "dataset", "model", "subset"]
ROWGROUP = 256_000

SCHEMA_BASE = [
    ("with_gt", pa.bool_()), ("triple", pa.int16()), ("dataset", pa.string()),
    ("model", pa.string()), ("subset", pa.string()),
    ("position", pa.string()), ("c_begin", pa.int8()), ("c_end", pa.int8()),
    ("strategy", pa.string()), ("layer_range", pa.string()),
    ("gamma", pa.float64()), ("w", pa.string()), ("n_seeds", pa.int8()),
]


def metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "_acc_" in c]


def read_cell(path: Path) -> pd.DataFrame:
    """Config columns as strings — see the dtype hazards in the module docstring."""
    return pd.read_csv(path, sep="\t",
                       dtype={c: "string" for c in ("position", "strategy",
                                                    "layer_range", "w", "c_begin",
                                                    "c_end", "gamma")})


def reduce_cell(df: pd.DataFrame, unit: Unit, model: str, subset: str) -> pd.DataFrame:
    """Seed-average one cell: 3 rows per config -> 1."""
    mcols = metric_cols(df)
    g = df.groupby(CONFIG_COLS, sort=False, observed=True, dropna=False)
    out = g[mcols].mean()
    out["n_seeds"] = g.size()
    out = out.reset_index()
    bad = int((out["n_seeds"] != 3).sum())
    if bad:
        raise SystemExit(f"{unit.key()}/{model}/{subset}: {bad} configs not present in "
                         f"all 3 seeds — select_config would have skipped them")
    # Grouped on strings (hazard-free); cast to their real types only now, for the
    # on-disk schema. `w` and `layer_range` stay strings — they carry "all" and "-".
    out["c_begin"] = out["c_begin"].astype("int8")
    out["c_end"] = out["c_end"].astype("int8")
    out["gamma"] = out["gamma"].astype("float64")
    out["n_seeds"] = out["n_seeds"].astype("int8")
    for c in ("position", "strategy", "layer_range", "w"):
        out[c] = out[c].astype("object")
    out.insert(0, "subset", subset)
    out.insert(0, "model", model)
    out.insert(0, "dataset", unit.dataset)
    out.insert(0, "triple", unit.triple[0])
    out.insert(0, "with_gt", unit.gt)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed-lo", type=int, default=1)
    p.add_argument("--seed-hi", type=int, default=50)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sel_path, grid_path = OUT / "selections_all.tsv", OUT / "grid_all.parquet"
    if sel_path.exists() and grid_path.exists() and not args.force:
        print(f"[skip] {sel_path.name} and {grid_path.name} exist (--force to rebuild)")
        return 0

    units = [Unit(gt, tuple(t), ds)
             for gt in (False, True)
             for t in triples(args.seed_lo, args.seed_hi)
             for ds in ("ww", "traceelephant", "correct-error")]
    cfgs = {d: load_cfg(d) for d in ("ww", "traceelephant", "correct-error")}

    sel_frames, index_rows, writer, buf, buf_rows = [], [], None, [], 0
    schema = None
    done = skipped = 0

    for unit in units:
        cfg = cfgs[unit.dataset]
        cs = cells(unit.dataset, cfg)
        sel = unit.root / "select" / "selection.tsv"
        if not valid_selection(sel, len(cs)):
            skipped += 1
            continue
        s = pd.read_csv(sel, sep="\t")
        s.insert(0, "triple", unit.triple[0])
        s.insert(0, "with_gt", unit.gt)
        s.insert(2, "dataset", unit.dataset)
        sel_frames.append(s)

        for model, subset in cs:
            cell = unit.root / "sweep" / model / subset / "sweep.tsv"
            if not valid_sweep(cell):
                raise SystemExit(f"incomplete sweep for {unit.key()}/{model}/{subset}")
            red = reduce_cell(read_cell(cell), unit, model, subset)
            if schema is None:
                schema = pa.schema(SCHEMA_BASE
                                   + [(c, pa.float64()) for c in metric_cols(red)])
                writer = pq.ParquetWriter(grid_path, schema, compression="zstd",
                                          compression_level=7, write_statistics=True)
            red = red[[f.name for f in schema]]
            buf.append(red)
            buf_rows += len(red)
            index_rows.append({"with_gt": unit.gt, "triple": unit.triple[0],
                               "dataset": unit.dataset, "model": model,
                               "subset": subset, "rows": len(red)})
            if buf_rows >= ROWGROUP:
                writer.write_table(pa.Table.from_pandas(pd.concat(buf), schema=schema,
                                                        preserve_index=False))
                buf, buf_rows = [], 0
        done += 1
        if done % 24 == 0:
            print(f"  {done} units collected...", flush=True)

    if writer is None:
        print("nothing collected")
        return 1
    if buf:
        writer.write_table(pa.Table.from_pandas(pd.concat(buf), schema=schema,
                                                preserve_index=False))
    writer.close()

    allsel = pd.concat(sel_frames, ignore_index=True)
    allsel.to_csv(sel_path, sep="\t", index=False)
    idx = pd.DataFrame(index_rows)
    idx.to_csv(OUT / "grid_index.tsv", sep="\t", index=False)

    n_grid = pq.ParquetFile(grid_path).metadata.num_rows
    print(f"\nunits collected: {done}   skipped (not complete): {skipped}")
    print(f"  {sel_path}  {len(allsel):,} rows")
    print(f"  {grid_path}  {n_grid:,} rows  "
          f"{grid_path.stat().st_size/2**20:.0f} MB")
    assert n_grid == int(idx["rows"].sum()), "parquet rows != grid_index sum"
    print("  parquet row count matches grid_index.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
