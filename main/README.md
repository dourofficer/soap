# `main/` — running SOAP

The method and the six frozen choices are documented in `main/__init__.py`; the repo-level
picture is in `../README.md` and `../CLAUDE.md`. This file is the operational guide.

## The four commands

```bash
python -m main extract   --config configs-main/ww.yaml     # GPU; only for a new dataset
python -m main sweep     --config configs-main/ww.yaml     # base grid -> pick -> rescore grid
python -m main select    --config configs-main/ww.yaml     # read-side reduction, no recompute
python -m main reproduce --config configs-main/ww.yaml --row backprop
```

Shared flags: `--set key=value` (repeatable, dot-path, YAML-parsed), `--model`,
`--subset`, `--device`, `--force`, `--dry-run`. With-GT is `--set gt=true` on every
command — it selects the `results-gt/` tree and the pinned `[question, answer]` context.

There is deliberately **no `--seed`**: seeds are frozen per subset, and a reported number
is the mean over that triple, so narrowing to one seed would produce a number that means
nothing.

## Output layout

```
results-nogt/<ds>/                       # results-gt/<ds>/ under --set gt=true
├── activations/<model>/<subset>/        # seeded by scripts/seed_results.sh
├── attention/<model>/<subset>/
├── sweep/<model>/<subset>/{sweep.tsv, run_stamp.json}
├── select/{selection.tsv, run_stamp.json}
└── reproduce/<model>/<subset>/<row>_seed-<n>_<split>.{steps.tsv,preds.tsv,json}
```

`sweep.tsv` holds both grids in one schema. Base rows carry the `strategy = "base"`
sentinel with `layer_range="-"`, `w="-"`, `gamma=0.0` and are dense over
`position x c_begin x c_end x seed`; rescore rows exist only for the selected base config,
so the table is deliberately ragged.

`selection.tsv` has one row per `(model, subset, row)` with `row` in
`svd | backprop | succ-strong | succ-near`, at full float precision.

## Changing the seeds

The frozen triples live under `seeds:` in `configs-main/<ds>.yaml`, one per subset, shared
by every backbone. They come from the shared-window rule — per subset, the window
maximizing the SUM of the strategy's step accuracy across backbones.

Editing them (or `splits`) changes the seed→partition mapping, so the existing tables
would no longer mean the same thing. `run_stamp.json` catches that: the next run fails
with a diff of the changed knobs rather than silently mixing partitions. Re-run with
`--force` to recompute, or point `--set results_base=<other-tree>` at a fresh tree.

## Reusing the reference extractions

`scripts/seed_results.sh` hardlink-copies `outputs/<ds>/{activations,attention}` into
`results-nogt/<ds>/` (and `outputs-gt/` into `results-gt/`). The two extractors write
byte-compatible artifacts — same `{step}.{pool}.{shorthand}` keys, both poolings, the same
attention key quartet — so this is the same file, not a conversion, and costs no disk.
`scripts/check_extract_parity.py` proves that equivalence tensor-by-tensor.

`main extract` is therefore only needed for a dataset with no extractions yet. It is
resumable: existing `.safetensors` are skipped.

## Cost

Per `(model, subset)`, the base pass is `positions x 210 bands x 3 seeds` scored in
batched metric calls (~11 positions for Qwen3.5, which exposes only its full-attention
blocks, vs ~35 for DeepSeek), and the rescore pass is `4 ranges x 6 ws x 3 strategies x 7
gammas x 3 seeds` — with all gammas evaluated in one broadcast. That lands around 8k–24k
rows and a ~2 MB table per cell.
