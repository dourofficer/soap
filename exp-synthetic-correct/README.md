# exp-synthetic-correct

Does synthetic data help failure-attribution scoring? SVD is fit **only on the
train split of correct-full/magentic** (the merged CORRECT-Error dataset); the
fitted singular vectors then score the val/test splits of **correct-full, ww,
and traceelephant**, and CRR rescoring runs on top of those cross-dataset base
scores. Fully isolated from the main pipeline: code imports from `src/` /
`experiments/`, but every result lands under `exp-synthetic-correct/results/`.
Existing `outputs-<ds>/activations` and `outputs-<ds>/attention` are read-only
inputs (they are split-agnostic, so the new split protocol reuses them as-is).

## Protocol

| | split | mechanism |
|---|---|---|
| correct-full (source) | train/val/test = 50/25/25 | production nested split: `_validate_and_derive_split_ratios` + two same-seed `split_data` passes |
| ww, traceelephant | val/test = 50/50 | one `split_data(files, 0.5, seed)` pass; val = first part (the production 3-way validator rejects train-less splits, so it is bypassed by construction) |

- Models: `deepseek-8b`, `qwen3.5-9b` (the only two with complete
  activations+attention across all three datasets). Seeds: 1, 2, 3 — the same
  seed drives both the source's nested split and each target's 50/50 split.
- Splits are per (model, subset) over trajectory filenames sorted by
  `int(stem)`, exactly as in production.
- Base scores: full production config grid (positions × centered × band ×
  weighted + norm baselines, N_COMPONENTS=20), metrics step@1/agent@1.
- Selection: best config per (pooling, seed) under **both** conventions —
  `sel_by=val` (leak-free, the headline) and `sel_by=test` (mirrors the
  production tables). Selected configs' raw per-step scores are persisted so
  the rescore stage never refits.
- CRR: production axes (orients × 4 layer ranges × gammas × ws) plus
  `gamma=0.0`, which must reproduce the undiscounted metrics exactly.

## Running

From the repo root (scripts are invoked by path — the directory name is
hyphenated, so there is no `python -m` form):

```bash
python exp-synthetic-correct/sweep.py --stage all --dry-run       # print plan
CUDA_VISIBLE_DEVICES=0 python exp-synthetic-correct/sweep.py --stage svd
python exp-synthetic-correct/sweep.py --stage tables --validate
CUDA_VISIBLE_DEVICES=0 python exp-synthetic-correct/sweep.py --stage rescore
python exp-synthetic-correct/sweep.py --stage disc
python exp-synthetic-correct/sweep.py --stage summary
```

`--set` takes dot-path overrides as in the main sweeps, e.g. a smoke run:

```bash
python exp-synthetic-correct/sweep.py --stage all \
  --set models='[deepseek-8b]' --set poolings='[mean]' --set seeds='[1]' \
  --set positions='[embed, act/15]' \
  --set rescore.gammas='[0.0, 0.5]' --set rescore.ws='[all]'
```

## Results layout

```
results/
├── svd/{model}/{ds}/{subset}/svd_pooling-{p}_seed-{s}.tsv   full config grid (val+test)
├── base-scores/{model}/{ds}/{subset}/selected_pooling-{p}_seed-{s}.pt
│       selected configs + raw per-step score tensors + split file lists
├── undiscounted/{model}/{ds}/{subset}/weighted_false.tsv    production schema + sel_by
├── rescore/sweep/{model}/{ds}/{subset}/svd.tsv              CRR sweep
├── rescore/reduced/{model}/{ds}/{subset}/svd.tsv            best per (pooling, seed, sel_by), CRR params picked on disc TEST
├── rescore/reduced/{model}/{ds}/{subset}/svd_valsel.tsv     …picked on disc VAL
└── summary/results_table.tsv, per_seed.tsv                  mean±std over seeds per convention
```

Reading the summary: `convention=val` means the base config was selected on
val AND the CRR hyperparameters were selected on disc-val — reported metrics
are always **test**. `convention=test` mirrors the production convention
(optimistically biased for a generalization claim; kept for comparability).

## Conventions that bite (inherited)

- Raw SVD projection scores are "lower = error" (`direction=asc` in the grid
  TSVs and in the persisted tensors). Rescoring orients them to
  "higher = error" (`negate`/`inverse`/`sigmoid`) before discounting and
  evaluates `direction=desc`.
- CRR `layer_range` labels index **attention rows**, not layer numbers:
  deepseek-8b stores 32 rows (`0-8/8-16/16-24/24-32`); qwen3.5-9b stores only
  8 rows (layers 3,7,…,31 → labels `0-2/2-4/4-6/6-8`). Never compare ranges
  across models.
- The cross-dataset scorer hard-asserts that every target (pooling, layer)
  key exists in the source fit — `score_all` would otherwise skip silently.
- Production's `src/svd/score.py --positions` is a no-op; here `positions`
  genuinely restricts `weight_names` at load time (used by smoke runs).

## Verification hooks

- **γ=0 rows** in the rescore sweep must equal the undiscounted metrics for
  the `negate` and `inverse` orients (exact rank-preserving maps on the
  positive raw scores). `sigmoid` can violate this legitimately: large-
  magnitude scores (e.g. norm-baseline configs, |s|≳90) saturate
  `sigmoid(-s)` to exactly 0/1 in float32, collapsing the ranking into ties —
  an inherited property of the production CRR orient, observed on
  ww/hand-crafted with a norm_l1-selected config.
- `build_tables --stage undisc` re-derives every winner from the full-grid
  TSV via the production `best_per_group` reducer (hard assert); `--validate`
  additionally recomputes metrics from the persisted score tensors.
- Equivalence regression: with `--set splits='{train: 0.3, val: 0.2, test: 0.5}'`
  and targets limited to correct-full, the stage-1 TSV must match the
  production `outputs-correct-full/weighted-projections/325/.../svd_pooling-*.tsv`.
