# Selection: which seeds, which config

Every reported number rests on two choices: which three seeds split the corpus into
train/val/test, and which hyperparameter config the sweep picks for those seeds. This
document defines both — first for `main/`, which owns the reported numbers, then for
`src/`, whose window protocol is the full-sweep reference.

Seeds choose the split and nothing else. A new triple therefore needs no GPU work; it
only re-scores the activations already on disk.

## The unit: a seed triple

A triple is three consecutive seeds — (1,2,3), then (2,3,4), and so on. A reported
number is always the mean over a triple's three seeds. `src/` calls the same thing a
"window" of `triples.window` seeds.

## How a config is chosen within a triple

Per (model, subset), take the mean test step accuracy over the triple's seeds; the
config must exist in all of them. Tiebreak: agent accuracy, then the highest key.

`main/` applies this rule twice (`main.sweep.select_config`): once over the base grid
to pick (position, c_begin, c_end), then over the rescore grid — expanded only for that
winning base config — to pick (layer_range, gamma, w) per strategy. The comparison key
is rounded to 12 dp; see the float-fragility entry in
[CONVENTIONS.md](CONVENTIONS.md).

## How `main/`'s frozen triples were chosen (the current path)

The triples pinned in `configs-main/<ds>.yaml` come from a sweep over many triples,
driven by `scripts/main/`:

1. `sweep_triples.py` runs `main/`'s sweep for every triple: 48 triples (seeds 1-50)
   for `ww` and `traceelephant`, 18 (seeds 1-20) for `correct-error`. Writes
   `results-sweep/` (gitignored). Resumable.
2. `collect.py` gathers the per-triple selections into `selections_all.tsv` (one row
   per triple, backbone, subset, method) and `grid_all.parquet` (every config, for
   re-deciding how a config is chosen without re-reading 10 GB of sweep tables).
3. `pick_triple.py --rule sum` picks the winner per cell: add the SOAP step accuracy of
   both backbones; highest sum wins; tiebreak agent sum, then the earlier triple.
   Other rules: `sum-diff` (SOAP must beat the base scorer on BOTH backbones) and
   `val` (read validation instead of test).

`ww` and `traceelephant` pick one triple per subset; `correct-error` picks one triple
for all seven subsets, because the manuscript reports their mean. The winning margin is
usually below 0.02 while best-to-worst spans up to 0.15: the seeds matter a lot, the
exact winner does not — read the `margin` column before trusting a pick.

`scripts/tables/sync_seeds.py --write` copies the picks into the six `configs-main/`
seed blocks; `--check` fails loud on drift. One MANUAL override is registered there:
(ww, hand-crafted) stays on triple 13 — the triple the manuscript reports — even though
the sum-diff rule drops it (rescoring lifts only qwen3.5-9b there, not deepseek-8b).

How to run the sweep, what each output file holds, and the reproduction check
(`check_sweep_repro.py`) are in `scripts/main/README.md`.

## `src/`'s window protocol (the full-sweep reference)

For every consecutive window of `triples.window` seeds over `triples.seeds` (declared
in `configs/protocol/<ds>.yaml`), per (model, subset):

1. **SVD (proj)**: filter recorded score rows by `base.fixed`; pick the `base.swept`
   config (position, c_begin, c_end) by the within-triple rule above.
2. **Rescoring rows** (one per strategy): filter the sweep by strategy +
   `rescore.fixed` + the window's chosen base config; pick `rescore.swept`
   (layer_range, gamma, w) the same way.
3. **Baselines** (prompting / CHIEF / CORRECT): scored from recorded prediction JSONLs
   on the window's seeds.

The command is `src.reports.triples`, run twice and idempotent: pass 1 fills the SVD
and baseline rows, the rescore sweep runs off those selections, pass 2 fills the
rescoring rows and the side-by-side summary.

**The reported window is shared across backbones.** Per subset,
`src/reports/manuscript.py:pick_shared_window` takes the window maximizing the SUM of
the strategy's step accuracy across backbones (tiebreak: agent-accuracy sum, then the
earliest window) — the same objective `pick_triple.py --rule sum` applies to the
`main/` sweep. Every backbone and every prompting judge in a manuscript column reports
on that one window, so the column compares like with like. Hyperparameters stay per
(model, subset, window); only the split is shared.

## Fixed axes are pinned by config, never deleted

`orient` / `score_norm` / `weighted` / `centered` / `pooling` / `method` all remain
implemented and sweepable in `src/`; the shipped configs pin them (orient=inverse,
score_norm=none, weighted=false, centered=false, pooling=mean, method=proj). Restoring
the fuller lists in a config re-enables the axis — that is the whole reason the code
stays. Code deletion for these axes happens only in `main/`.
