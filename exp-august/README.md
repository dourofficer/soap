# exp-august — shared-config / shared-seed protocols

Self-contained experiment: everything produced by code in this directory stays under
`exp-august/outputs/`. Code freely imports from `src/` but never modifies it; the one
behavioural change needed (reading extracted reps/attention from the main tree while
writing here) is a monkeypatch inside `run_rescore.py`.

Two protocols share the same config and axis declarations:

1. **Shared-config** (`focused_table.py` + `run_rescore.py`) — one cell per
   (model, subset); the 3 seeds are chosen jointly with the SVD config.
2. **Seed-window / "triples"** (`triples_table.py` + `run_rescore_triples.py`) —
   the same selection repeated for every consecutive triple of seeds (1,2,3),
   (2,3,4), …; the triple's seeds are given, only configs are selected. Used to
   choose seeds afterwards from the per-triple report (details below).

Selection everywhere: mean test step-acc, tiebreak agent acc, then HIGHEST config
key. Test-selected (optimistic).

## Protocol 1: shared-config

Per (model, subset) cell:

1. **SVD (proj)** — fixed axes `pooling=mean, method=proj, centered=False,
   weighted=False, direction=asc`; swept axes `position` and `[c_begin, c_end)`.
   Config and 3 seeds chosen **jointly**: per config, mean test step-acc over its own
   best-3 seeds (over the manifest's seed list); argmax.
2. The winning config's 3 seeds become the cell's **shared seeds**; every other row
   is evaluated on exactly those seeds.
3. **SOAP (full)** — backprop rescoring (manuscript eq:recap) on top of the cell's
   fixed base config. Rescore axes: fixed `orient=inverse, score_norm=none`;
   swept (`layer_range, gamma, w`) selected by best mean test step-acc over the
   shared seeds. Needs the sweeps from `run_rescore.py` (the main tree's sweeps only
   cover each seed's own base winner).
4. Baselines (All-at-once, Step-by-step, Binary search, CORRECT, CHIEF) are scored
   from the recorded prediction JSONLs on the shared seeds' test splits.

## Protocol 2: seed-window ("triples")

The purpose is seed selection — run all triples, then
pick the seeds you like based on the reported numbers.

Each dataset has a list of seeds for which the score stage was actually run (all four
datasets: 1–20 as of 2026-08-05; the list is written in the dataset's config, see the
Configs section). With `average_subsets: true` (correct-error), `triples_summary.tsv`
additionally carries one `subset=average` row per (model, window): the macro-average
of every metric over the dataset's subsets, for reporting a single CE number. Sliding a window of size 3 over that
list gives the seed triples (1,2,3), (2,3,4), (3,4,5), … — e.g. 18 triples for 20
seeds. Every triple is processed independently, exactly like one Protocol-1 cell
except that the triple's three seeds are **given** instead of being chosen:

1. **SVD (proj).** Among all base configs — the representation layer `position` and
   the spectral band `[c_begin, c_end)`; the remaining base axes stay fixed as in
   Protocol 1 — pick the one whose mean test step-accuracy over the triple's three
   seeds is highest. A config only qualifies if it was scored in all three seeds.
2. **SOAP (full).** Keep that triple's base config. Among all rescore settings —
   `layer_range`, `gamma`, `w`, with `orient=inverse` and `score_norm=none` fixed —
   pick the one with the highest mean test step-accuracy over the same three seeds.
   This step needs rescore sweeps that cover each triple's chosen base config on the
   triple's seeds; `run_rescore_triples.py` produces them (one sweep per
   (model, subset), covering all triples at once).
3. **Baselines.** All-at-once, Step-by-step, Binary search, CORRECT, and CHIEF are
   scored on the same three seeds' test splits, so every triple has its own baseline
   numbers to compare against.

Ties in any selection are broken by agent accuracy, then by the highest config key —
the same rule as Protocol 1.

After the run, `triples_summary.tsv` lists for every triple: the SVD step-accuracy
(`svd_step`), the SOAP step-accuracy (`soap_step`), and their difference
(`diff_step`). These are the two criteria for choosing seeds — you want a triple
where SOAP is accurate *and* clearly better than the base score. Nothing is chosen
automatically; the final call is made by you from that table. Per-triple details
(every hyperparameter and the baseline accuracies) are in `triples_selection.tsv`.

## With-GT mode (`gt: true`)

`configs/{ww,traceelephant}-gt.yaml` are copies of the plain configs plus `gt: true`.
That one flag switches both protocols to the **with-GT setting** — the proxy inputs
were extracted with a pinned `[question, answer]` block (see `src/data/context.py`,
`GT=1 ./scripts/extract.sh`). Three root families follow the flag:

- **artifact reads** (reps, attention, recorded scores) → the repo's `outputs-gt/<ds>/`
  (populated by GT extraction + a GT score-stage run:
  `python -m src.score.run --config configs/score/<ds>.yaml
  --set outputs_base=outputs-gt/<ds> --set "seeds=[1,...,20]"`);
- **protocol writes** (base tables, sweeps, selection/summary tables, provenance) →
  `exp-august/outputs-gt/<ds>/`;
- **baseline reads** stay on the plain `outputs/<ds>/baselines/` — their GT-ness is a
  corpus property (ww/TE prompts already embed the answer), not a knob here.

Only ww and traceelephant have gold answers; a `-gt` config for correct-* would fail
at extraction. Run loops are identical to the plain ones with `$ds-gt.yaml` configs.

## Configs

`configs/<ds>.yaml` is the single config for both scripts. Each stage declares its
axes explicitly — `base.fixed` / `base.swept` for SVD (proj), `rescore.fixed` /
`rescore.swept` (+ `rescore.strategy`) for SOAP (full):

```yaml
base:
  fixed: {pooling: mean, method: proj, centered: false, weighted: false, direction: asc}
  swept: [position, c_begin, c_end]
rescore:
  strategy: backprop
  fixed: {orient: inverse, score_norm: none}
  swept: [layer_range, gamma, w]
triples:            # Protocol 2 only
  window: 3         # how many consecutive seeds form one triple
  seeds: [1, 2, ..., 20]   # the scored seeds the window slides over
```

Fixed values act as row filters; swept axes are what gets selected (one shared value
per cell). The bottom of the config lists the full grid `run_rescore.py` sweeps
(`gammas`, `ws`, `orients`, `score_norms`, `strategies`, `n_ranges`); because the
sweep always covers that whole grid, moving an axis between `fixed` and `swept`
later only requires re-running `focused_table.py`, not the sweep — unless the new
value lies outside the recorded grid.

## Run (from repo root)

Protocol 1 (shared-config):

```bash
for ds in ww correct-error correct-full traceelephant; do
  python exp-august/focused_table.py --config exp-august/configs/$ds.yaml  # selection + table (SOAP (full) empty on first pass)
  python exp-august/run_rescore.py  --config exp-august/configs/$ds.yaml   # sweep the winning config on its 3 shared seeds
  python exp-august/focused_table.py --config exp-august/configs/$ds.yaml  # final table incl. SOAP (full)
done
```

Protocol 2 (seed-window):

```bash
for ds in ww correct-error correct-full traceelephant; do
  python exp-august/triples_table.py      --config exp-august/configs/$ds.yaml  # per-window selection + baselines (SOAP empty on first pass)
  python exp-august/run_rescore_triples.py --config exp-august/configs/$ds.yaml # sweep every window's base config (heavy: hours per dataset)
  python exp-august/triples_table.py      --config exp-august/configs/$ds.yaml  # final: fills SOAP (full) per window
done
```

The rescore runners skip cells whose sweep file already exists, so re-running a loop
after a config change is cheap. BUT the skip is blind to content: if the base
selection changed (e.g. after editing the manifest seed list or the config), pass
`--force` so stale sweeps are rebuilt — a stale sweep is detected at selection time
(the base-config filter finds no rows) and shows up as an empty SOAP cell.
All scripts accept `--model M` / `--subset S` narrowing.

## Outputs

Protocol 1: shared-config

- `outputs/<ds>/tables/325/results_focused.tsv` — the display grid.
- `outputs/<ds>/tables/325/focused_selection.tsv` — per row: **all** hyperparameters
  (fixed and swept), the shared seeds, and the reported means.
- `outputs/<ds>/reduced/325/<model>/<subset>/base_focused_test.tsv` — the 3-row base
  table fed to the rescore sweep.
- `outputs/<ds>/rescore/325/<model>/<subset>/sweep_focused.tsv` — the fixed-config
  rescore sweep over the full grid (all orients/score_norms/strategies recorded;
  the fixed/swept split is applied at selection time).

Protocol 2: seed-window ("triples")

- `outputs/<ds>/tables/325/triples_selection.tsv` — one row per (model, subset,
  window, method) with all hyperparameters, the window's seeds, and accuracies.
- `outputs/<ds>/tables/325/triples_summary.tsv` — per window: `svd_step`,
  `soap_step`, `diff_step` (+ agent columns) — the seed-choice criteria.
- `outputs/<ds>/reduced/325/<model>/<subset>/base_triples_test.tsv` — union base
  table (every window's chosen config x its seeds, deduplicated).
- `outputs/<ds>/rescore/325/<model>/<subset>/sweep_triples.tsv` — its rescore sweep.

Both: `outputs/<ds>/{tables,rescore}/runs/*.json` — provenance.
