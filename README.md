# SOAP — Spectral Scoring with Attention-Guided Propagation

Failure attribution in LLM multi-agent systems from a **proxy model's internals**. Given
a failed trajectory, predict the **decisive-error step** — the earliest step that
irrecoverably derailed the run — and hence the responsible agent.

Each step's hidden state gets a **base score** measuring how far it sits from the "common
mode" of typical steps (SVD geometry). A **rescoring strategy** then corrects that score
using the causal dependency structure the model itself reveals through attention, so a
late symptom is not mistaken for the cause:

```
S(s_t)  = 1 / (mean_{c in C} <v_t, V[:,c]>^2 + eps)                    higher = error
S~(s_i) = S(s_i) + gamma * (sum_{t>i} w_{i,t} S(s_t)) / (sum_{t>i} w_{i,t})
t_hat   = argmax_i S~(s_i)
```

Evaluation is step@k / agent@k, ranked within each trajectory. See `CLAUDE.md` for the
method in detail and the conventions that bite.

## Two packages

| | `main/` | `src/` |
|---|---|---|
| role | **primary runner** | the full sweep machinery |
| axes | pooling / centering / scorer / orientation / normalization **frozen in code** | all of them **implemented and sweepable**, pinned by config |
| seeds | one hardcoded 3-seed triple per subset | 18 seed windows over seeds 1–20 |
| stages | `extract → sweep → select → reproduce` (one command each) | `score → select → rescore → select` (two-pass) |
| owns | reproduction, per-step inspection | manuscript tables, baselines, figures |
| size | ~2.2k lines | ~5.4k lines |
| writes | `results-nogt/`, `results-gt/` | `outputs/`, `outputs-gt/` (frozen reference) |

`main/` imports nothing from `src/` (enforced by a test). Use `main/` for new runs; keep
`src/` for anything that needs an axis back, or the manuscript-shaped tables.

## Layout

```
data/<ds>/<subset>/*.json         the corpus (input; never written by a run)
main/                             the simplified runner — see main/README.md
  config.py   paths + frozen seeds + the run-stamp drift guard
  data.py     trajectories + per-step token spans (incl. with-GT)
  extract.py  activations + streaming-attention (faithful port of src/extract)
  stores.py   representation loading + the seed->partition mapping
  metrics.py  step@k / agent@k (loop + batched)
  score.py    uncentered SVD, folded-inverse proj score, ens-mid3
  rescore.py  attention -> W -> the three strategies
  sweep.py    base grid -> select -> rescore grid -> sweep.tsv, selection.tsv
  reproduce.py  re-run ONE frozen row -> per-step scores/ranks/predictions
src/                              the full sweep
  common/     config merge + --set, derived paths, uniform CLI, run provenance
  data/       trajectory representation + per-step context construction
  extract/    activations.py, attention.py
  stores.py   representation loading + train/val/test splits
  metrics.py  step@k / agent@k: reference loop + vectorized batch
  score/      scorers registry, SVD fit + grid, layer ensemble, runner
  rescore/    attention weights -> W, backprop + succ-strong/near, runner
  reports/    the seed-window ("triples") protocol, baselines, manuscript tables
  analysis/   geometry probe, score-distribution + qualitative + contamination figures
configs/      datasets/<ds>.yaml (manifest), score/, protocol/ (select + rescore)
configs-main/ <ds>.yaml — one config per dataset for main/
scripts/      extract.sh, run_pipeline.sh, seed_results.sh, check_extract_parity.py,
              check_gt_parity.py, copy_inputs.sh, import_prompting.py
tests/        test_main.py (main/ + parity with src/), test_invariants.py — both CPU-only
outputs/<ds>/       everything src/ produces (frozen reference)
results-nogt/<ds>/  everything main/ produces; results-gt/ for the with-GT setting
```

## Quickstart — `main/`

```bash
# 1. seed the results trees from the frozen extractions (hardlink copy, free)
scripts/seed_results.sh

# 2. sweep -> select -> inspect
python -m main sweep     --config configs-main/ww.yaml
python -m main select    --config configs-main/ww.yaml
python -m main reproduce --config configs-main/ww.yaml --row backprop

cat results-nogt/ww/select/selection.tsv
```

With-GT is `--set gt=true` on every command (writes `results-gt/`). For a dataset with no
extractions yet, run `python -m main extract --config configs-main/<ds>.yaml` first — that
is the only GPU-heavy stage.

## Quickstart — `src/`

```bash
# corpus in data/<ds>/<subset>/*.json, then extract representations + attention
DATASET=correct-error GPU=0 ./scripts/extract.sh
# score -> select (pass 1) -> rescore -> select (pass 2)
DS=correct-error bash scripts/run_pipeline.sh
cat outputs/correct-error/tables/325/triples_summary.tsv
# manuscript-shaped tables (shared seed triple per column)
python -m src.reports.manuscript --strategy backprop
```

Adopting extractions produced elsewhere: `SRC=/path/to/tree DATASET=<ds> ./scripts/copy_inputs.sh`.

## Stages

Every `src/` stage takes the same CLI (`--config`, `--set k=v`, `--model/--subset/--seed`,
`--device`, `--dry-run`, `--force`) and runs from the repo root:

| stage | command | output |
|---|---|---|
| extract reps | `python -m src.extract.activations --model M --model-path P --input data/<ds> --subset S --output outputs/<ds>/activations/M/S` | `.safetensors` per trajectory |
| extract attn | `python -m src.extract.attention --model M --model-path P --input data/<ds> --subset S --output-root outputs/<ds>/attention` | `.safetensors` per trajectory |
| score | `python -m src.score.run --config configs/score/<ds>.yaml` | `scores/<tag>/<model>/<subset>/seed-<n>.tsv` |
| select (x2) | `python -m src.reports.triples --config configs/protocol/<ds>.yaml` | `tables/<tag>/triples_{selection,summary}.tsv` |
| rescore | `python -m src.rescore.run --config configs/protocol/<ds>.yaml` | `rescore/.../sweep.tsv` |
| tables | `python -m src.reports.manuscript --strategy backprop` | `outputs/manuscript-tables/*.tsv` |
| geometry | `python -m src.analysis.geometry --config configs/datasets/<ds>.yaml` | stdout tables |

`main/`'s equivalents are the four subcommands of `python -m main`.

## Inspecting a config (not just scoring it)

The sweep collapses each run to one number. To see what a config *did*, reproduce it —
this re-runs one frozen winner and writes the per-step signal:

```bash
python -m main reproduce --config configs-main/ww.yaml --row backprop --split test
```

`*.steps.tsv` has one row per (trajectory, step) with `base` / `final`, the
within-trajectory `rank`, `is_pred` and the gold `is_mistake` flag — plot a trajectory's
score curve directly, or concatenate several rows' files to diff methods on identical
rows. `*.preds.tsv` gives one row per trajectory including `true_step_rank`, which is what
you want for error analysis. Reproduction asserts the reproduced mean accuracy equals the
recorded one. `--split all` re-uses the train-fitted SVD to score every trajectory in the
subset, including ones never in an eval split.

## Adding a dataset

1. `data/<ds>/<subset>/*.json` — each with `history` (ordered turns; step `t ==
   history[t]`, 0-indexed), `question_ID`, `mistake_agent`, `mistake_step`, and optionally
   `level` / `question` / `ground_truth`.
2. `configs-main/<ds>.yaml` — models, `model_paths`, subsets, `max_tokens`, splits, the
   frozen seed triples, and both sweep grids.
3. `python -m main extract`, then `sweep` / `select`.

For the `src/` path instead: `configs/datasets/<ds>.yaml` plus one thin config per stage.

## Tests

```bash
pytest tests/ -q                                   # CPU, no data or weights needed
python scripts/check_extract_parity.py             # main/ vs the reference extractions (GPU)
```

`tests/test_main.py` pins `main/` to `src/` where they must agree — the seed→partition
mapping (also frozen as a golden literal), the metric quirks, the base score bit-for-bit,
`build_W`, the context spans and the selection rule — then covers the invariants that
survive `src/` being retired: γ=0 identity, `w="all"` coincidence, vec-vs-reference-loop
for the succ variants, the backprop transpose by hand, batched-vs-loop metrics under ties,
the with-GT context block, and that `main/` never imports `src/`.
