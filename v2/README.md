# attribscope

Failure attribution in LLM multi-agent systems from a **proxy model's internals**. Given
a failed trajectory, predict the **decisive-error step** — the earliest step that
irrecoverably derailed the run — and hence the responsible agent.

Each step's hidden state gets a **base score** measuring how far it sits from the
"common mode" of typical steps (SVD geometry). A **rescoring strategy** then corrects
that score using the causal dependency structure the model itself reveals through
attention, so a late symptom is not mistaken for the cause. Evaluation is step@k /
agent@k, ranked within each trajectory.

See `CLAUDE.md` for the method in detail and the conventions that bite.

## Layout

```
data/<ds>/<subset>/*.json         the corpus (input; never written by a run)
src/
  common/     config merge + --set, derived paths, uniform CLI, run provenance
  data/       trajectory representation + per-step context construction
  extract/    activations.py (pooled hidden states), attention.py (attention mass)
  stores.py   representation loading + train/val/test splits
  metrics.py  step@k / agent@k: reference loop + vectorized batch
  score/      scorers registry, SVD fit + grid, layer ensemble, runner
  rescore/    attention weights -> W, strategies (discount / backprop), runner
  reports/    reduce (best config per seed, both conventions), results table
  reproduce/  re-run ONE frozen config -> per-step scores for inspection
  analysis/   geometry probe (what the singular vectors actually represent)
configs/      datasets/<ds>.yaml (manifest) + <stage>/<ds>.yaml (thin, per stage)
scripts/      extract.sh, run_pipeline.sh, copy_inputs.sh
tests/        test_invariants.py (CPU, standalone) + test_parity.py (optional)
outputs/<ds>/ everything a run produces (activations, attention, scores, tables, ...)
```

## Quickstart

```bash
cd v2/
# 1. corpus in data/<ds>/<subset>/*.json, then extract representations + attention
DATASET=correct-full GPU=0 ./scripts/extract.sh
# 2. score -> reduce -> rescore -> reduce -> tables
DATASET=correct-full GPU=0 ./scripts/run_pipeline.sh
cat outputs/correct-full/tables/325/results.tsv
```

Adopting extractions produced elsewhere instead of running step 1:
`SRC=/path/to/tree DATASET=correct-full ./scripts/copy_inputs.sh`.

Narrow or override anything: `MODEL=qwen3.5-9b SEED=1 EXTRA_SET="--set gammas=[0.5]"
./scripts/run_pipeline.sh`.

## Stages

Every stage takes the same CLI (`--config`, `--set k=v`, `--model/--subset/--seed`,
`--device`, `--dry-run`, `--force`) and is run from `v2/`:

| stage | command | output |
|---|---|---|
| extract reps | `python -m src.extract.activations --model M --model-path P --input data/<ds> --subset S --output outputs/<ds>/activations/M/S` | `.safetensors` per trajectory |
| extract attn | `python -m src.extract.attention --model M --model-path P --input data/<ds> --subset S --output-root outputs/<ds>/attention` | `.safetensors` per trajectory |
| score | `python -m src.score.run --config configs/score/<ds>.yaml` | `scores/<tag>/<model>/<subset>/seed-<n>.tsv` |
| reduce (base) | `python -m src.reports.reduce --config configs/reduce/<ds>.yaml --set stage=base` | `reduced/.../base_{test,val}.tsv` |
| rescore | `python -m src.rescore.run --config configs/rescore/<ds>.yaml` | `rescore/.../sweep.tsv` |
| reduce (crr) | `python -m src.reports.reduce --config configs/reduce/<ds>.yaml --set stage=crr` | `reduced/.../{crr,backprop}_{test,val}.tsv` |
| tables | `python -m src.reports.tables --config configs/tables/<ds>.yaml` | `tables/<tag>/results.tsv` |
| reproduce | `python -m src.reproduce.run --config configs/reproduce/<ds>.yaml` | `reproductions/.../*.steps.tsv` |
| geometry | `python -m src.analysis.geometry --config configs/datasets/<ds>.yaml` | `analysis/<tag>/geometry/...` |

## Inspecting a config (not just scoring it)

The sweep collapses each run to one number. To see what a config *did*, reproduce it —
this re-runs one frozen winner and writes the score at every pipeline stage, per step:

```bash
python -m src.reproduce.run --config configs/reproduce/correct-full.yaml \
    --set tables=[base_test,crr_test,backprop_test] --set split=test
```

`*.steps.tsv` has one row per (trajectory, step) with `base / oriented / normalized /
final`, the within-trajectory `rank`, `is_pred` and the gold `is_mistake` flag — plot a
trajectory's score curve directly, or concatenate several tables (each tagged with a
`table` column) to diff methods on identical rows. `*.preds.tsv` gives one row per
trajectory including `true_step_rank`, which is what you want for error analysis.
Reproduction asserts it recovers the recorded accuracy exactly.

## Adding a dataset

1. `data/<ds>/<subset>/*.json` — each with `history` (ordered turns; step `t ==
   history[t]`, 0-indexed), `question_ID`, `mistake_agent`, `mistake_step`, and
   optionally `level` / `question`.
2. `configs/datasets/<ds>.yaml` — models, `model_paths`, subsets, `max_tokens`, splits,
   seeds, ks.
3. One thin config per stage (`configs/<stage>/<ds>.yaml`) — usually a copy with
   `dataset:` swapped.
4. Extract, then run the pipeline.

## Tests

```bash
pytest tests/test_invariants.py -q          # CPU, no data needed — the correctness story
LEGACY_REPO=/path pytest tests/test_parity.py -q   # optional migration check
```

Invariants cover the scorer identities, `γ=0` identity for both strategies, the backprop
transpose, vectorized-vs-reference rescoring, and batched-vs-loop metrics under ties.
