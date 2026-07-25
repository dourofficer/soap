# Running the SVD + CRR pipeline

Four wrapper scripts drive the whole pipeline. They all `cd` to the repo root and
invoke the stage drivers via `python -m`, so **run them from anywhere** — no need
to `cd` first.

```
scripts/
  gen_embeddings.sh     1. extract per-step representations (activations)   [GPU]
  extract_attention.sh  2. extract per-step attention mass into predecessors [GPU]
  run_analysis.sh       3. svd → undiscounted tables → rescore → discounted tables
  reproduce.sh          4. re-run the pipeline for the best config (validate / apply) [GPU]
  _common.sh            shared knobs + helpers (sourced, not run directly)

  extraction/           per-model fan-outs of steps 1–2 over all datasets
  scoring/              per-model fan-outs of step 3 over all datasets
```

Typical order: `gen_embeddings.sh` and `extract_attention.sh` first (they can run
in parallel on different GPUs), then `run_analysis.sh`, then optionally
`reproduce.sh`.

## Shared knobs (environment variables)

Every script reads these from the environment (see `scripts/_common.sh`):

| Var         | Default          | Meaning                                                        |
|-------------|------------------|----------------------------------------------------------------|
| `DATASET`   | `correct-error`  | Which dataset/config family to run (`correct-error`, `traceelephant`, `ww`). |
| `MODELS`    | *(config's list)*| Space-separated model shorthands to restrict to, e.g. `"qwen3.5-9b"`. |
| `GPU`       | `0`              | `CUDA_VISIBLE_DEVICES` for the GPU stages.                      |
| `DRY_RUN`   | `0`              | `1` → print the commands, don't execute.                       |
| `EXTRA_SET` | *(empty)*        | Extra `--set key=value` overrides forwarded to the stage driver. |

`DATASET` selects the per-stage config `experiments/<stage>/configs/$DATASET.yaml`,
which in turn pulls shared knobs (models, subsets, split ratios, output roots) from
the dataset manifest `experiments/datasets/$DATASET.yaml`. **To onboard a new
dataset you only write one manifest + thin stage configs** — see `_common.sh`.

## 1. Generate embeddings (activations)

```bash
# Default: DATASET=correct-error, all models/subsets from the manifest, GPU 0
./scripts/gen_embeddings.sh

# Pick a dataset + model + GPU
DATASET=traceelephant MODELS=qwen3.5-9b GPU=1 ./scripts/gen_embeddings.sh

# Preview the commands without running
DRY_RUN=1 ./scripts/gen_embeddings.sh

# Narrow to one subset / change max_tokens via EXTRA_SET
EXTRA_SET="--set subsets=[gaia] --set max_tokens=4096" ./scripts/gen_embeddings.sh
```

Writes `outputs-<dataset>/activations/<model>/<subset>/*.safetensors`.

## 2. Extract attention

```bash
DATASET=correct-error GPU=0 ./scripts/extract_attention.sh

DRY_RUN=1 ./scripts/extract_attention.sh
```

Writes `outputs-<dataset>/attention/<model>/<subset>/*.safetensors`.
`max_tokens` **must match** the activations run (both come from the manifest, so
they already agree).

## 3. Run the analysis chain

Runs, in order: `svd` → `undisc` (undiscounted tables) → `rescore` (CRR sweep) →
`disc` (discounted tables). Assumes activations + attention already exist.

```bash
# Full chain
DATASET=correct-error GPU=0 ./scripts/run_analysis.sh

# Re-run only part of the chain (any subset/order of the stage names)
STAGES="rescore disc" ./scripts/run_analysis.sh

# Preview
DRY_RUN=1 ./scripts/run_analysis.sh

# Override sweep axes for the GPU stages (svd + rescore)
EXTRA_SET="--set poolings=[last] --set seeds=[1,2,3]" ./scripts/run_analysis.sh
```

Outputs (under `outputs-<dataset>/`, split-tagged, e.g. `325` = 0.3/0.2/0.5):
- `weighted-projections/<tag>/…`      base SVD scores
- `undiscounted-splits/<tag>/…`       best base configs
- `discounted-splits/sweep/<tag>/…`   full CRR sweep
- `discounted-splits/reduced/<tag>/…` best CRR config per (pooling, seed)

`STAGES` also accepts an optional 5th stage, `reproduce` (off by default):

```bash
STAGES="svd undisc rescore disc reproduce" ./scripts/run_analysis.sh
```

## 4. Reproduce / apply the best config

Reads the reduced discounted table (best config per pooling/seed) and re-runs the
**full** SVD → orient → CRR-discount pipeline. Two modes via `split`:

- `split=test` / `split=val` — **validate**: recompute step@1 / agent@1 and check
  they match the reduced table.
- `split=all` — **apply**: score every trajectory in the subset and emit the
  predicted decisive step per trajectory.

```bash
# Validate the best configs against the table (default: split=test)
DATASET=correct-error GPU=0 ./scripts/reproduce.sh

# Apply the frozen config to ALL trajectories
EXTRA_SET="--set split=all" ./scripts/reproduce.sh

# Only the top-K seeds, or a single explicit config
EXTRA_SET="--set split=all --set top_k=1" ./scripts/reproduce.sh
EXTRA_SET="--set select=explicit --set explicit.pooling=last --set explicit.seed=1 \
           --set explicit.position=act/3 --set explicit.c_begin=1 --set explicit.c_end=9 \
           --set explicit.centered=false --set explicit.svd_orient=inverse \
           --set explicit.layer_range=2-4 --set explicit.gamma=0.4 --set explicit.w=1" \
  ./scripts/reproduce.sh
```

Writes `outputs-<dataset>/reproductions/<tag>/<split>/<model>/<subset>/`:
`predictions.tsv`, `step_scores.tsv`, `metrics.tsv`.

## End-to-end example (correct-error, one GPU)

```bash
DATASET=correct-error GPU=0 ./scripts/gen_embeddings.sh
DATASET=correct-error GPU=0 ./scripts/extract_attention.sh
DATASET=correct-error GPU=0 ./scripts/run_analysis.sh
DATASET=correct-error GPU=0 ./scripts/reproduce.sh          # validate
```

## Fanning out across datasets × models

`scripts/run_analysis.sh` takes one `DATASET` and one GPU per invocation.
`scripts/scoring/` wraps it to loop over models × datasets:

```
scripts/scoring/
  run_scoring.sh            generic MODELS × DATASETS loop (configure both via env)
  deepseek-8b_analysis.sh   thin wrapper, model pinned
  qwen3.5-9b_analysis.sh    thin wrapper, model pinned
```

`run_scoring.sh` reads two extra knobs on top of the shared ones, plus `STAGES`:

| Var        | Default                       | Meaning                                     |
|------------|-------------------------------|---------------------------------------------|
| `MODELS`   | `qwen3.5-9b deepseek-8b`      | Space-separated shorthands; looped one at a time. |
| `DATASETS` | `ww traceelephant correct-error` | Space-separated datasets; looped inner.     |

```bash
# Everything: both models × all 3 datasets, sequential on GPU 0
./scripts/scoring/run_scoring.sh

# Pick models and datasets explicitly
MODELS="deepseek-8b" DATASETS="ww correct-error" GPU=1 ./scripts/scoring/run_scoring.sh

# Preview, or re-run only part of the chain
DRY_RUN=1 ./scripts/scoring/run_scoring.sh
STAGES="rescore disc" ./scripts/scoring/run_scoring.sh
```

Both report builders write per `(model, subset)` — `out_root/<model>/<subset>/` —
so looping models produces the same files as one combined run; nothing is
overwritten. The loop **fails fast**: a failing `(model, dataset)` aborts the rest.

## End-to-end all datasets (two GPUs)

Extraction first (steps 1–2), then scoring (step 3). One model per GPU, so the two
columns can run in parallel:

```bash
GPU=2 bash ./scripts/extraction/deepseek-8b_activations.sh
GPU=2 bash ./scripts/extraction/deepseek-8b_attention.sh
GPU=2 bash ./scripts/scoring/deepseek-8b_analysis.sh

GPU=3 bash ./scripts/extraction/qwen3.5-9b_activations.sh
GPU=3 bash ./scripts/extraction/qwen3.5-9b_attention.sh
GPU=3 bash ./scripts/scoring/qwen3.5-9b_analysis.sh
```

## Tips

- **Preview first.** Prefix any command with `DRY_RUN=1` to see exactly what will run.
- **Idempotent.** Stages skip work whose output already exists, so re-running is safe.
- **Config override precedence** (low → high): dataset manifest < stage config <
  `EXTRA_SET`/`--set`. Use `--set key.subkey=value` for nested keys.
- **New dataset?** Add `experiments/datasets/<ds>.yaml` (+ thin stage configs), then
  run every script with `DATASET=<ds>`. See the note at the top of `_common.sh`.
