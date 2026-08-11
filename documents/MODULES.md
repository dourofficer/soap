# Core modules

A map of who does what, `src/` first, then `main/`. Read the entry for a module before
you change it; each one records the behaviour that is not obvious from its code.

## `src/` — the full sweep

- **`src/common/`** keeps orchestration DRY across every stage: `config.py` (the one
  `load_stage_config` + manifest merge), `paths.py` (derives every stage's root from
  `(dataset, split-tag)` — never hand-write `outputs/...`), `cli.py` (the uniform
  runner CLI), `provenance.py` (a JSON run record per invocation).

- **`src/data/`** — how a trajectory is represented and how a step's context is built.
  `context.py` assembles `input_ids` from independently-tokenised pieces so **every
  token belongs to exactly one step**; deriving spans from re-rendered template
  prefixes is off-by-a-scaffold and breaks both pooling and attention attribution.
  With-GT mode pins a `[question, answer]` block under the `GT_STEP = -1` sentinel
  (never scored, dropped by `build_W`).

- **`src/extract/`** — `activations.py` (one forward pass per step in context; pool
  hidden states to one vector per step) and `attention.py` (per-step attention into
  predecessors, without ever materialising an (N,N) matrix).

- **`src/stores.py`** — loads per-trajectory `.safetensors` into `(pooling, name)`-keyed
  stores sharing one `StoreKeeper`, and owns `split_files` (splits are by TRAJECTORY;
  the seed→partition mapping is the experiment's identity, so it must not drift).

- **`src/metrics.py`** — `compute_metrics` (reference loop) and `compute_metrics_batch`
  (vectorized). The batch version ranks via the closed-form identity
  `rank(i) = 1 + #{s_j > s_i} + #{j<i : s_j == s_i}`, which reproduces the stable
  sort's earliest-step tie-break exactly — `torch.topk` is not tie-stable and must not
  be used.

- **`src/score/`** — `scorers.py` (registry + `METHOD_DIRECTION`), `svd.py` (fit
  keeping the FULL spectrum since weighted `proj` needs it; plus `score_config`, which
  doubles as the reproduction primitive), `ensemble.py` (`ens-mid3`), `run.py`. Every
  scorer is documented in [SCORERS.md](SCORERS.md).

- **`src/rescore/`** — `weights.py` (attention → per-strategy dense `W` sets: `build_W`
  row-trims for backprop, `mask_columns_*` column-mask the full W for the succ
  variants; `strategy_mats`/`WCache` bundle them), `strategies.py` (orient / score_norm
  / the shared `backprop_vec` + `STRATEGIES` dispatch, plus the `backprop_succ_loop`
  reference the vectorized path is tested against), `run.py` (builds the union base
  table from the window selections, then sweeps).

- **`src/reports/`** — `baselines.py` (display maps, recorded-score loader, baseline
  prediction scoring), `triples.py` (the window protocol; two-pass idempotent — see
  [SELECTION.md](SELECTION.md)), `manuscript.py` (manuscript-shaped tables: prompting
  rows + SOAP per GT setting, every cell's rows evaluated on that cell's
  `pick_shared_window` seed triple). Prompting predictions live at
  `outputs[-gt]/<ds>/baselines/prompting/<judge>/<subset>/` (judges gpt-4o/gpt-5
  imported from `../attrib-prompting` by `scripts/import_prompting.py`; backbone-named
  dirs are the older local-model runs).

- **`src/analysis/`** — geometry probe + figure data. `qualitative.py` imports
  `main.reproduce` (reproduction lives in `main/`). `main/` must never import `src/`;
  the reverse is fine.

## `main/` — the simplified runner

Self-contained, ~2.2k lines against `src/`'s ~5.4k. `cli.py`/`__main__.py` (the four
commands — see `main/README.md`), `config.py` (paths, frozen seeds, the
`run_stamp.json` drift guard), `data.py` (trajectories + token spans), `extract.py` (a
FAITHFUL port of `src/extract` — same keys, both poolings, `raw_attn_per_head` kept, so
its artifacts are byte-compatible with `outputs/`), `stores.py`, `metrics.py` (no
`direction` axis; keys are `step@k`, not `step@k_desc`), `score.py`, `rescore.py`,
`sweep.py`, `reproduce.py`. Its base score folds the inverse in — `S = 1/(pi+eps)` — so
ranking is always descending and there is no orient/score_norm/centered/weighted
plumbing at all.

## Scripts that read them

- `scripts/main/` — sweeps `main/` over many seed triples and picks the reported ones
  (`scripts/main/README.md`).
- `scripts/prompting/` — scores the prompting baselines on SOAP's exact test splits;
  `verify.py` guards its four silent failure modes.
- `scripts/tables/` — `make_main_tables.py` builds the manuscript tables from
  `results-sweep/selections_all.tsv` + `results-prompting/by_column.tsv`;
  `sync_seeds.py` owns the seed blocks in `configs-main/`.
- Parity checkers: see [TESTING.md](TESTING.md).
