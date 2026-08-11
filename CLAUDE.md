## What this is

Failure attribution in LLM multi-agent systems using a **proxy model's internal
representations**. Given a failed trajectory (e.g. from the Who&When, CORRECT-Error, and
TraceElephant benchmarks), predict which step was the *decisive error* — the earliest
step that irrecoverably derailed the run — and hence which agent is responsible.

Two packages implement it. **`src/` is the full sweep**: every scorer, orientation,
normalization and centering arm stays implemented so any axis can be swept again by
editing a config, even though the production configs pin all of them. **`main/` is the
simplified, self-contained runner** with those axes frozen in code and seeds hardcoded —
it is the primary entry point for new runs and owns reproduction. `main/` imports
nothing from `src/`.

## The method

Prediction is the argmax over rescored per-step scores: `t̂ = argmax_t S̃(s_t)`.

A **base score** `S(s_t)` rates each step's anomaly from the proxy model's hidden state.
A **rescoring strategy** then corrects it using the causal dependency structure the
model itself reveals through attention.

**Base score.** Fit an SVD to the TRAIN split's step matrix; the top right-singular
vectors span the "common mode" of what a typical step looks like (the top uncentered
singular vector *is* the mean direction — `cos(u₀, mean) = 1.00` — and carries 65-90% of
all energy). Decisive-error steps sit slightly off that common mode. Two dual readings,
both supported:
- `proj` — mean squared projection **onto** the band: `(1/|C|) Σ_{c∈C} ⟨ṽ_t, u_c⟩²`.
  Error steps project *smaller*, so this is "lower = error" (`asc`) and needs orienting.
- `resid` / `angres` — the **distance off** the band. Natively "higher = error"
  (`desc`), so no orientation is needed at all. `angres` (`sin²θ = r²/‖ṽ‖²`) is the
  principled member: norm-free and bounded.

`norm_l1`/`norm_l2` are baselines, evaluated in both directions.

**Dependency weights.** `w_{i,t}` is the fraction of step-`t` query attention landing in
predecessor step `i`: head-averaged over a layer band, then renormalized over
predecessors so `Σ_{i<t} w_{i,t} = 1`.

**Rescoring strategies.** All three are the SAME single-pass, hub-normalized
blame-collection (they read the ORIGINAL `S`, so corrections never cascade):

    S̃(s_i) = S(s_i) + γ·(Σ_t M[t,i]·S(s_t)) / (Σ_t M[t,i])      = s + γ(Mᵀs)/(Mᵀ1)

They differ only in which matrix `M` they aggregate through — i.e. where the top-w
sparsification lives (`src/rescore/weights.strategy_mats`):

- `backprop` (SOAP) — predecessor-side: each step t keeps its w strongest predecessors
  (rows trimmed + renormalized by `build_W`). Selective: a step is lifted only if some
  successor ranked it top-w; steps that win no slot pass through. Empirically strongest.
- `succ-strong` — successor-side: W stays full; each step i collects from its w
  strongest successors (columns masked by weight). Lifts nearly every step at any w.
- `succ-near` — successor-side: step i collects from its w NEAREST scored successors.

At `w="all"` the three coincide exactly. `γ=0` recovers the base scorer. The hub
normalization (dividing by attention *received*) prevents merely-popular steps — plans,
restatements — from accruing blame.

Base scores rank **within a trajectory** — never across. Evaluation reports step@k /
agent@k.

## The selection protocol (seed windows / "triples")

THE selection convention of the pipeline, replacing the old reduce conventions. For
every consecutive window of `triples.window` seeds over `triples.seeds` (declared in
`configs/protocol/<ds>.yaml`), per (model, subset):

1. **SVD (proj)**: filter recorded score rows by `base.fixed`; pick the `base.swept`
   config (position, c_begin, c_end) maximizing mean test step-acc over the window's
   seeds (config must exist in all of them; tiebreak agent acc, then highest key).
2. **Rescoring rows** (one per strategy): filter the sweep by strategy +
   `rescore.fixed` + the window's chosen base config; pick `rescore.swept`
   (layer_range, gamma, w) the same way.
3. **Baselines** (prompting / CHIEF / CORRECT): scored from recorded prediction JSONLs
   on the window's seeds.

Selection is test-selected over the window — optimistic by construction; that is the
protocol.

**The reported seed triple is shared across backbones.** Base scoring and rescoring run
on all 18 windows; then, per SUBSET, `src/reports/manuscript.py:pick_shared_window` takes
the window maximizing the SUM of the strategy's step accuracy across backbones (tiebreak:
agent-accuracy sum, then earliest window). Every backbone and every prompting judge in
that column reports on that one triple, so a manuscript column compares like with like.
Hyperparameters stay per (model, subset, window); only the split is shared.

**Fixed axes are pinned by CONFIG, never deleted.** `orient`/`score_norm`/`weighted`/
`centered`/`pooling`/`method` all remain implemented and sweepable in `src/`; the shipped
configs pin them (orient=inverse, score_norm=none, weighted=false, centered=false,
pooling=mean, method=proj). Restoring the fuller lists in a config re-enables the axis —
that is the whole reason the code stays. Code deletion for these axes happens only in
`main/`.

## The pipeline (and where each stage lives)

1. **Per-step representations** — one forward pass per step in context; pool hidden
   states to one vector `v_t`. → `src/extract/activations.py`
2. **Attention mass** — per-step attention into predecessors, without ever materialising
   an (N,N) matrix. → `src/extract/attention.py`
3. **Base score** — fit SVD on train, score the config grid per seed. → `src/score/`
4. **Select (pass 1)** — per-window SVD config + baseline rows. → `src/reports/triples.py`
5. **Rescoring sweep** — union base table over windows, then orient → normalize →
   strategy, all γ at once. → `src/rescore/`
6. **Select (pass 2)** — same command as 4; now fills the rescoring rows and the
   side-by-side summary. → `src/reports/triples.py`
7. **Reproduce** — re-run selected rows, emit per-step scores/ranks/predictions, assert
   the recorded accuracy. → `main/reproduce.py` (no longer in `src/`)
8. **Analyse** — geometry probe + figure data. → `src/analysis/`

`scripts/run_pipeline.sh` drives 3→6 end to end (`DS=<ds> bash scripts/run_pipeline.sh`).

## Repository layout

- **`src/` = the full sweep machinery.** Each stage is a package with its algorithm
  plus a thin `run.py` exposing the same CLI.
- **`main/` = the simplified runner** (~2.2k lines vs `src/`'s ~5.4k). Self-contained,
  fixed axes removed from the code, seeds hardcoded, one `sweep` command. Owns
  reproduction. Reads `configs-main/<ds>.yaml`, writes `results-nogt/` / `results-gt/`.
- **`configs/` = what gets run.** `configs/datasets/<ds>.yaml` is the per-dataset
  manifest and single source of truth (models, model_paths, subsets, max_tokens, splits,
  seeds, ks). `configs/score/<ds>.yaml` drives scoring; `configs/protocol/<ds>.yaml` is
  ONE config driving select + rescore (+ windows + sweep grid). `configs-main/<ds>.yaml`
  is `main/`'s single per-dataset config.
- **`data/<ds>/<subset>/*.json` = the corpus.** Input, never regenerated by a run.
- **`outputs/<ds>/` = everything a `src/` run produces.** GT-mode runs mirror into
  `outputs-gt/<ds>` purely via `--set outputs_base=...`; extraction takes `--gt`.
  These two trees are FROZEN as the reference: `main/` never writes into them.
- **`results-nogt/<ds>/`, `results-gt/<ds>/` = what `main/` produces.** Their
  `activations/` and `attention/` are seeded from `outputs[-gt]/` by
  `scripts/seed_results.sh` (hardlink copy — the artifacts are byte-compatible, so this
  is the same file, not a conversion).
- Legacy artifacts under `outputs/<ds>/reduced/<tag>/crr_*.tsv`, `outputs/*/xfit-*` and
  `results_synthfit*.tsv` are pre-protocol runs with no remaining consumer. Left in
  place; do not regenerate. (`reduced_root` itself is still live — the rescore stage
  writes `base_triples.tsv` there.)

Three pieces keep orchestration DRY, all in `src/common/`: `config.py` (the one
`load_stage_config` + manifest merge), `paths.py` (derives every stage's root from
`(dataset, split-tag)` — never hand-write `outputs/...`), `cli.py` (the uniform runner
CLI), `provenance.py` (a JSON run record per invocation).

## Running

Everything runs **from the repo root**, with one CLI shared by every stage:

```
python -m src.<stage>.run --config configs/<stage>/<ds>.yaml \
    [--set key.subkey=value ...]        # dot-path YAML overrides
    [--model M] [--subset S] [--seed N] # narrowing
    [--device cuda|cpu] [--dry-run] [--force]
```

(`src.reports.triples` is the select stage; it and `src.rescore.run` both take the
`configs/protocol/<ds>.yaml`.) Precedence: manifest < stage config < `--set` <
narrowing flags. Every runner skips cells whose output exists (`--force` overrides)
and writes a provenance record. Onboarding a dataset = one manifest + one thin config
per stage.

## Core modules

- **`src/data/`** — how a trajectory is represented and how a step's context is built.
  `context.py` assembles `input_ids` from independently-tokenised pieces so **every token
  belongs to exactly one step**; deriving spans from re-rendered template prefixes is
  off-by-a-scaffold and breaks both pooling and attention attribution. With-GT mode pins
  a `[question, answer]` block under the `GT_STEP = -1` sentinel (never scored, dropped
  by `build_W`).
- **`src/stores.py`** — loads per-trajectory `.safetensors` into `(pooling, name)`-keyed
  stores sharing one `StoreKeeper`, and owns `split_files` (splits are by TRAJECTORY;
  the seed→partition mapping is the experiment's identity, so it must not drift).
- **`src/metrics.py`** — `compute_metrics` (reference loop) and `compute_metrics_batch`
  (vectorized). The batch version ranks via the closed-form identity
  `rank(i) = 1 + #{s_j > s_i} + #{j<i : s_j == s_i}`, which reproduces the stable sort's
  earliest-step tie-break exactly — `torch.topk` is not tie-stable and must not be used.
- **`src/score/`** — `scorers.py` (registry + `METHOD_DIRECTION`), `svd.py` (fit keeping
  the FULL spectrum since weighted `proj` needs it; plus `score_config`, which doubles as
  the reproduction primitive), `ensemble.py` (`ens-mid3`), `run.py`.
- **`src/rescore/`** — `weights.py` (attention → per-strategy dense `W` sets:
  `build_W` row-trims for backprop, `mask_columns_*` column-mask the full W for the
  succ variants; `strategy_mats`/`WCache` bundle them), `strategies.py` (orient /
  score_norm / the shared `backprop_vec` + `STRATEGIES` dispatch, plus the
  `backprop_succ_loop` reference the vectorized path is tested against), `run.py`
  (builds the union base table from the window selections, then sweeps).
- **`src/reports/`** — `baselines.py` (display maps, recorded-score loader, baseline
  prediction scoring), `triples.py` (the protocol; two-pass idempotent),
  `manuscript.py` (manuscript-shaped tables: prompting rows + SOAP per GT setting,
  every cell's rows evaluated on that cell's pick_window seed triple). Prompting
  predictions live at `outputs[-gt]/<ds>/baselines/prompting/<judge>/<subset>/`
  (judges gpt-4o/gpt-5 imported from `../attrib-prompting` by
  `scripts/import_prompting.py`; backbone-named dirs are the older local-model runs).
- **`main/`** — the simplified runner, self-contained. `config.py` (paths, frozen
  seeds, the `run_stamp.json` drift guard), `data.py` (trajectories + token spans),
  `extract.py` (a FAITHFUL port of `src/extract` — same keys, both poolings,
  `raw_attn_per_head` kept, so its artifacts are byte-compatible with `outputs/`),
  `stores.py`, `metrics.py` (no `direction` axis; keys are `step@k`, not `step@k_desc`),
  `score.py`, `rescore.py`, `sweep.py`, `reproduce.py`. Its base score folds the inverse
  in — `S = 1/(pi+eps)` — so ranking is always descending and there is no orient/
  score_norm/centered/weighted plumbing at all.
- **`src/analysis/qualitative.py`** imports `main.reproduce` (reproduction lives in
  `main/`). `main/` must never import `src/`; the reverse is fine.

## Conventions that bite

- **Two sign conventions.** `proj` is "lower = error"; the rescoring math assumes
  "higher = error". Route `proj` through `orient` first; distance scorers need nothing.
  `METHOD_DIRECTION` is the source of truth, and `allowed_orients` auto-restricts
  native-desc methods to `orient=none`.
- **`sigmoid` orientation saturates.** On large-magnitude scores `sigmoid(-s) ≈ 0` for
  every step, collapsing the ranking to a tie. The undiscounted reference is therefore
  taken from the base row's own metric, never recomputed from an oriented score.
- **Float before scoring.** Scorers cast back to `R.dtype`; passing fp16 `R` rounds the
  scores and flips near-ties. `score_config` floats `R` first — do not "optimise" this.
- **backprop's slot-consumption order is semantics.** `build_W` applies top-w selection
  BEFORE dropping unscored buckets (turn-0 human, GT block), so those can claim a slot
  and then vanish. Deriving the row trim from the already-filtered full W would
  silently change every backprop number. The succ masks, by contrast, are DEFINED on
  the filtered full W.
- **Layer indexing differs by stage.** Activations store `embed` (tuple index 0),
  `act/k` (tuple index k+1), and `act/{N-1}_normed`. Attention stores one row per
  attention block, so `layer_range` labels index ATTENTION blocks (8 for Qwen3.5, which
  is hybrid and only exposes full-attention layers; 32 for DeepSeek), not positions.
- **Split ratios have one home.** They live in the manifest and derive the split-tag
  (`0.3/0.2/0.5 → "325"`) naming every split-tagged root. Extraction is NOT tagged, so
  overriding `split_tag` reruns analysis while reusing the expensive forward passes.
- **Protocol seeds ≠ manifest seeds.** The window universe is `triples.seeds` in the
  protocol config; the manifest's `seeds` list may be narrower (it caps score-stage
  sweeps). `triples.py` overrides the seed restriction when loading scores — keep it.
- **Metric quirks are intentional.** `agent@k` lowercases the gold role but
  `standardize_role()`s the candidates; trajectories with no gold mistake are skipped yet
  still counted in the divisor; ties resolve to the earliest step. These define the
  metric — changing them silently changes every number. `main/` reproduces all of them;
  only the key naming differs (`step@1`, since there is no direction axis).
- **Test-selected by design.** The protocol selects per window on test metrics; the
  score files still carry val columns if a leak-free convention is ever needed.
- **The selection tiebreak is float-fragile in `src/`.** Accuracies are rationals with
  small denominators, so two configs that tie mathematically can land 1-2 ulps apart once
  averaged over seeds — the addends differ even though their sum does not (26+24+27 vs
  25+24+28, over 63). `src/`'s `select_shared` compares raw floats, so SUMMATION ORDER —
  i.e. row order in `sweep.tsv` — can decide a tie before the documented agent tiebreak is
  consulted. `main.sweep.select_config` and `manuscript.pick_shared_window` round the
  comparison key to 12 dp, which restores the intended rule. Two ww cells differ between
  the packages for exactly this reason; the step accuracy is identical either way and
  `main/`'s pick has the higher agent accuracy. Pinned by
  `test_selection_tiebreak_survives_float_noise`.
- **Legacy artifacts on disk.** `outputs/<ds>/reduced/crr_*.tsv`, `sweep_v2.tsv`,
  `outputs/*/xfit-*` and `results_synthfit*.tsv` are archived pre-protocol runs whose
  only consumer (the `xfit` strand) was deleted in the 2026-08 cleanup. They now have no
  reader — left in place as a record; do not regenerate. `reduced_root` itself is still
  live: the rescore stage writes `base_triples.tsv` there.
- **`outputs/` and `outputs-gt/` are frozen reference.** `main/` writes to
  `results-nogt/` / `results-gt/` instead, seeded from them by `scripts/seed_results.sh`.
  The two extractors produce byte-compatible artifacts, so that seeding is a hardlink
  copy, not a conversion (`scripts/check_extract_parity.py` proves it).

## `main/` — the simplified runner

Six choices are FROZEN in `main/`'s code, not exposed as config: `pooling=mean` (for
scoring; the extractor still writes both), uncentered SVD, `method=proj`, orientation
folded in as `1/(pi+eps)`, `score_norm=none`, `weighted=false`. What remains swept is
`position x c_begin x c_end` and `layer_range x gamma x w x strategy`. Seeds are a
hardcoded triple per subset in `configs-main/<ds>.yaml`, shared across backbones.

Two structural consequences:

- **No separate base table.** At `gamma=0`, `S~ = s` exactly for every matrix, and
  `1/(pi+eps)` descending is order-identical to `pi` ascending (ties included). So the
  `gamma=0` rows ARE the base score; `sweep.py` asserts that against the `strategy="base"`
  rows on every run. `undisc_*`, `base_triples.tsv` and the two-pass dance are all gone.
- **One `sweep` command**, because the rescore grid is expanded only for the SELECTED base
  config. The full cross product would be ~3.7M rows per seed; staged it is ~24k.

`ens-mid3` is the ONE deliberate numerical divergence from `src/`: `src/` orients ensemble
members by negation before z-scoring (orient was a separate downstream stage that
`ens-mid3` bypassed), `main/` uses the folded inverse. Every single-position number is
bit-identical; only `ens-mid3` differs — and it has been selected in 0 of 4,428 rows.

## Testing

`pytest tests/ -q` runs standalone on CPU. `tests/test_invariants.py` is `src/`'s
correctness story:
scorer identities, `γ=0` identity for all three strategies, `orient=none` identity, the
backprop transpose by hand, vec-vs-reference-loop for the succ variants, `w="all"`
coincidence of all strategies, sink pass-through, strongest≠nearest divergence,
column-mask correctness, batched-vs-loop metrics on tie-saturated inputs, and the
with-GT context block (pinned GT prefix, truncation, `build_W` sentinel drop).
`tests/test_main.py` is `main/`'s: Group A pins it to `src/` where they must agree (the
seed→partition mapping — also frozen as a golden literal — keeper row order, the metric
quirks, the base score bit-for-bit, `fit_svd`, `build_W` including the GT-sentinel slot
consumption, the column masks, the context spans, the selection rule), Group B covers the
invariants that survive `src/` being retired, plus an `ast` walk asserting `main/` never
imports `src/`.

`scripts/check_extract_parity.py` (GPU) proves `main/extract.py` reproduces the reference
extractions bit-for-bit — needed because the bulk tensors are COPIED into the results
trees rather than recomputed.
