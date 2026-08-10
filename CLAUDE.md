## What this is

Failure attribution in LLM multi-agent systems using a **proxy model's internal
representations**. Given a failed trajectory (e.g. from the Who&When, CORRECT-Error, and
TraceElephant benchmarks), predict which step was the *decisive error* — the earliest
step that irrecoverably derailed the run — and hence which agent is responsible.

The pre-protocol pipeline (reduce `_test`/`_val` conventions, discount/CRR strategy) is
archived VERBATIM at `src_v2/` — never edit it; `src/` is the live pipeline.

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
- `resid` / `angres` / `maha` — the **distance off** the band. Natively "higher = error"
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
protocol. `orient`/`score_norm` remain sweepable axes in code but the protocol FIXES
them (orient=inverse, score_norm=none) and the configs pin the sweep grid to those
values; restoring the full lists in the protocol config re-enables sweeping them.

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
   the recorded window-mean accuracy. → `src/reproduce/`
8. **Analyse** — geometry probe + figure data. → `src/analysis/`

`scripts/run_pipeline.sh` drives 3→6 end to end (`DS=<ds> bash scripts/run_pipeline.sh`).

## Repository layout

- **`src/` = all logic.** Each stage is a package with its algorithm plus a thin
  `run.py` exposing the same CLI.
- **`configs/` = what gets run.** `configs/datasets/<ds>.yaml` is the per-dataset
  manifest and single source of truth (models, model_paths, subsets, max_tokens, splits,
  seeds, ks). `configs/score/<ds>.yaml` drives scoring; `configs/protocol/<ds>.yaml` is
  ONE config driving select + rescore (+ windows + sweep grid); `configs/reproduce/`
  drives reproduction.
- **`data/<ds>/<subset>/*.json` = the corpus.** Input, never regenerated by a run.
- **`outputs/<ds>/` = everything a run produces.** GT-mode runs mirror into
  `outputs-gt/<ds>` purely via `--set outputs_base=...`; extraction takes `--gt`.
- **`src_v2/`** = frozen archive of the pre-protocol pipeline. **`exp-august/`,
  `exp-soap/`** = the experiment dirs whose frozen sweeps/selections are the regression
  oracle the current `src/` was validated against.

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
  the FULL spectrum since `maha` needs the tail; plus `score_config`, which doubles as
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
- **`src/reproduce/`** — `core.py` re-runs one frozen row and exposes per-step signal
  at every stage (`base → oriented → normalized → final`); `run.py` drives it from
  `triples_selection.tsv` and asserts the reproduced window mean equals the recorded
  accuracy (to its 4-decimal rounding).
- **`src/xfit/`** — cross-dataset generalization strand, still defined against the
  RETIRED reduce conventions: everything it needs from them (reduce_base/reduce_crr,
  legacy pooling/seed selection, the discount/CRR strategy, the list-valued WCache)
  lives in `src/xfit/legacy.py`, copied from `src_v2/` unchanged. Only xfit (and the
  optional legacy-parity test) may import it.

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
  metric — changing them silently changes every number.
- **Test-selected by design.** The protocol selects per window on test metrics; the
  score files still carry val columns if a leak-free convention is ever needed.
- **`maha` omits the global 1/(n−1)** — it cancels under per-trajectory ranking.
- **Legacy artifacts on disk.** `outputs/<ds>/reduced/` and `sweep_v2.tsv` files are
  the archived pre-protocol runs (xfit's verify gate reads the reduced trees) — do not
  delete or regenerate them.

## Testing

`pytest tests/test_invariants.py` runs standalone on CPU and is the correctness story:
scorer identities, `γ=0` identity for all three strategies, `orient=none` identity, the
backprop transpose by hand, vec-vs-reference-loop for the succ variants, `w="all"`
coincidence of all strategies, sink pass-through, strongest≠nearest divergence,
column-mask correctness, batched-vs-loop metrics on tie-saturated inputs, and the
with-GT context block (pinned GT prefix, truncation, `build_W` sentinel drop).
`tests/test_xfit_paper.py` covers the xfit strand. `tests/test_parity.py` is an
OPTIONAL migration check against an external legacy tree (`LEGACY_REPO=<path>`), not
part of the repo's own guarantees.
