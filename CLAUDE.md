# SOAP — Spectral Scoring with Attention-Guided Propagation

## What this is

A multi-agent LLM run fails. Somewhere in its trajectory one step — the **decisive
error** — derailed it beyond recovery. This repo predicts that step, and hence the
responsible agent, from the internal representations of a **proxy model** that merely
reads the trajectory. Benchmarks: Who&When (`ww`), CORRECT-Error (`correct-error`), and
TraceElephant (`traceelephant`).

Two packages implement the method. **`src/` is the full sweep**: every scorer,
orientation, normalization and centering arm stays implemented, so any axis can be
swept again by editing a config, even though the production configs pin all of them.
**`main/` is the simplified runner**: those axes are frozen in code, the seeds are
frozen in config, and it owns reproduction and the reported numbers. `main/` imports
nothing from `src/`; the reverse is fine.

## The method

The prediction is an argmax over rescored per-step scores: `t̂ = argmax_t S̃(s_t)`. A
**base score** `S(s_t)` rates each step's anomaly from the proxy model's hidden state.
A **rescoring strategy** then corrects it using the causal dependency structure the
model itself reveals through attention, so a late symptom is not mistaken for the
cause.

**Base score.** Fit an SVD to the TRAIN split's step matrix. The top right-singular
vectors span the "common mode" of what a typical step looks like: the top uncentered
singular vector *is* the mean direction (`cos(u₀, mean) = 1.00`) and carries 65-90% of
all energy. Decisive-error steps sit slightly off that common mode. Two dual readings,
both supported:

- `proj` — mean squared projection **onto** the band:
  `(1/|C|) Σ_{c∈C} ⟨ṽ_t, u_c⟩²`. Error steps project *smaller*, so this reads
  "lower = error" (`asc`) and needs orienting.
- `resid` / `angres` — the distance **off** the band. Natively "higher = error"
  (`desc`), so it needs no orientation. `angres` (`sin²θ = r²/‖ṽ‖²`) is the principled
  member: norm-free and bounded.

`norm_l1`/`norm_l2` are baselines, evaluated in both directions. Every scorer and its
geometry: `documents/SCORERS.md`.

**Dependency weights.** `w_{i,t}` is the fraction of step-`t` query attention landing
in predecessor step `i`: head-averaged over a layer band, then renormalized over
predecessors so `Σ_{i<t} w_{i,t} = 1`.

**Rescoring strategies.** All three are the SAME single-pass, hub-normalized
blame-collection. They read the ORIGINAL `S`, so corrections never cascade:

    S̃(s_i) = S(s_i) + γ·(Σ_t M[t,i]·S(s_t)) / (Σ_t M[t,i])      = s + γ(Mᵀs)/(Mᵀ1)

They differ only in which matrix `M` they aggregate through — i.e. where the top-w
sparsification lives (`src/rescore/weights.strategy_mats`):

- `backprop` (SOAP) — predecessor-side: each step t keeps its w strongest predecessors
  (rows trimmed + renormalized by `build_W`). Selective: a step is lifted only if some
  successor ranked it top-w; steps that win no slot pass through. Empirically
  strongest.
- `succ-strong` — successor-side: W stays full; each step i collects from its w
  strongest successors (columns masked by weight). Lifts nearly every step at any w.
- `succ-near` — successor-side: step i collects from its w NEAREST scored successors.

At `w="all"` the three coincide exactly. `γ=0` recovers the base scorer. The hub
normalization (dividing by attention *received*) prevents merely-popular steps —
plans, restatements — from accruing blame.

Base scores rank **within a trajectory** — never across. Evaluation reports
step@k / agent@k.

## The pipeline (and where each stage lives)

1. **Per-step representations** — one forward pass per step in context; pool hidden
   states to one vector `v_t`. → `src/extract/activations.py`
2. **Attention mass** — per-step attention into predecessors, without ever
   materialising an (N,N) matrix. → `src/extract/attention.py`
3. **Base score** — fit SVD on train, score the config grid per seed. → `src/score/`
4. **Select (pass 1)** — per-window SVD config + baseline rows. →
   `src/reports/triples.py`
5. **Rescoring sweep** — union base table over windows, then orient → normalize →
   strategy, all γ at once. → `src/rescore/`
6. **Select (pass 2)** — same command as 4; now fills the rescoring rows and the
   side-by-side summary. → `src/reports/triples.py`
7. **Reproduce** — re-run selected rows, emit per-step scores/ranks/predictions,
   assert the recorded accuracy. → `main/reproduce.py` (no longer in `src/`)
8. **Analyse** — geometry probe + figure data. → `src/analysis/`

`scripts/run_pipeline.sh` drives 3→6 end to end (`DS=<ds> bash
scripts/run_pipeline.sh`).

**From results to manuscript** (the current path, built on `main/`):
`scripts/main/sweep_triples.py` runs `main/`'s sweep over every seed triple and writes
`results-sweep/`; `pick_triple.py` chooses the reported triple per cell;
`scripts/tables/sync_seeds.py` freezes the picks into `configs-main/`;
`scripts/prompting/evaluate.py` scores the prompting baselines on the same test splits
and writes `results-prompting/`; `scripts/tables/make_main_tables.py` joins the two
into `tables/*.tsv`, which the manuscript reads. How the picks are made:
`documents/SELECTION.md`.

## Repository layout

- **`src/`** — the full sweep machinery. Each stage is a package with its algorithm
  plus a thin `run.py` exposing the same CLI. Module map: `documents/MODULES.md`.
- **`main/`** — the simplified runner (~2.2k lines vs `src/`'s ~5.4k). Self-contained,
  fixed axes removed from the code, seeds frozen per subset, one `sweep` command. Owns
  reproduction. Reads `configs-main/<ds>.yaml`, writes `results-nogt/` /
  `results-gt/`. Operational guide: `main/README.md`.
- **`configs/`** — what `src/` runs. `configs/datasets/<ds>.yaml` is the per-dataset
  manifest and single source of truth (models, model_paths, subsets, max_tokens,
  splits, seeds, ks); `configs/score/<ds>.yaml` drives scoring;
  `configs/protocol/<ds>.yaml` is ONE config driving select + rescore (+ windows +
  sweep grid).
- **`configs-main/`** — `main/`'s single per-dataset config, one plain and one `-gt`
  file per dataset. Holds the frozen seed triples; only
  `scripts/tables/sync_seeds.py` may edit the seed blocks.
- **`data/<ds>/<subset>/*.json`** — the corpus. Input, never regenerated by a run.
- **`outputs/<ds>/`, `outputs-gt/<ds>/`** — everything a `src/` run produces, without
  and with GT. FROZEN as the reference: `main/` never writes into them.
- **`results-nogt/<ds>/`, `results-gt/<ds>/`** — what `main/` produces. Their
  `activations/` and `attention/` are seeded from `outputs[-gt]/` by
  `scripts/seed_results.sh` (hardlink copy — the artifacts are byte-compatible).
- **`results-sweep/`** (gitignored) — the seed-triple sweep: per-triple selections,
  the full grid, and the `best_triples_*.tsv` picks. Reader's guide:
  `scripts/main/README.md`.
- **`results-prompting/`** — prompting-baseline scores on SOAP's exact test splits
  (`by_cell` / `by_column` / `by_seed`), from `scripts/prompting/evaluate.py`.
- **`tables/`** — the manuscript's tables as TSV, built by
  `scripts/tables/make_main_tables.py`. The LaTeX reads its numbers from these.
- **`manuscript/`** — the ICLR 2026 LaTeX source.
- **`scripts/`** — drivers and checkers: `scripts/main/` (triple sweep),
  `scripts/prompting/` (baseline scoring + verification), `scripts/tables/` (tables +
  seed sync), parity checkers (`documents/TESTING.md`), and the shell drivers
  (`run_pipeline.sh`, `run_main_all.sh`, `seed_results.sh`, `extract.sh`).
- **`datagen/`** — synthetic trajectory generation for TraceElephant (Magentic-One +
  Captain-Agent over question pools). Its own world; see `datagen/README.md` and
  `datagen/SCRIPTS.md`.
- **`documents/`** — the deep documentation this file points to.
- **`artifacts/`** — exported figure data and qualitative examples for the manuscript.
- **`tests/`** — the CPU test suite (`documents/TESTING.md`).

## Running

Everything runs **from the repo root**. `src/` shares one CLI across every stage:

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

`main/` has four commands — `extract`, `sweep`, `select`, `reproduce` — documented in
`main/README.md`; `scripts/run_main_all.sh` drives all datasets and both GT settings.

## `main/` — the simplified runner

Six choices are FROZEN in `main/`'s code, not exposed as config: `pooling=mean` (for
scoring; the extractor still writes both), uncentered SVD, `method=proj`, orientation
folded in as `1/(pi+eps)`, `score_norm=none`, `weighted=false`. What remains swept is
`position x c_begin x c_end` and `layer_range x gamma x w x strategy`. Seeds are a
frozen triple per subset in `configs-main/<ds>.yaml`, shared across backbones.

Two structural consequences:

- **No separate base table.** At `gamma=0`, `S̃ = s` exactly for every matrix, and
  `1/(pi+eps)` descending is order-identical to `pi` ascending (ties included). So the
  `gamma=0` rows ARE the base score; `sweep.py` asserts that against the
  `strategy="base"` rows on every run. `undisc_*`, `base_triples.tsv` and the two-pass
  dance are all gone.
- **One `sweep` command**, because the rescore grid is expanded only for the SELECTED
  base config. The full cross product would be ~3.7M rows per seed; staged it is ~24k.

`ens-mid3` is the ONE deliberate numerical divergence from `src/`: `src/` orients
ensemble members by negation before z-scoring (orient was a separate downstream stage
that `ens-mid3` bypassed), `main/` uses the folded inverse. Every single-position
number is bit-identical; only `ens-mid3` differs — and it has been selected in 0 of
4,428 rows.

## Where the details live

This file orients; the depth is in `documents/`. Read the matching document BEFORE
changing the code it covers:

- **`documents/SELECTION.md`** — how the reported seed triples and hyperparameter
  configs are chosen, in both packages. Read before touching seeds, windows,
  selection rules, or anything under `scripts/main/`.
- **`documents/MODULES.md`** — what each module in `src/` and `main/` does and the
  non-obvious behaviour it hides. Read before editing a module you don't know.
- **`documents/CONVENTIONS.md`** — the conventions that bite: sign conventions, layer
  indexing, frozen trees, float-fragile tiebreaks, the cross-repo GT folder trap.
  Read before changing scoring, rescoring, metrics, or any path handling.
- **`documents/TESTING.md`** — the test suite and every parity checker, and when to
  run each. Read before trusting a regenerated artifact or a long sweep.
- **`documents/SCORERS.md`** — every base scorer, the SVD geometry they exploit, and
  what the results say about them.
- Operational guides: `main/README.md` (running `main/`), `scripts/main/README.md`
  (the triple sweep), `datagen/README.md` (trajectory generation).

## How to write here

All prose in this repo — documentation, docstrings, comments, chat conversations —
follows William Zinsser's *On Writing Well*. When you write or revise any of it, hold
to these rules:

- **Lead with the point.** The first sentence states what the piece is about and why
  it matters. No windup.
- **One step at a time, in logical sequence.** Each sentence rests on the one before,
  so the reader never has to jump a gap.
- **Lead from the familiar to the unfamiliar.** Start where the reader already is,
  then move outward; use analogy to connect the unknown to the known.
- **Write for an intelligent stranger.** Never overestimate the reader's knowledge,
  never underestimate their intelligence.
- **Strip every sentence to its cleanest components.** Nothing in good writing does no
  work. Cut clutter: qualifiers ("a bit", "sort of"), inflated phrases ("at this point
  in time"), adverbs the verb already implies.
- **Use active verbs.** Passive voice hides who did what — the chronic disease of
  technical prose.
- **Prefer short, concrete words** over long Latinate ones; avoid nominalizations
  (nouns manufactured from verbs).
- **Avoid jargon and acronym soup.** If a term is unavoidable, define it in plain
  language the first time it appears.
