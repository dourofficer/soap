# Experimental plan

Every pending experiment for the manuscript, made concrete: data, splits, backbones,
anchor configs, procedure, and cost. Agreed 2026-08-13; revised 2026-08-17 (coverage,
naming, orientations, grids, A7 re-selection); 2026-08-23 (B1 baseline rows);
2026-08-25 (E2 concretized on the gathered synthetic corpora); 2026-08-27 (S1 scalability
planned); 2026-09-03 (B2 open-backbone prompting rows); 2026-09-06 (B3 ErrorProbe,
paper mode); 2026-09-07 (B3 GPT-4o row). The two main experiments fill
`fig:transfer` and `tab:synth`; the seven ablations fill `tab:scorefn`, `tab:weights`,
`tab:position`, `tab:attnsel`, `fig:gamma`, `fig:layers`, `fig:datasize`; S1 fills
`fig:scale`.

## Global conventions

Every experiment below follows these rules. State a deviation explicitly or it is a bug.

- **Coverage.** Every run covers BOTH backbones (qwen3.5-9b, deepseek-8b) and all four
  of WW-AG, WW-HC, TE-Cap, TE-Mag. CE is usually excluded (exception: A2, which
  absorbs a main-table row). Which cells the manuscript shows is decided later — the
  runs produce everything. Unless stated otherwise, runs are without-GT.

- **Splits.** 30/20/50 reference(train)/val/test at trajectory level, on the FROZEN
  seed triples (source of truth: `configs-main/<ds>.yaml`):

  | Subset | Triple (no-GT) | Triple (with-GT) |
  |---|---|---|
  | WW-AG | 3, 4, 5 | 38, 39, 40 |
  | WW-HC | 13, 14, 15 | 13, 14, 15 |
  | CE (all 7 subsets) | 17, 18, 19 | 17, 18, 19 |
  | TE-Cap | 22, 23, 24 | 2, 3, 4 |
  | TE-Mag | 15, 16, 17 | 9, 10, 11 |

  No experiment re-splits. A reported number is the mean over the triple's three seeds.

- **Selection rule.** Any knob that needs selecting — the λ of temporal bias, the
  synthetic-fit hyperparameters, A7's per-fraction configs — uses the SAME rule as
  Table 1: mean TEST step accuracy over the triple, tiebreak agent accuracy. This is
  deliberately optimistic and uniform, so every comparison is fair; the whole protocol
  converts to val-selection later in one sweep.

- **Anchor config** = the `backprop` row selected for the frozen triple, per
  (backbone, subset). Ablations start from it and vary exactly ONE axis. All anchors
  live in `results-{nogt,gt}/<ds>/select/selection.tsv`, regenerated on the frozen
  triples and verified against Tables 1–2. The without-GT anchors:

  | Backbone | Subset | Position | Band | Attn layers | γ | w |
  |---|---|---|---|---|---|---|
  | qwen3.5-9b | WW-AG | act/27 | [1, 7) | 0–2 | 0.6 | 1 |
  | qwen3.5-9b | WW-HC | act/31 | [0, 5) | 4–6 | 0.1 | 2 |
  | qwen3.5-9b | TE-Cap | act/23 | [0, 3) | 6–8 | 0.1 | 5 |
  | qwen3.5-9b | TE-Mag | act/23 | [0, 2) | 6–8 | 1.0 | 4 |
  | deepseek-8b | WW-AG | act/3 | [1, 5) | 24–32 | 1.0 | 1 |
  | deepseek-8b | WW-HC | act/31 | [0, 4) | — | 0.0 | — |
  | deepseek-8b | TE-Cap | act/31_normed | [0, 18) | 8–16 | 0.2 | all |
  | deepseek-8b | TE-Mag | act/10 | [0, 4) | 24–32 | 0.1 | 1 |

  (DeepSeek WW-HC selected γ=0: rescoring is a no-op there.) CE's 14 anchors
  (7 subsets × 2 backbones) sit in `results-nogt/correct-error/select/selection.tsv`.

- **Metric.** Step-level accuracy, mean over the triple (±1 std shading in figures);
  agent accuracy recorded alongside.

- **Cost classes.** *free* = filter the existing sweep grid; *CPU* = rescore existing
  activations, no forward passes; *GPU* = new forward passes.

- **Code, results, and naming.** All ablation/experiment code lives in
  `scripts/ablations/`, one runner per experiment, named `<exp>_<slug>.py`; its output
  is `results-ablations/<exp>_<slug>.tsv` (a directory `<exp>_<slug>/` if one file is
  not enough). `<exp>` is the experiment id (a1…a7, a6a/a6b, e1, e2), the slug says
  what it varies: `a1_scorefn`, `a2_weights`, `a3_position`, `a4_window`, `a5_gamma`,
  `a6a_rep_layer`, `a6b_attn_band`, `a7_datasize`, `e1_transfer`, `e2_synthfit`;
  baseline scorers use `b<n>`: `b1_rb_baselines`.
  When an experiment finishes, its headline numbers are written into THIS file under a
  **Results** block (full precision stays in the TSV). `manuscript/` is never edited
  by these runs — the implied edits are listed at the end and applied only on request.

## Step 0 — regenerate stale anchors  `[CPU]`  — DONE 2026-08-14/15

- [x] Stale cells: TraceElephant (both trees, pre-freeze seeds 8–10 / 14–16) and —
  found a day later by A2's self-check — `results-gt/ww` (seeds 3–5 vs frozen 38–40).
  CE's selections were already on the frozen triple 17–19 in both trees. Re-ran
  `main sweep --force` + `main select --force` for the stale cells
  (`configs-main/<ds>[-gt].yaml` — never `--set gt=true` on the plain config, its
  seeds differ).
- **Verified:** every regenerated row — config and accuracy — matches the
  frozen-triple rows of `results-sweep/selections_all.tsv`, and the SOAP/base cells
  of Tables 1–2 are unchanged. All twelve select trees now match their frozen
  triples (stamps audited).

## Main experiments

### E1 — Cross-distribution transfer (`tab:transfer`)  `[CPU]`  — REVISED + DONE 2026-08-20

- [x] **Target.** Is a fitted SOAP specific to the distribution it was tuned on?
- **Procedure (revised 2026-08-20; supersedes the frozen-anchor design below).**
  4×4 source→target grid over {WW-AG, WW-HC, TE-Cap, TE-Mag}, one grid per backbone.
  For each pair: fit R on the SOURCE's train split; re-partition the TARGET for the
  cross setting — its val = its main-experiment train + val files (the target's
  train split is unused for fitting here), its test = the main-experiment test
  split, unchanged. RE-SELECT the full config per pair (dense base grid, then the
  backprop rescore grid on the winning base config) under TWO conventions:
  (1) *test-selected* — on mean target-test step accuracy, optimistic, as in
  Table 1; (2) *val-selected* — on mean target-val step accuracy, reporting test.
  Dependency weights always come from the target trajectories' own attention. Seeds
  pair positionally: source seed i's train split ↔ target seed i's val/test splits;
  report the 3-seed mean.
- **Sanity check.** Under the test convention the diagonal repeats Table 1's
  selection problem exactly, so those cells must reproduce the selection table —
  asserted in the runner. The REPORTED tables put the main-experiment in-distribution
  numbers on every diagonal, whichever convention the off-diagonal cells use.
- **Deliverable.** FOUR 4×4 tables: {test-selected, val-selected} × {qwen3.5-9b,
  deepseek-8b}. Which appear in the main text is decided later.
- (CE dropped from the grid with the pooled-source design; can be reinstated later if
  a 5×5 is wanted.)
- **Results** — `results-ablations/e1_transfer.tsv` (merged from
  `e1_parts_reselect/`; `scripts/ablations/e1_transfer.py`, rewritten; columns
  include `convention`; base and soap rows per pair, val metrics alongside test;
  all 8 test-convention diagonals reproduced the selection table exactly). The four
  tables — SOAP step acc %, rows = source, diagonal (bold) = main-exp in-dist:

  **qwen3.5-9b, test-selected**

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **47.62** | 32.18 | 34.11 | 21.01 |
  | WW-HC | 35.45 | **34.48** | 34.11 | 22.46 |
  | TE-Cap | 26.98 | 33.33 | **35.66** | 21.74 |
  | TE-Mag | 28.57 | 29.89 | 34.88 | **23.19** |

  **qwen3.5-9b, val-selected**

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **47.62** | 26.44 | 17.05 | 18.84 |
  | WW-HC | 23.81 | **34.48** | 20.16 | 21.01 |
  | TE-Cap | 20.63 | 28.74 | **35.66** | 16.67 |
  | TE-Mag | 24.87 | 26.44 | 25.58 | **23.19** |

  **deepseek-8b, test-selected**

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **45.50** | 25.29 | 34.88 | 30.43 |
  | WW-HC | 40.74 | **28.74** | 42.64 | 33.33 |
  | TE-Cap | 29.63 | 32.18 | **42.64** | 28.99 |
  | TE-Mag | 35.45 | 29.89 | 30.23 | **30.43** |

  **deepseek-8b, val-selected**

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **45.50** | 20.69 | 25.58 | 27.54 |
  | WW-HC | 39.68 | **28.74** | 35.66 | 23.91 |
  | TE-Cap | 29.10 | 20.69 | **42.64** | 18.12 |
  | TE-Mag | 31.22 | 24.14 | 23.26 | **30.43** |

  Reading: once the configuration is re-selected on the target, the reference R
  itself transfers far better than the frozen-anchor design suggested. Under the
  optimistic test convention most cross cells land within a few points of the
  diagonal, and a foreign reference can even beat the in-distribution one
  (DeepSeek WW-HC→TE-Cap ties 42.64; WW-HC→TE-Mag 33.33 vs 30.43). The honest
  val convention restores the gap — the diagonal wins every column on both
  backbones — but the degradation is graded, not catastrophic (DeepSeek
  WW-HC→WW-AG keeps 39.68 of 45.50): what is distribution-specific is mostly the
  hyperparameter configuration, not the spectral reference.

#### Superseded frozen-anchor design (results kept for reference)

- Old procedure: freeze the source's full anchor config, evaluate on the target's
  test split. Results: `results-ablations/e1_transfer_frozen-anchor.tsv`.
  SOAP step acc %, rows = source:

  | qwen3.5-9b | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **47.62** | 3.45 | 20.16 | 5.80 |
  | WW-HC | 23.81 | **34.48** | 20.16 | 14.49 |
  | TE-Cap | 12.70 | 13.79 | **35.66** | 20.29 |
  | TE-Mag | 11.64 | 10.34 | 27.91 | **23.19** |

  | deepseek-8b | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **45.50** | 3.45 | 20.16 | 9.42 |
  | WW-HC | 13.76 | **28.74** | 34.11 | 12.32 |
  | TE-Cap | 15.87 | 21.84 | **42.64** | 10.14 |
  | TE-Mag | 24.34 | 19.54 | 16.28 | **30.43** |

  Reading: transfer degrades sharply off-diagonal — the diagonal wins every column
  but one (DeepSeek TE-Mag→WW-AG 24.34 is the best non-diagonal source for WW-AG
  but still 21 points under in-distribution). A fitted SOAP is distribution-specific;
  the closest cross pair is DeepSeek WW-HC→TE-Cap (34.11 vs 42.64 in-dist).

### E2 — Synthetic reference trajectories (`tab:synth`)  `[GPU]`  — DONE 2026-08-25

- [x] **Target.** Does SOAP work when no corpus from the target system exists to fit R?
- **Design in one sentence.** IDENTICAL to the main experiment — same frozen triples,
  same val/test partitions, same selection rule — except the fit set: R is fit on a
  synthetic corpus instead of the seed's train split, which goes unused.
- **Scope.** WW-AG and WW-HC only — `tab:synth` shrinks from five subsets to two.
  Generators **Qwen3.5-9B** and **GPT-4o** (Qwen3.5-35B-A3B is dropped everywhere;
  the `mixed` corpus is out of scope).
- **Data.** The gathered corpora in `../datagen/data/synthetic/`, produced by the
  harness that generated each WW subset — CaptainAgent for WW-AG, Magentic-One for
  WW-HC — with the generator LLM as the agents' backbone. Each corpus is FILTERED to
  the trajectories whose question appears in the target subset, so the fit set's
  question pool is identical to WW's (the fresh runs, not the extra
  gaia/assistantbench questions the generation also covered, and none of
  magentic-qwen9b's ~1,300 off-pool trajectories). Trajectories are used as-is
  regardless of task success. The filtered corpora:

  | Target | Corpus | Kept / generated | Question coverage |
  |---|---|---|---|
  | WW-AG | `captain-gpt4o` | 126 / 198 | 126 of 126 |
  | WW-AG | `captain-qwen9b` | 124 / 196 | 124 of 126 |
  | WW-HC | `magentic-gpt4o` | 55 / 198 | 55 of 58 |
  | WW-HC | `magentic-qwen9b` | 55 / 1,502 | 55 of 58 |

  One corpus per (target, generator), shared across the triple's seeds — per-seed
  variance comes from the val/test splits alone. The same questions appear in the
  fit set and in val/test BY DESIGN (that is the use case: re-run your own agents on
  the tasks you want to diagnose, then fit on those runs); no step label is ever
  read from the synthetic side — caption note, and the runner records the per-split
  overlap counts.
- **Procedure.** Materialize the filtered corpora under `data/synthetic/` →
  extract activations for both backbones (GPU; activations ONLY — dependency
  weights come from the target trajectories' own attention, already extracted) →
  fit R once per (corpus, backbone, position), reused across the three seeds →
  RE-SELECT the full config per (target, generator, backbone) by the standard rule
  (dense base grid, then the backprop rescore grid on the winning base config) →
  report S and +SOAP. Each row is "the best that reference corpus can do", matching
  the optimistic protocol of the real-corpus row; val metrics are recorded alongside
  for the protocol-wide val-selection conversion later.
- **Sanity check.** The same code path run with the real train split as reference is
  exactly Table 1's selection problem, so those cells must reproduce
  `results-nogt/ww/select/selection.tsv` — asserted in the runner, and they double
  as the Real row.
- **Rows.** Real (= Table 1) / synthetic Qwen3.5-9B / synthetic GPT-4o, base and
  SOAP each.
- **Code and output.** `scripts/ablations/e2_synthfit.py` (base/rescore grid
  machinery lifted from `e1_transfer.py`) → `results-ablations/e2_synthfit.tsv`;
  extraction via `configs-main/synthetic.yaml` into
  `results-nogt/synthetic/activations/` (mind the torchvision/torchaudio guard —
  environment note at the end of this file).
- **Cost.** GPU: ~360 trajectories × 2 backbones of activation extraction. CPU:
  8 selection problems (2 targets × 2 generators × 2 backbones), each the size of
  one E1 pair.
- **Results** — `results-ablations/e2_synthfit.tsv` (merged from `e2_parts/`;
  `scripts/ablations/e2_synthfit.py`; corpora staged by `e2_stage_data.py`,
  extracted 2026-08-25; all four real cells reproduced the selection table
  exactly; overlap columns confirm the intended coverage — WW-AG val ~26/26 and
  test 62–63/63, WW-HC val 12/12 and test ~27/29, the shortfall being the 2+3
  never-generated questions). SOAP step acc % per reference (base in
  parentheses):

  | qwen3.5-9b | WW-AG | WW-HC |
  |---|---|---|
  | Real (= Table 1) | 47.62 (39.15) | 34.48 (33.33) |
  | Synthetic Qwen3.5-9B | 41.80 (33.33) | 31.03 (28.74) |
  | Synthetic GPT-4o | 46.56 (40.74) | 29.89 (28.74) |

  | deepseek-8b | WW-AG | WW-HC |
  |---|---|---|
  | Real (= Table 1) | 45.50 (38.62) | 28.74 (28.74) |
  | Synthetic Qwen3.5-9B | 39.68 (35.98) | 29.89 (26.44) |
  | Synthetic GPT-4o | 38.10 (33.33) | 25.29 (25.29) |

  Reading: SOAP survives the loss of the real corpus. Every synthetic cell lands
  within 1–7 points of its real row — the best generator per cell within ~1 point
  on Qwen WW-AG (46.56 vs 47.62) and ABOVE real on DeepSeek WW-HC (29.89 vs
  28.74) — and every synthetic SOAP row stays far above the representation
  baselines of B1. The selected configs move with the reference (Qwen WW-AG picks
  act/31 [1,13) under the Qwen corpus vs act/27 [1,7) real), echoing E1: the
  distribution-specific part is the hyperparameter configuration, and re-selecting
  it on the synthetic fit recovers most of the accuracy. Rescoring keeps working
  on synthetic references — up to +8.5 over base (Qwen WW-AG, Qwen corpus) —
  except where γ=0 is selected (DeepSeek WW-HC GPT-4o: SOAP = base, as in the
  real anchor). Neither generator dominates: GPT-4o's corpus wins Qwen/WW-AG,
  the Qwen3.5-9B corpus wins the other three cells, so a cheap open-weights
  generator is a viable source of reference trajectories.

## Scalability

### S1 — Scalability to larger backbones (`fig:scale`)  `[GPU for SOAP; CPU for the baselines]`  — DONE 2026-08-28

- [x] **Target.** Does SOAP keep working — and keep its margin over the
  representation-based baselines — as the proxy grows? One line per method, one
  panel per WW subset, x = backbone size.
- **Scope.** WW-AG and WW-HC only, WITHOUT GT ONLY (no with-GT arm), on the
  frozen triples (WW-AG 3, 4, 5; WW-HC 13, 14, 15) and the same 30/20/50
  partitions as Table 1 — nothing is re-split. Methods: **SOAP** and its **base score** (the γ=0 row), **OAT** and
  **StepFinder** (family A). This replaces the manuscript's current plan of
  All-at-Once + AgenTracer + OAT: All-at-Once has no open-weight run at these sizes
  and AgenTracer is not set up, whereas OAT and StepFinder are already trained and
  predicted at every size (below). All four methods read the SAME backbone, so the
  comparison is within one representation.
- **Backbones** (weights in `../hub/Qwen/`, config facts checked 2026-08-27):

  | Name | Family | Layers | Attention blocks | Hidden | Notes |
  |---|---|---|---|---|---|
  | `qwen3.5-9b` | Qwen3.5, hybrid | 32 | 8 full-attention | 4096 | = Table 1; no new runs |
  | `qwen3-14b` | **Qwen3**, dense | 40 | 40 | 5120 | every block is an attention block |
  | `qwen3.5-27b` | Qwen3.5, hybrid | 64 | 16 full-attention | 5120 | fits one H200 in bf16 (~54 GB) |

  **The 14B point is Qwen3-14B, not Qwen3.5-14B** — the manuscript's "Qwen3.5-14B"
  does not exist on disk (`../hub/Qwen/Qwen3.5-14B/` is empty) and the baselines
  were run on Qwen3-14B. Caption note: the middle point crosses to the previous
  Qwen generation; the x-axis is parameter count. Qwen3.5-4B is available on both
  sides (weights + baseline predictions) and can be added as a fourth point if the
  curve needs a left anchor; default is three points, as in the manuscript.
- **Baselines — already done, verified 2026-08-27.** Predictions live in
  `../attrib-prompting/outputs-rb-{nogt,gt}/ww/<subset>/<backbone>/`, produced
  2026-08-24, for `qwen3.5-4b`, `qwen3.5-9b`, `qwen3-14b`, `qwen3.5-27b` (and
  `deepseek-8b`): `oat.s42–46` and `stepfinder.s42–46` in every cell, 127 (WW-AG)
  / 59 (WW-HC) JSONs per run, i.e. the full 126/58 corpus, in BOTH GT settings.
  StepFinder also has `stepfinder-tsel` (checkpoint test-selected),
  `stepfinder-pca` (PCA to 128 dims instead of the first-128 slice) and
  `stepfinder-pca-tsel`. The reported row is `stepfinder` (val-selected, first-128
  slice) — the Table-1 convention; the PCA variant is recorded in the TSV because the
  slice is arbitrary for a decoder and a reader will ask.
  **Do not read `../attrib-prompting/scale_ww.tsv`**: it aggregates on
  attrib-prompting's own splits (OAT / 9B / WW-AG 21.98 there vs 16.72 on the
  frozen triple in B1). Every baseline number in this experiment is re-scored on the
  frozen triples by the B1 runner (`scripts/ablations/b1_rb_baselines.py`, judges
  extended with `qwen3-14b` and `qwen3.5-27b`), mean over the five training seeds,
  exactly as in Tables 1–2.
- **SOAP — needs extraction at 14B and 27B.** Nothing exists for either backbone in
  `outputs/` or `results-nogt/`. Procedure, identical to the main experiment:
  1. Add the two backbones to `configs-main/ww.yaml` itself — `models:
     [qwen3.5-9b, deepseek-8b, qwen3-14b, qwen3.5-27b]` plus their `model_paths`
     (`../hub/Qwen/Qwen3-14B`, `../hub/Qwen/Qwen3.5-27B`). Seeds, splits and grids
     (`positions: all`, 20 components, `n_ranges: 4`, γ and w grids, `backprop`)
     stay verbatim, so `run_stamp.json` accepts the existing tables. Everything
     lands in `results-nogt/ww/` next to the 9B/DeepSeek cells:
     `{activations,attention}/<model>/<subset>/`, `sweep/<model>/<subset>/`, and
     new rows in `select/selection.tsv`. The existing 9B and DeepSeek rows must
     come out of `select --force` byte-identical (selection is per model, so
     adding models cannot move them — asserted anyway). No `results-gt/` run.
  2. `python -m main extract --config configs-main/ww.yaml --model <m>` for the two
     new backbones, both subsets, activations + attention. Resumable per
     trajectory; one GPU per backbone in parallel (GPUs 3–7 are idle).
  3. `python -m main sweep --model <m>` then `python -m main select --force` — the
     standard rule (mean TEST step accuracy over the triple, tiebreak agent
     accuracy), dense base grid then the rescore grid on the winning base config.
  4. Report base (γ=0) and SOAP per (backbone, subset), plus the selected config,
     read from `select/selection.tsv` like every other anchor.
- **Layer conventions for the new backbones** (CONVENTIONS.md: `layer_range`
  indexes ATTENTION blocks, positions index decoder blocks):
  - `qwen3-14b` goes through the default dense `ModelAdapter` (as DeepSeek does):
    40 positions `act/1..40` (+ `embed`, `_normed`), attention bands of 10 blocks
    (0–10, 10–20, 20–30, 30–40).
  - `qwen3.5-27b` goes through `Qwen35Adapter`: 16 extracted blocks, so 16
    positions and attention bands of 4 blocks (0–4, 4–8, 8–12, 12–16).
  - The base grid grows with the position count (14B: ~40 × 210 bands × 3 seeds);
    still CPU-cheap.
- **Pre-flight checks before the GPU runs.**
  - Qwen3's chat template defaults to thinking ON; `Qwen35Adapter` passes
    `enable_thinking=False` but the default adapter passes nothing. Confirm the
    dense adapter renders Qwen3-14B steps without `<think>` blocks (or give Qwen3
    its own adapter with the same flag) — otherwise the step boundaries shift.
  - `max_tokens: 8192` and bf16 as for 9B; confirm the 27B attention extractor's
    per-step hooks cover the 16 full-attention blocks (`extract_block_indices`).
  - Extraction timing from the July 9B run (`results-nogt/ww` mtimes): WW-AG
    ~10 min, WW-HC ~2 h 15 min for activations + attention on one H200 (WW-HC
    trajectories are long). Budget ~1.5× for 14B and ~3× for 27B: about 4 h and
    7 h respectively, run in parallel. Disk: 14B activations ~4 GB (40 layers ×
    5120 vs 9B's 637 MB), 27B ~1.6 GB.
- **Sanity checks (asserted in `s1_scale.py --stage merge`; verified by hand
  2026-08-30 against the sweep tables and the raw predictions).** (i) The 9B SOAP/base cells reproduce
  Table 1 exactly — WW-AG 47.62 / 39.15, WW-HC 34.48 / 33.33. (ii) The 9B OAT and
  StepFinder cells reproduce B1 — OAT 16.72 / 10.11, StepFinder 15.87 / 13.33.
  (iii) 0 missing predictions in every baseline cell.
- **Selection note.** Every SOAP point is re-selected at its own size (the same
  optimistic test-selection as Table 1); the baselines carry no selection beyond
  their val-selected checkpoint. Val metrics are recorded alongside for the
  protocol-wide val-selection conversion later.
- **With-GT.** Not run. (The baselines' with-GT predictions exist in
  `outputs-rb-gt/` should this ever change; SOAP would need a second extraction.)
- **Deliverable — DONE 2026-08-30.** `scripts/ablations/s1_scale.py` (stages: `soap` =
  per-seed rows of the selected config and its γ=0 base from `results-nogt/ww/
  sweep/`, checked against `select/selection.tsv` → `baselines` = B1 scoring of the
  four backbones, all five StepFinder variants, one row per training seed →
  `merge`). Per-seed parts in `results-ablations/s1_parts/`; `s1_scale.tsv` = one
  row per (method, backbone, subset) with mean / std / agent / val and, for SOAP,
  the selected position / band / attention band / γ / w — the runner regenerates
  the 2026-08-28 file byte-identically; `s1_scale_summary.tsv` = the "mean (std)"
  strings of the tables above, step and agent, 4B–27B. Figure:
  `plot_figures.py --only fig_scale` → `artifacts/ablations/fig_scale.pdf`, copied
  to `manuscript/assets/` — grouped BARS (fig_sensitivity style, not lines) per
  backbone size, panels WW-AG / WW-HC, bars SOAP / base / OAT / StepFinder, ±1 std
  error bars; 4B omitted (baselines only).
- **Cost.** GPU: two extractions (14B, 27B) × two subsets, ≈ 11 GPU-hours in
  parallel on two H200s. CPU: two sweeps the size of a Table-1 cell pair (the 14B
  base grid is ~5× the 9B one), plus B1 rescoring (seconds).
- **Results — baselines, PREFILLED 2026-08-27** (scored on the frozen triples with
  `scripts/prompting/evaluate.py`'s rules through the B1 path, `outputs-rb-nogt`
  only; 0 missing predictions in all 40 cells; the 9B `oat` / `stepfinder` cells
  reproduce B1 exactly). Mean over triple × 5 training seeds; std over training
  seeds in parentheses. Step acc %:

  | WW-AG | Qwen3.5-4B | Qwen3.5-9B | Qwen3-14B | Qwen3.5-27B |
  |---|---|---|---|---|
  | OAT | 20.00 (1.4) | 16.72 (1.7) | 13.97 (1.7) | 20.42 (1.6) |
  | StepFinder (reported: val-sel., first-128) | 15.34 (4.0) | 15.87 (3.4) | 22.65 (4.2) | 13.54 (2.3) |
  | StepFinder-tsel (test-sel. checkpoint) | 24.02 (1.9) | 24.55 (4.3) | 29.63 (1.9) | 18.31 (2.1) |
  | StepFinder-pca (val-sel., PCA-128) | 22.12 (3.7) | 25.19 (5.3) | 21.90 (4.1) | 24.87 (3.0) |
  | StepFinder-pca-tsel | 29.95 (1.4) | 30.48 (2.3) | 24.97 (2.5) | 28.68 (3.5) |
  | SOAP (base) | — | 47.62 (39.15) | 42.86 (40.21) | 44.97 (38.62) |

  | WW-HC | Qwen3.5-4B | Qwen3.5-9B | Qwen3-14B | Qwen3.5-27B |
  |---|---|---|---|---|
  | OAT | 11.03 (1.0) | 10.11 (1.3) | 16.78 (2.2) | 12.41 (1.3) |
  | StepFinder (reported) | 10.80 (3.0) | 13.33 (5.4) | 14.02 (3.3) | 16.55 (7.0) |
  | StepFinder-tsel | 11.95 (4.1) | 14.48 (6.5) | 13.56 (4.3) | 11.03 (6.7) |
  | StepFinder-pca | 10.11 (2.5) | 8.05 (3.4) | 6.67 (1.9) | 9.43 (2.5) |
  | StepFinder-pca-tsel | 10.57 (5.7) | 10.80 (3.2) | 4.83 (2.6) | 7.36 (1.7) |
  | SOAP (base) | — | 34.48 (33.33) | 25.29 (25.29) | 34.48 (33.33) |

  Agent acc %:

  | | 4B AG | 9B AG | 14B AG | 27B AG | 4B HC | 9B HC | 14B HC | 27B HC |
  |---|---|---|---|---|---|---|---|---|
  | OAT | 46.35 | 40.53 | 45.29 | 48.15 | 55.17 | 53.10 | 57.24 | 46.90 |
  | StepFinder | 42.43 | 39.26 | 46.98 | 39.89 | 32.64 | 37.01 | 40.00 | 38.39 |
  | StepFinder-tsel | 48.36 | 42.65 | 50.69 | 42.96 | 33.79 | 38.16 | 43.45 | 35.86 |
  | StepFinder-pca | 51.75 | 48.78 | 42.65 | 44.02 | 32.41 | 33.33 | 32.18 | 32.41 |
  | StepFinder-pca-tsel | 55.13 | 51.01 | 44.87 | 50.37 | 35.17 | 34.48 | 32.87 | 33.79 |

  Reading: neither baseline scales. OAT is flat within noise on WW-AG (14–20 over
  4B→27B, the 14B dip being the Qwen3 point) and stays at 10–17 on WW-HC; the
  reported StepFinder moves 13–23 on WW-AG and 11–17 on WW-HC with no monotone
  trend and training-seed stds of 2–7 points. Even the most generous variant
  (pca-tsel, test-selected checkpoint AND a better projection, ~30 on WW-AG) sits
  17 points under 9B SOAP; on WW-HC every variant is below 17 against SOAP's 34.5.
  The experiment therefore hinges on SOAP's own 14B/27B points: a flat-or-rising
  SOAP line over flat baselines is the figure.
- **Results — SOAP, DONE 2026-08-28** — `results-ablations/s1_scale.tsv` (SOAP and
  baseline rows in one file; SOAP rows read from `results-nogt/ww/select/
  selection.tsv`, which now holds 32 rows — the 16 pre-existing 9B/DeepSeek rows
  came out of `select --force` byte-identical). Extraction 2026-08-27/28 on two
  H200s (`logs/s1_extract_*.log`; WW-HC attention 1 h at 14B, 2 h at 27B); the
  pre-flight thinking-flag concern was moot — without a generation prompt Qwen3
  and Qwen3.5 render the steps identically. Step acc %, mean over the triple (std
  over seeds), with the selected config:

  | WW-AG | base | SOAP | position | band | attn | γ | w |
  |---|---|---|---|---|---|---|---|
  | Qwen3.5-9B (= Table 1) | 39.15 (5.6) | 47.62 (3.2) | act/27 | [1,7) | 0–2 | 0.6 | 1 |
  | Qwen3-14B | 40.21 (4.6) | 42.86 (5.7) | act/35 | [1,5) | 0–10 | 0.4 | 3 |
  | Qwen3.5-27B | 38.62 (7.5) | 44.97 (3.3) | act/59 | [1,7) | 0–4 | 0.4 | 1 |

  | WW-HC | base | SOAP | position | band | attn | γ | w |
  |---|---|---|---|---|---|---|---|
  | Qwen3.5-9B (= Table 1) | 33.33 (5.3) | 34.48 (3.5) | act/31 | [0,5) | 4–6 | 0.1 | 2 |
  | Qwen3-14B | 25.29 (8.0) | 25.29 (8.0) | act/16 | [0,1) | 30–40 | 0.1 | all |
  | Qwen3.5-27B | 33.33 (2.0) | 34.48 (3.5) | act/63_normed | [0,4) | 8–12 | 0.1 | all |

  Agent acc %: WW-AG 60.32 / 55.03 / 59.26 (SOAP, 9B/14B/27B), WW-HC 67.82 /
  57.47 / 68.97.

  Reading: SOAP holds its level with scale rather than growing with it. Within the
  Qwen3.5 family the 27B point matches the 9B point on both subsets (WW-AG 44.97 vs
  47.62, within one seed-std; WW-HC 34.48 = 34.48, with the SAME γ=0.1 lift over an
  identical 33.33 base), and its selected configs mirror the 9B ones — a late layer,
  band starting at 1 on WW-AG and at 0 on WW-HC, the first attention band on WW-AG.
  Rescoring lifts every backbone on WW-AG (+2.6 to +8.5) and the two Qwen3.5 sizes
  on WW-HC. The Qwen3-14B point is the outlier: competitive on WW-AG (42.86) but
  10 points lower on WW-HC (25.29), where its selection degenerates to a
  mid-stack layer with a one-component band [0,1) and γ=0.1 with w=all — a
  rescoring that changes nothing. Because that point also crosses model
  generations (Qwen3, not Qwen3.5), it is a family effect as much as a size
  effect — caption note; the clean scale comparison is 9B → 27B within Qwen3.5.
  Against the baselines the margin is intact at every size: the best OAT /
  StepFinder (reported) cell is 22.65 on WW-AG and 16.78 on WW-HC, against SOAP's
  worst of 42.86 and 25.29.
- **Manuscript edits — APPLIED 2026-08-30.** The "Scalability to larger backbones"
  paragraph rewritten with the results; representatives are OAT and StepFinder
  (All-at-Once, AgenTracer dropped); "Qwen3.5-14B" → Qwen3-14B with the family
  note in text and caption; the placeholder replaced by `assets/fig_scale.pdf`.
  Old text kept as a dated comment. Still open: Appendix `tab:proxies` (six
  proxies, 8B–27B) can take the same per-backbone base/SOAP numbers.

## Ablations

All ablations: start from the anchor config, vary exactly one axis, hold everything
else fixed. The anchor's own value appears as one point of every sweep and must
coincide with the Table-1 number.

### A1 — Alternative scoring functions (`tab:scorefn`)  `[CPU + one GPU pass]`  — DONE 2026-08-17

- [x] **Target.** Is the spectral band the right base score? No rescoring anywhere in
  this table.
- **Rows** (everything else — layer, band width |C| — at anchor):
  1. *Perplexity* — mean NLL of the step's tokens under the proxy, in context. Needs a
     new GPU extraction pass (token logprobs).
  2. *Random subspace* — projection onto a random orthonormal basis of dimension |C|,
     redrawn per seed.
  3. *Top subspace* — band [0, |C|).
  4. *Tail subspace* — trailing |C| of the 20 computed components.
  5. *Full spectrum* — all 20 computed components (NOT the theoretical full spectrum,
     which would equal the squared L2 norm).
  6. *L1 norm* (exists in `src/` as `norm_l1`).
  7. *L2 norm* (`norm_l2`).
  8. *Spectral band (ours)* = Table 1's base row.
- **Orientations are FIXED, not selected**: perplexity reads "higher = more error";
  every projection-based row (random/top/tail/full/ours) reads "lower = more error".
  The norm rows follow the projection family ("lower = more error" — the full
  spectrum is the squared-L2 limit). No uncertainty scorer (blue note superseded).
- **Results** — `results-ablations/a1_scorefn/scorefn.tsv` (+ per-step NLL under
  `a1_scorefn/nll/`; `scripts/ablations/a1_scorefn.py`, stages `nll` then `score`;
  every "ours" row reproduces Table 1's base row exactly). Step acc %:

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Perplexity | 3.70 | 8.05 | 6.20 | 7.97 |
  | Random subspace | 13.23 | 9.20 | 17.83 | 7.25 |
  | Top subspace | 12.70 | 33.33 | 33.33 | 21.01 |
  | Tail subspace | 6.35 | 6.90 | 6.20 | 1.45 |
  | Full spectrum | 12.17 | 31.03 | 32.56 | 17.39 |
  | L1 norm | 18.52 | 18.39 | 15.50 | 5.07 |
  | L2 norm | 14.29 | 26.44 | 27.13 | 15.22 |
  | Spectral band (ours) | 39.15 | 33.33 | 33.33 | 21.01 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Perplexity | 4.23 | 14.94 | 5.43 | 3.62 |
  | Random subspace | 10.58 | 16.09 | 12.40 | 12.32 |
  | Top subspace | 14.29 | 28.74 | 40.31 | 29.71 |
  | Tail subspace | 10.58 | 4.60 | 18.60 | 5.07 |
  | Full spectrum | 16.93 | 19.54 | 39.53 | 22.46 |
  | L1 norm | 33.86 | 16.09 | 20.16 | 8.70 |
  | L2 norm | 15.87 | 20.69 | 26.36 | 9.42 |
  | Spectral band (ours) | 38.62 | 28.74 | 40.31 | 29.71 |

  Reading: ours is at least tied-best in every cell. Where the anchor band starts at
  component 0 (WW-HC, TE-Cap, TE-Mag), "top subspace" IS the anchor band, so those
  ties are by construction — footnote. The discriminating cells are WW-AG (both
  backbones, anchor starts at 1): dropping the top component is worth ~25 points
  over top/full (39.2 vs 12.7 on Qwen), so WHERE the band sits matters, not just its
  width. Perplexity, random and tail are near-degenerate everywhere; the norms trail
  far behind. DeepSeek TE-Cap's near-tie of full (39.53) follows from its wide
  anchor [0,18) ≈ the full spectrum.

### A2 — Effect of attention-guided rescoring (`tab:weights`)  `[CPU]`  — DONE 2026-08-15

- [x] **Target.** Two questions in one table: does rescoring help at all (base vs
  SOAP), and does the help come from the attention-derived weights rather than from
  any successor aggregation (uniform vs SOAP)?
- **Scope — wider than the other ablations.** All five subsets (CE included), both
  backbones, both GT settings: this table ABSORBS the main tables' "SOAP (w/o
  rescoring)" row, so it covers every cell that row covered.
- **Rows**: *Base score* (S, no rescoring); *uniform (unnormalized)* — every weight 1,
  each step receives the raw SUM of all its successors' base scores; *uniform
  (normalized)* — the MEAN over all successors; *SOAP*. The uniform rows keep the
  anchor's base config and γ and replace only the weights (content-agnostic, all
  successors, no top-w).
- **Results** — `results-ablations/a2_weights.tsv` (`scripts/ablations/a2_weights.py`;
  44 cells × 4 rows; every base row reproduced its selection accuracy exactly before
  the uniform rows were written; with-GT mirror in the TSV). Without-GT step acc %:

  | qwen3.5-9b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | Base score | 39.15 | 33.33 | 61.38 | 33.33 | 21.01 |
  | Uniform (unnormalized) | 16.93 | 16.09 | 57.85 | 14.73 | 5.07 |
  | Uniform (normalized) | 40.21 | 34.48 | 60.78 | 35.66 | 15.94 |
  | SOAP | 47.62 | 34.48 | 61.78 | 35.66 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | Base score | 38.62 | 28.74 | 64.19 | 40.31 | 29.71 |
  | Uniform (unnormalized) | 16.40 | 28.74 | 57.01 | 10.08 | 4.35 |
  | Uniform (normalized) | 35.45 | 28.74 | 62.14 | 41.86 | 28.99 |
  | SOAP | 45.50 | 28.74 | 64.68 | 42.64 | 30.43 |

  Reading: the unnormalized sum collapses toward "predict step 1"; the normalized
  mean sits at or near base and below SOAP wherever rescoring matters — the gain
  comes from the attention weights, not from successor aggregation as such.
- **Caveat.** Cells whose anchor selected γ=0 (DeepSeek WW-HC; several CE subsets —
  6 of 7 for DeepSeek) have all three rescoring rows equal to base by construction —
  footnote.
- **Manuscript edits (later).** Main tables drop the "SOAP (w/o rescoring)" row;
  `tab:weights` widens to five columns and gains the base row; `tab:scorefn`'s
  caption cross-reference repoints to this table's base row.

### A3 — Position-based baselines (`tab:position`)  `[CPU]`  — DONE 2026-08-17

- [x] **Target.** Is SOAP's gain just a preference for early steps?
- **Rows** (spectral base score fixed; bracketed by the base and SOAP rows):
  1. *Temporal bias, z-scored*: z-score the base score within each trajectory, then
     add −λ·t/T.
  2. *Temporal bias, raw*: add −λ·t/T to the raw base score (no z-scoring).
  3. *Earliest of top-5*: predict the earliest step among the 5 highest-scoring; no
     parameter.
- λ ∈ {0.1, 0.2, …, 1.0} for both temporal-bias variants — the same grid shape as γ —
  test-selected by the standard rule.
- **Fairness note.** SOAP's row is simply the Table-1 anchor (no γ co-sweep here);
  every method stands on its most optimistic config.
- **Results** — `results-ablations/a3_position.tsv` (`scripts/ablations/a3_position.py`;
  every λ recorded, `selected` marks the winner; base rows reproduce the selection
  exactly). 2026-08-18: CE added (14 cells, `--configs configs-main/correct-error.yaml`,
  merged into the TSV; CE part also at `a3_position_ce.tsv`) so the manuscript's
  merged rescoring table could show all five subsets — Qwen CE macro-averages:
  temporal-z 61.62, temporal-raw 41.65, earliest-top5 4.51. Selected rows, step acc %:

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Base score | 39.15 | 33.33 | 33.33 | 21.01 |
  | Temporal bias, z-scored | 40.74 (λ=1.0) | 34.48 (λ=0.7) | 34.11 (λ=0.1) | 21.01 (λ=0.2) |
  | Temporal bias, raw | 22.22 (λ=0.1) | 3.45 (λ=1.0) | 6.20 (λ=1.0) | 5.07 (λ=1.0) |
  | Earliest of top-5 | 21.16 | 14.94 | 9.30 | 5.07 |
  | SOAP | 47.62 | 34.48 | 35.66 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Base score | 38.62 | 28.74 | 40.31 | 29.71 |
  | Temporal bias, z-scored | 38.10 (λ=0.1) | 28.74 (λ=0.1) | 41.09 (λ=0.4) | 28.26 (λ=0.4) |
  | Temporal bias, raw | 38.62 (λ=1.0) | 3.45 (λ=1.0) | 6.20 (λ=1.0) | 20.29 (λ=0.1) |
  | Earliest of top-5 | 29.10 | 13.18 | 10.34 | 5.80 |
  | SOAP | 45.50 | 28.74 | 42.64 | 30.43 |

  Reading: no position-based baseline reaches SOAP anywhere (DeepSeek WW-HC ties it,
  but that anchor's γ=0 makes SOAP identical to base there).
  The z-scored temporal bias sits at or a point above base; the raw variant is
  degenerate wherever the base score's scale dwarfs λ (it either matches base or
  collapses); earliest-of-top-5 is far below base everywhere. SOAP's gain is not an
  early-step preference.

### A4 — Context window w (`tab:attnsel`)  `[free]`  — DONE 2026-08-17

- [x] **Target.** Sensitivity to the top-w sparsification of the dependency weights.
- Vary w ∈ {1, 2, 3, 4, 5, all}, everything else at anchor. Read directly from the
  existing sweep grid.
- **Caveat.** DeepSeek WW-HC's anchor has γ=0, so its row is flat by construction —
  footnote it.
- **Results** — `results-ablations/a4_window.tsv` (`scripts/ablations/a4_window.py`;
  pure grid filter, anchor w reproduces the selection exactly). Step acc %, anchor
  in bold:

  | qwen3.5-9b | w=1 | 2 | 3 | 4 | 5 | all |
  |---|---|---|---|---|---|---|
  | WW-AG | **47.62** | 41.80 | 40.74 | 40.21 | 40.21 | 40.21 |
  | WW-HC | 33.33 | **34.48** | 32.18 | 32.18 | 32.18 | 32.18 |
  | TE-Cap | 28.68 | 29.46 | 34.88 | 34.88 | **35.66** | 34.88 |
  | TE-Mag | 15.94 | 20.29 | 18.84 | **23.19** | 18.84 | 18.12 |

  | deepseek-8b | w=1 | 2 | 3 | 4 | 5 | all |
  |---|---|---|---|---|---|---|
  | WW-AG | **45.50** | 38.62 | 33.33 | 34.39 | 37.57 | 37.57 |
  | WW-HC | 28.74 | 28.74 | 28.74 | 28.74 | 28.74 | **28.74** |
  | TE-Cap | 35.66 | 38.76 | 41.09 | 41.09 | 41.86 | **42.64** |
  | TE-Mag | **30.43** | 30.43 | 30.43 | 29.71 | 28.99 | 28.99 |

  Reading: WW favors small w (sharp top-1/2 dependencies) while TE-Cap grows with w
  up to "all" — the right window is dataset-shaped, and a badly wrong w can land
  below the base score (Qwen TE-Mag w=1: 15.94 vs base 21.01), so w is worth
  selecting rather than fixing.

### A5 — Propagation strength γ (`fig:gamma`)  `[CPU]`  — DONE 2026-08-17

- [x] **Target.** Sensitivity to γ; γ=0 is the base scorer and the reference level.
- γ ∈ {0, 0.1, 0.2, 0.3, …, 0.9, 1.0} at anchor. The grid holds
  {0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}; the missing 0.3/0.5/0.7/0.9 need a cheap CPU
  rescore at the anchor, and the overlapping seven double as parity checks. ±1 std
  over the triple.
- **Results** — `results-ablations/a5_gamma.tsv` (`scripts/ablations/a5_gamma.py`;
  PER-SEED rows for the ±1 std shading; all 168 overlapping grid rows matched the
  sweep exactly). Mean step acc % over the triple:

  | qwen3.5-9b | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | WW-AG | 39.15 | 40.74 | 43.92 | 44.44 | 44.44 | 46.03 | **47.62** | 47.09 | 46.56 | 44.97 | 44.44 |
  | WW-HC | 33.33 | **34.48** | 31.03 | 28.74 | 25.29 | 17.24 | 13.79 | 10.34 | 9.20 | 8.05 | 8.05 |
  | TE-Cap | 33.33 | **35.66** | 34.88 | 32.56 | 29.46 | 28.68 | 28.68 | 28.68 | 27.91 | 28.68 | 28.68 |
  | TE-Mag | 21.01 | 19.57 | 19.57 | 18.84 | 17.39 | 17.39 | 17.39 | 17.39 | 18.12 | 21.01 | **23.19** |

  | deepseek-8b | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | WW-AG | 38.62 | 38.10 | 38.62 | 38.62 | 39.68 | 42.33 | 42.33 | 43.39 | 44.44 | 44.44 | **45.50** |
  | WW-HC | **28.74** | 27.59 | 27.59 | 24.14 | 18.39 | 18.39 | 16.09 | 16.09 | 14.94 | 12.64 | 12.64 |
  | TE-Cap | 40.31 | 42.64 | **42.64** | 41.86 | 41.09 | 40.31 | 37.98 | 34.88 | 31.78 | 31.01 | 28.68 |
  | TE-Mag | 29.71 | **30.43** | 26.81 | 23.91 | 23.19 | 23.19 | 23.19 | 23.19 | 23.19 | 23.19 | 23.19 |

  Reading: two regimes. WW-AG (both backbones) climbs monotonically toward large γ —
  the correction carries most of the signal. The other cells peak at small γ
  (0.1–0.2) and decay beyond it — the base score dominates and the correction is a
  nudge. Qwen TE-Mag is the odd one out (dip then recovery to its γ=1.0 anchor). The
  densified points interpolate smoothly; no hidden structure between grid points.

### A6 — Layer position (`fig:layers`)  `[part free, part CPU]`  — DONE 2026-08-17

- [x] **Target.** Where in the network the signal lives — two axes, three views.
- **(a) Representation layer, anchor band**: sweep embed → act/1..N → final-normed,
  holding band, attn layers, γ, w at anchor; plot base AND post-rescoring accuracy
  per layer. Base-per-layer is free (the base grid is dense over positions);
  post-rescoring per layer needs new CPU rescoring runs (the sweep expanded the
  rescore grid only for the winning layer).
- **(a′) Representation layer, best band per layer** `[free]`: base score only; for
  each layer report the accuracy of its BEST spectral band (test-selected over the
  triple by the standard rule). Shows each layer at its own optimum instead of
  through the anchor's band. Which of (a)/(a′) the paper plots is decided later.
- **(b) Attention layer band**: vary over the 4 equal bands (Qwen: 0–2, 2–4, 4–6,
  6–8; DeepSeek: 0–8, 8–16, 16–24, 24–32), all else at anchor — free from the grid.
- **Results (a)/(a′)** — `results-ablations/a6a_rep_layer.tsv`
  (`scripts/ablations/a6a_rep_layer.py`; variants `anchor-base` / `anchor-soap` /
  `best-band` per position; computed base rows verified against the grid, anchor
  position reproduces the selection). Headlines:
  - The anchor position is the per-layer argmax of `anchor-soap` in 7 of 8 cells
    (exception: DeepSeek TE-Mag, where act/9 edges act/10, 31.88 vs 30.43).
  - The layer profile is dataset-shaped: WW-HC (both backbones) concentrates hard in
    the LAST layer (act/31 ≈ 33.3/28.7 vs ≤ 19.5 mid-stack); WW-AG and TE peak in the
    late-middle (Qwen act/23–27, DeepSeek act/3 for WW-AG); embeddings and early
    layers are far weaker.
  - `best-band` barely moves most layers (the anchor band is near each layer's own
    optimum at the peak; off-peak layers gain a few points), so (a) and (a′) tell the
    same story — either can be plotted.
  - Rescoring (`anchor-soap` vs `anchor-base`) lifts nearly every layer on WW-AG
    (e.g. Qwen embed 28.6→36.5, act/19 29.6→40.7), not just the anchor — the
    correction is not tuned to one layer's quirks.
- **Results (b)** — `results-ablations/a6b_attn_band.tsv`
  (`scripts/ablations/a6b_attn_band.py`; pure grid filter, anchors verified). Step
  acc % by attention band, anchor in bold:

  | qwen3.5-9b | 0–2 | 2–4 | 4–6 | 6–8 |
  |---|---|---|---|---|
  | WW-AG | **47.62** | 40.21 | 42.33 | 44.97 |
  | WW-HC | 34.48 | 32.18 | **34.48** | 32.18 |
  | TE-Cap | 35.66 | 35.66 | 34.88 | **35.66** |
  | TE-Mag | 18.12 | 13.04 | 21.74 | **23.19** |

  | deepseek-8b | 0–8 | 8–16 | 16–24 | 24–32 |
  |---|---|---|---|---|
  | WW-AG | 37.04 | 38.62 | 43.92 | **45.50** |
  | WW-HC | 28.74 | **28.74** | 28.74 | 28.74 |
  | TE-Cap | 41.86 | **42.64** | 42.64 | 42.64 |
  | TE-Mag | 28.26 | 28.99 | 28.99 | **30.43** |

  Reading: the band matters least where γ is small (TE-Cap, WW-HC flat) and a few
  points where the correction is strong (WW-AG spans 40.2–47.6 on Qwen); DeepSeek
  consistently prefers late attention blocks (24–32).

### A7 — Quantity of unlabeled reference data (`fig:datasize`)  `[CPU, new code]`  — DONE 2026-08-17

- [x] **Target.** Data efficiency of the reference fit.
- Val/test fixed (frozen triples). Per seed, subsample the train split to
  {1/3, 2/3, 1} of its trajectories (= 10/20/30% of the corpus), refit R, and
  RE-SELECT the full config per fraction — base grid then rescore grid, test-selected
  over the triple by the standard rule — rather than reusing the anchor. Each
  fraction is thus "the best SOAP can do with that much reference data", matching the
  optimistic protocol everywhere else. Report base and rescored accuracy per
  fraction; the 1-fraction cell must reproduce Table 1.
- **Expectation.** WW-HC's 10% point fits R on ~6 trajectories — wide std; show it.
- **Results** — `results-ablations/a7_datasize.tsv` (merged from
  `results-ablations/a7_parts/`; `scripts/ablations/a7_datasize.py`; subsample =
  seeded shuffle + prefix, so fractions are nested; every fraction-1 cell reproduced
  Table 1's base AND SOAP rows exactly). SOAP step acc % per fraction (base in
  parentheses):

  | qwen3.5-9b | 1/3 | 2/3 | 1 |
  |---|---|---|---|
  | WW-AG | 43.39 (35.98) | 44.44 (38.10) | 47.62 (39.15) |
  | WW-HC | 33.33 (33.33) | 33.33 (33.33) | 34.48 (33.33) |
  | TE-Cap | 34.88 (33.33) | 34.88 (33.33) | 35.66 (33.33) |
  | TE-Mag | 23.19 (21.74) | 23.19 (21.74) | 23.19 (21.01) |

  | deepseek-8b | 1/3 | 2/3 | 1 |
  |---|---|---|---|
  | WW-AG | 37.04 (36.51) | 46.03 (38.62) | 45.50 (38.62) |
  | WW-HC | 29.89 (27.59) | 28.74 (28.74) | 28.74 (28.74) |
  | TE-Cap | 41.86 (37.21) | 42.64 (39.53) | 42.64 (40.31) |
  | TE-Mag | 31.88 (31.16) | 30.43 (30.43) | 30.43 (29.71) |

  Reading: SOAP is remarkably data-efficient — with a third of the reference corpus
  (6–12 trajectories) every cell is within ~4 points of its full-data number, and
  most are within 1–2. The visible jump is DeepSeek WW-AG 1/3→2/3 (37.0→46.0),
  where the small fit also changes the selected config (act/6, one component). The
  optimistic re-selection means small-fraction numbers can exceed full-data ones
  (DeepSeek WW-HC 1/3: 29.89; TE-Mag 1/3: 31.88) — selection noise on tiny fits,
  worth a caption note.

### A8 — Mean vs last-token pooling (appendix `tab:pooling`)  `[CPU]`  — DONE 2026-09-03

- [x] **Target.** Is mean pooling the right step representation? The extractor
  stored both poolings, so this is a re-selection on the `last` stores.
- **Procedure.** Per without-GT cell (both backbones x WW-AG/WW-HC/TE-Cap/TE-Mag),
  re-select the full config on last-token representations by the standard rule
  (dense base grid, then the backprop rescore grid) — A7's machinery at fraction 1
  with `POOLING="last"`. The `mean` rows are Table 1's selection rows.
- **Results** — `results-ablations/a8_pooling.tsv` (`scripts/ablations/a8_pooling.py`;
  merged from `a8_parts/`). Step acc %, SOAP (base in parentheses):

  | pooling | Qwen WW-AG | Qwen WW-HC | Qwen TE-Cap | Qwen TE-Mag | DS WW-AG | DS WW-HC | DS TE-Cap | DS TE-Mag |
  |---|---|---|---|---|---|---|---|---|
  | mean (= Table 1) | 47.62 (39.15) | 34.48 (33.33) | 35.66 (33.33) | 23.19 (21.01) | 45.50 (38.62) | 28.74 (28.74) | 42.64 (40.31) | 30.43 (29.71) |
  | last token | 29.63 (29.10) | 24.14 (22.99) | 25.58 (24.03) | 23.19 (21.01) | 30.69 (29.10) | 22.99 (22.99) | 24.81 (24.81) | 26.09 (24.64) |

  Reading: mean pooling wins everywhere but TE-Mag (tie on Qwen); the last-token
  base score loses 8-16 points on WW-AG, WW-HC and TE-Cap even at its own best
  configuration. Rescoring still lifts last-token scores where gamma>0, from a much
  lower base.

### A9 — Accuracy at k and MRR for SOAP, OAT, StepFinder (appendix `tab:ranked-step`, `tab:ranked-agent`)  `[CPU]`  — DONE 2026-09-03

- [x] **Target.** Ranked-shortlist quality for the three methods that score every
  step; the prompting judges emit one verdict and are excluded.
- **Procedure.** SOAP/base: re-score the anchor per cell (both backbones, all eleven
  subsets, both GT trees) and rank at k in {1,3,5} plus MRR (`mrr` added to
  `main/metrics.py`, loop and batch paths, tested). OAT/StepFinder: rank the
  per-step `scores`/`score_step_indices` stored in their prediction JSONs
  (`../attrib-prompting/outputs-rb-{nogt,gt}`), descending, earliest tie first; a
  gold step OAT never scored is a miss at every k; mean over the triple then over
  the five training seeds. Asserted: SOAP/base step@1 == selection tables;
  baseline step@1/agent@1 == B1 `by_cell.tsv`.
- **Results** — `results-ablations/a9_ranked.tsv` (`scripts/ablations/a9_ranked.py`;
  merged from `a9_parts/`); manuscript table `tables/appendix_ranked.tsv`.
  Without-GT step acc % at k=1/3/5:

  | qwen3.5-9b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 16.72/56.72/86.46 | 10.11/25.75/38.39 | 56.50/74.60/82.18 | 15.35/53.02/73.80 | 18.84/32.17/40.29 |
  | StepFinder | 15.87/55.24/75.87 | 13.33/29.66/34.48 | 30.03/59.04/68.58 | 18.29/37.21/50.70 | 6.67/17.10/28.12 |
  | base | 39.15/71.96/92.59 | 33.33/43.68/49.43 | 61.38/79.20/85.58 | 33.33/60.47/75.19 | 21.01/31.16/41.30 |
  | SOAP | 47.62/74.60/91.01 | 34.48/42.53/50.57 | 61.78/79.61/85.74 | 35.66/63.57/75.97 | 23.19/32.61/42.75 |

  | deepseek-8b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 12.70/45.19/83.92 | 15.86/37.01/42.30 | 52.50/74.08/83.17 | 17.98/47.13/75.04 | 13.04/31.01/41.45 |
  | StepFinder | 18.94/60.32/78.94 | 10.80/21.61/29.20 | 36.91/63.05/74.12 | 14.73/41.24/58.60 | 4.20/11.01/21.45 |
  | base | 38.62/72.49/91.53 | 28.74/42.53/59.77 | 64.19/79.96/87.01 | 40.31/58.14/75.97 | 29.71/34.78/47.83 |
  | SOAP | 45.50/76.19/93.12 | 28.74/42.53/59.77 | 64.68/80.30/87.41 | 42.64/62.02/78.29 | 30.43/39.13/44.20 |

  MRR (WW-AG, Qwen): OAT 41.46, StepFinder 41.23, base 58.83, SOAP 63.92.
  Reading: SOAP keeps its lead at every k; rescoring helps at k=1 and 3 and is
  within a point of base at k=5 (it moves the gold step to the top, not the tail).
  Agent@k saturates for everyone by k=3 (two- or three-agent trajectories).

### C1 — Inference cost per trajectory (appendix `tab:cost`, `app:compute`)  `[CPU; timing needs an idle GPU]`  — TOKENS DONE 2026-09-13, timing partial

- [x] **Target.** The appendix claims the baselines are compute-intensive; give the
  number: judge calls, prompt tokens, output tokens and wall-clock per trajectory for
  every prompt-based baseline against SOAP's one-forward-pass-per-step.
- **Scope (user decision 2026-09-13).** WW-AG and WW-HC only; SOAP on qwen3.5-9b;
  judges qwen3.5-9b (tokens + time) and gpt-4o (tokens only, no API re-runs); both GT
  settings in the TSV, the table shows without-GT. Methods: all_at_once,
  step_by_step, binary_search, correct, chief, errorprobe_paper.
- **Procedure.**
  1. Tokens, offline and exact: `../attrib-prompting/scripts/cost_report.py` drives each
     method's generator program with the stored `calls` responses, so it re-yields the
     very prompts the run issued; prompts are tokenized with the Qwen3.5-9B chat template
     (thinking off) or tiktoken's o200k encoding (cookbook per-message overhead), capped
     at the run's `truncate_prompt_tokens` (15,872 for ErrorProbe on the open judge). Every
     replay is checked against the stored prediction (`n_mismatch`). Six with-GT
     open-judge prompting cells were imported from a legacy JSONL with empty `calls`:
     All-at-Once is still costed (one prompt, no response needed), Step-by-Step and
     Binary Search are not (`n_costed = 0`).
     Output: `../attrib-prompting/reports/cost_ww.tsv`.
  2. Time: `scripts/ablations/c1_cost.py` harvests the predictors' own
     `wrote <dir> (N/N files, Xs)` lines from `../attrib-prompting/logs/open/*.log` and
     `logs/cost/*.log` (batched vLLM on one H200 -> throughput, not latency), computes
     SOAP's tokens as the sum over scoreable steps of the extractor's exact input length
     (`main.data.build_step_input`, 8,192 budget) and its time from
     `logs/c1_timing_<subset>.log` when a timed run exists, else from the July mtimes
     (10 min WW-AG, 2 h 15 min WW-HC). Output: `results-ablations/c1_cost.tsv`;
     `--print-table` emits the `tab:cost` rows.
  3. Still to run on an idle GPU (all eight busy 2026-09-13; commands in
     `../attrib-prompting/scripts/TODO.md`): ErrorProbe paper mode timing on the open
     judge (no `wrote` line exists), the six legacy with-GT prompting cells with call
     logs, and SOAP's timed extraction.
- **Reading the numbers.** Step-by-Step on the open judge batches EVERY step's prompt
  (`step_mode: batch`, the vendored control flow stops at the first "Yes" on API
  judges), so its open-judge call count is the trajectory length, not the early-stop
  count; the GPT-4o row shows the early-stop cost. Both are what was actually run.
- **Results** — see `results-ablations/c1_cost.tsv` (filled below when the replay
  finishes).

### C4 — Cost split into extraction and operation, setup and inference (appendix `app:compute`, `tab:cost-setup` / `tab:cost-inference` / `tab:cost-judges`)  `[GPU]`  — DONE 2026-09-15

- [x] **Target.** Replace C1's single table (batched judge throughput vs SOAP's July
  extraction mtimes) with a faithful, two-stage cost analysis: every representation-based
  method is timed through its OWN extraction code as run, one trajectory at a time on one
  H200, with peak GPU memory; then the operation on the stored vectors (training / SVD
  fit; scorer at inference) is timed apart. Judges are timed one trajectory at a time.
- **Extraction schemes, as run (user decision 2026-09-14: report the difference, do not
  argue single-pass equivalence).** OAT: one pass over the whole trajectory (cap 262,144,
  last layer). StepFinder: one pass per step over the step text alone (cap 8,192) plus
  one per distinct agent name, memoized. SOAP: two passes per scoreable step (activation
  stage + attention stage), each over the step in its context under the 8,192 budget.
- **Scripts / outputs.**
  * `scripts/ablations/c4_extract_cost.py` -> `results-ablations/c4_extract_cost/soap_qwen3.5-9b_<subset>.tsv`
    (per trajectory: passes, tokens, seconds, peak/reserved GB, both stages; all 126 + 58).
  * `../attrib-prompting/scripts/cost_rb_extract.py --method {oat,stepfinder}` ->
    `oat_qwen3.5-9b.tsv`, `stepfinder_qwen3.5-9b*.tsv` (inference on WW; setup = 103
    MCP-Atlas successes for OAT, the vendored regenerated failures for StepFinder,
    1,564 + 2,604; the hand-crafted corpus was timed in two shards, `--tag _hc-shard{1,2}`).
  * `scripts/ablations/c4_stage2_cost.py` -> `soap_stage2_qwen3.5-9b.tsv` (SVD fit,
    projection, rescoring; CPU and GPU; first seed of the triple, Table-1 config) and
    `soap_reference_stems_qwen3.5-9b.json` (the reference split, for the setup sums).
  * `../attrib-prompting/scripts/cost_rb_stage2.py` -> `rb_stage2_qwen3.5-9b.tsv`
    (OAT neural CDE + conformal decoder, StepFinder BiLSTM, in process, seed 42).
  * Training time: `../attrib-prompting/logs/cost/{oat,stepfinder}-train-timing.log`
    (5 seeds: OAT 255.5 s; StepFinder 259.0 s WW-AG corpus, 577.2 s WW-HC corpus).
  * Judges: `../attrib-prompting/reports/latency/` (`scripts/cost_latency_ww.sh`, one
    trajectory at a time on vLLM; sample = `ids_<subset>.json`, the 40 / 17 trajectories
    whose every step fits the budget; tokens replayed by `scripts/cost_report.py`).
  * `scripts/ablations/c4_cost_tables.py --print-table` joins everything into
    `results-ablations/c4_cost_{setup,inference,judges}.tsv` and prints the table bodies.
- **Results (Qwen3.5-9B, without GT).**
  * Setup extraction (passes / tokens / s / peak GB): OAT 103 / 507,727 / 59 / 19.3 (both
    subsets); StepFinder 24,393 / 2.26M / 3,154 / 15.1 (WW-AG) and 52,001 / 3.09M / 6,774 /
    15.0 (WW-HC); SOAP 619 / 1.08M / 204 / 23.8 (37 WW-AG reference trajectories) and
    1,482 / 8.81M / 1,784 / 23.8 (17 WW-HC). Operation: OAT 255.5 s, StepFinder 259.0 /
    577.2 s, SOAP SVD 0.15 / 0.30 s (CPU).
  * Inference extraction per trajectory (passes / tokens / s / peak GB mean, max): OAT
    1 / 3,499 / 0.42 / 15.6, 26.3 and 1 / 19,587 / 2.13 / 18.9, 31.6; StepFinder 8.8 / 2,813
    / 2.01 / 15.0, 16.5 and 39.2 / 17,028 / 7.41 / 15.4, 16.5; SOAP 16.4 / 30,764 / 5.54 /
    19.7, 23.8 and 101.0 / 623,352 / 160.6 / 23.2, 25.1. SOAP's peak is bounded by the
    budget (25.1 GB max on the 105-step trajectory); OAT's follows trajectory length.
  * Operation at inference (ms per trajectory, CPU / GPU): OAT 2.6 / 5.2 and 12.2 / 26.3;
    StepFinder 63.6 / 2.3 and 88.2 / 3.5; SOAP 0.7 / 2.6 and 5.4 / 15.8.
  * Judges, one trajectory at a time (s; calls; prompt tok; gen tok), WW-AG: AAO 2.0 / 1 /
    2,811 / 312; SBS (batch of one trajectory's steps) 2.0 / 8.6 / 16,729 / 992; BS 0.2;
    CORRECT 1.9; CHIEF 23.9 / 6 / 27,442 / 4,362; ErrorProbe 5.9 / 5 / 11,874 / 1,494;
    SOAP on the same 40: 5.4 s, 16.2 passes, 28,869 tokens, 0 generated. WW-HC: SOAP 15.3 s
    vs AAO 2.1, ErrorProbe 6.5, CHIEF 23.5.
  * Weights: SOAP loads the full checkpoint (17.5 GB), the baselines its text decoder
    (14.8 GB); stated in the table captions.
- **Manuscript.** `app:compute` rewritten 2026-09-15 (three tables; old subsection kept
  commented). The C1 table `tab:cost` is retired.
- **Revision 2026-09-17 (user request).** Three corrections to the reduced two-table
  version of `app:compute`:
  * **One-pass convention.** SOAP is charged ONE forward pass per trajectory, since
    causal attention makes one pass over the trajectory yield every step's states and
    attention; the per-step extractor re-reads the prefix as an implementation choice.
    Timed as the last step's activation pass (the whole trajectory in its context under
    the budget) by `scripts/ablations/c4_onepass_cost.py` ->
    `results-ablations/c4_extract_cost/soap_onepass_qwen3.5-9b.tsv` (pure forward, warm,
    3 repeats, idle H200): 0.38 s WW-AG / 0.76 s WW-HC over all trajectories (peak 23.7 /
    23.8 GB), 0.36 / 0.62 s on the in-budget judge sample. The old "s/forward pass"
    cells (0.34 / 1.59) pooled the per-step loop, whose WW-HC figure is inflated by
    re-tokenizing the context for every step (129-step trajectory: 99 s tokenizing vs
    105 s of forward passes; pure passes are 0.60-0.81 s each).
  * **StepFinder CPU scoring was an artifact.** 63.6 / 88.2 ms came from torch's lazy
    default thread pool on the 240-core host; pinning the pool (any of 1/4/16 threads)
    gives 1.8 / 5.0 ms for the same 278k-parameter BiLSTM (GPU 2.2 / 3.4 ms). Both
    stage-2 timers now take `--cpu-threads` (default 4) and were re-run with 10 repeats:
    OAT 2.4 / 10.4 ms, StepFinder 1.8 / 5.0, SOAP 0.7 / 5.5 (projection + rescoring);
    SVD fit 0.08 / 0.29 s. Old TSVs kept as `*.2026-09-15.bak.tsv`.
  * **`tab:cost-judges` = one pass + scoring per trajectory** for SOAP (0.36 / 0.62 s)
    against every judge request per trajectory for the prompt-based methods (unchanged).
    `c4_cost_tables.py` now emits `one_pass_s`, `scoring_ms`, `one_pass_plus_scoring_s`
    and reads the one-pass TSV; `--print-table` prints the manuscript layout.

## Baselines

### B1 — Representation-based baselines: OAT and StepFinder (`tab:main`, `tab:main-gt`)  `[CPU; predictions from ../attrib-prompting]`  — DONE 2026-08-23

- [x] **Target.** Fill the OAT and StepFinder rows of Tables 1–2 — the two baselines
  that, like SOAP, read a model's internal vectors instead of prompting a judge.
- **Scope — wider than the ablations, like A2.** All five subsets (CE included), both
  backbones, both GT settings, on the frozen triples. Nothing is trained or re-run
  here: the predictions (one JSON per trajectory, the prompting baselines' schema)
  live in `../attrib-prompting/outputs-rb-nogt/` (without GT) and `outputs-rb-gt/`
  (with GT), code in `../attrib-prompting/baselines-rp/`. This runner only scores them
  on SOAP's test splits with the rules of `scripts/prompting/evaluate.py` (same split
  code, missing prediction = wrong), so the cells are comparable with every other row.
- **OAT** (one-class tracer; paper: Yeh et al., arXiv:2607.12747, vendored code).
  Trains on SUCCESSFUL trajectories only — 103 MCP-Atlas tool-use runs shipped with
  the paper, none from our corpora — so it sees no error label and no trajectory from
  the test distribution. One model per (extractor, training seed), seeds 42–46; the
  same model scores all twelve subsets. Per subset it adapts its latent statistics to
  ALL trajectories of that subset by CORAL (unsupervised; test included — same kind of
  transductive use as fitting nothing on labels, but note it). Extractor = the
  backbone's last layer, mean-pooled; the paper's own extractor (Qwen3.5-27B) was not
  run. Prediction = the step with the largest anomaly distance; step 0 (the user turn)
  is never scored. Marks: Actual data ✗, Supervised ✗. Known deviations from the
  paper: LR 1e-4 (paper 4e-5); trajectories whose gold step is the filtered step 0 are
  kept and counted wrong (~45 across the corpora), not dropped.
- **StepFinder** (Zhu et al., vendored code) — a supervised step classifier over
  per-step embeddings with a position prior. Its faithful encoder is
  **Qwen3-Embedding-0.6B** (the paper's); the Qwen3.5-9B / DeepSeek-8B runs take the
  first 128 coordinates of the decoder's last hidden state, which the port documents
  as an arbitrary slice for a decoder (principled only for an MRL embedder). Two
  families, scored separately:
  - **Family A — regenerated corpus (`stepfinder.s42–s46`)**, the paper's own
    protocol: train on the LLM-regenerated, step-labelled failures released with the
    paper (1,564 Alg-Gen-style + 2,604 Hand-Crafted-style trajectories; subset →
    training corpus mapped as in the paper), checkpoint selected on a held-out val
    split of that corpus (the paper selects on its test set; the port refuses to
    write such predictions). One model per training seed 42–46, predictions for every
    trajectory. Marks: Actual data ✗ (regenerated), Supervised ✓. Leakage: the
    regenerated corpus shares TASKS with TE-Cap (22/85), TE-Mag (51/91) and CE-gaia
    (30/50); WW is clean. Per-record flag `train_task_overlap`.
  - **Family B — in-corpus (`stepfinder.e<seed>`)**, SOAP's own protocol: for split
    seed `s`, train on that seed's 30 % train partition of the subset under test
    (same cut as SOAP's reference split), select on its val partition, predict its
    val+test. Encoder Qwen3-Embedding-0.6B, without-GT only. Trained for seeds 1–20
    on WW and TE, 1–3 on CE — so it meets the frozen triple on WW-AG, WW-HC and
    TE-Mag only; TE-Cap (22–24) and CE (17–19) would need new training runs. Marks:
    Actual data ✓, Supervised ✓.
- **With-GT setting.** Models are shared across settings; only the inputs change. OAT
  appends the gold answer to the question row (left context of every step);
  StepFinder appends it to the first step's content. Family B is not run with GT.
- **Aggregation.** A cell is the mean over the triple's three test splits, as
  everywhere; OAT and family A then average over their five training seeds (the
  per-training-seed values and the best-of-five stay in the TSV — best-of-five is
  test-selected and runs 2–13 points higher, e.g. StepFinder/Qwen CE 43.53). Family B
  is a diagonal: seed `s`'s model on seed `s`'s test split.
- **Results** — `results-ablations/b1_rb_baselines/` (`scripts/ablations/
  b1_rb_baselines.py`; `by_seed` / `by_cell` / `by_column` per training seed,
  `by_column_mean_over_train_seeds` with min/max, `family_b`; 0 missing predictions
  in every Qwen/DeepSeek cell; CE macro-averages its 7 subsets). Step acc %, mean
  over triple × 5 training seeds; the SOAP rows repeat Tables 1–2 for reference.
  **Reported in the manuscript: OAT and StepFinder family A, on the Qwen3.5-9B and
  DeepSeek-8B strands** — the embedding-encoder and family-B rows are for reference:

  **Without GT**

  | qwen3.5-9b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 16.72 | 10.11 | 56.50 | 15.35 | 18.84 |
  | StepFinder (A, regenerated) | 15.87 | 13.33 | 30.03 | 18.29 | 6.67 |
  | SOAP | 47.62 | 34.48 | 61.78 | 35.66 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 12.70 | 15.86 | 52.50 | 17.98 | 13.04 |
  | StepFinder (A, regenerated) | 18.94 | 10.80 | 36.91 | 14.73 | 4.20 |
  | SOAP | 45.50 | 28.74 | 64.68 | 42.64 | 30.43 |

  | qwen3-embedding-0.6b (StepFinder's own encoder) | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | StepFinder (A, regenerated) | 24.02 | 11.72 | 32.75 | 17.98 | 21.16 |
  | StepFinder (B, in-corpus, frozen triple) | 29.63 | 12.64 | — | — | 7.97 |
  | StepFinder (B, in-corpus, all trained seeds) | 29.29 (20) | 15.00 (20) | 56.53 (3) | 16.74 (20) | 8.70 (20) |

  (Family B's last row is the port's own diagonal mean over every trained seed,
  count in parentheses — NOT the frozen triple; CE there is the macro-mean over seeds
  1–3, per subset: arc 80.70, gaia 56.00, hotpot 50.40, math500 65.40, mmlu_pro 70.29,
  musique 48.50, wikimqa 24.43.)

  **With GT** (Table 2 shows Qwen; DeepSeek goes to the appendix, whose with-GT
  table is still the old placeholder layout)

  | qwen3.5-9b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 22.12 | 8.05 | 56.21 | 17.98 | 20.00 |
  | StepFinder (A, regenerated) | 18.52 | 12.87 | 28.52 | 13.64 | 8.99 |
  | SOAP | 43.39 | 34.48 | 60.59 | 36.43 | 21.01 |

  | deepseek-8b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT | 20.85 | 17.01 | 52.39 | 18.14 | 13.77 |
  | StepFinder (A, regenerated) | 18.94 | 9.89 | 24.99 | 15.04 | 6.23 |
  | SOAP | 44.97 | 29.89 | 65.19 | 42.64 | 36.96 |

  | qwen3-embedding-0.6b | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | StepFinder (A, regenerated) | 21.80 | 10.34 | 30.06 | 15.04 | 17.25 |

  **Agent acc %, without GT** (with-GT mirror in the TSV):

  | | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | OAT / qwen3.5-9b | 40.53 | 53.10 | 70.58 | 46.67 | 62.03 |
  | OAT / deepseek-8b | 42.33 | 56.32 | 69.04 | 50.08 | 63.91 |
  | StepFinder A / qwen3.5-9b | 39.26 | 37.01 | 60.03 | 47.75 | 57.83 |
  | StepFinder A / deepseek-8b | 48.99 | 41.15 | 67.29 | 45.12 | 42.61 |
  | StepFinder A / qwen3-embedding-0.6b | 46.98 | 44.83 | 61.70 | 49.30 | 70.87 |
  | StepFinder B / qwen3-embedding-0.6b | 49.74 | 50.57 | — | — | 57.97 |

  Reading: neither baseline comes near SOAP on the two backbones — on WW-AG the gap
  is ~30 points, on TE-Cap 17–25, on WW-HC 13–21. CE is the one column where OAT is
  competitive (56.5 vs 61.8 on Qwen): its trajectories are short, so a short-corpus
  prior goes a long way (A1's perplexity/random rows tell the same story). StepFinder
  is strongest on its own encoder and on in-corpus training (family B on WW-AG 29.6,
  the best non-SOAP representation number), which says its regenerated corpus
  transfers poorly to these systems — and family B is supervised on the very
  partition SOAP uses unlabeled. Rescoring in the with-GT setting moves OAT up on
  WW-AG (+5 on Qwen, +8 on DeepSeek) and StepFinder down or sideways (mean Δ −1.6 in
  the port's own census), the opposite sign to the prompting baselines.
- **Caveats to carry into the captions.** (i) Mean over five training seeds, not
  best. (ii) OAT's extractor is a 9B/8B proxy, not the paper's 27B; the paper's
  per-step annotation (every contributing step) differs from ours (one decisive
  step), so its printed numbers are not comparable. (iii) StepFinder's decoder
  runs use an arbitrary 128-dim slice; the faithful encoder is the embedding model,
  which has no SOAP counterpart in the table strands. (iv) Family A's training corpus
  overlaps TE and CE-gaia tasks. (v) StepFinder's published numbers select the
  checkpoint on test (+6–10 points); these select on val.
- **Manuscript edits — APPLIED 2026-08-23.** Tables 1–2: OAT/StepFinder dashes filled
  with family A on the Qwen3.5-9B and DeepSeek-8B strands (Table 2: Qwen only);
  StepFinder's marks corrected to Actual data ✗, Supervised ✓; the stray `x\`` at
  the Table-2 OAT row removed; provenance comments point at
  `results-ablations/b1_rb_baselines/`. Still open: the StepFinder description at
  the `\TODO` in Setup; the embedding-encoder StepFinder row (own strand or
  footnote); the with-GT DeepSeek rows for Appendix `app:gt`, whose table is still
  the old placeholder layout.

### B2 — Prompt-based baselines on the open backbones (`tab:main`)  `[CPU; predictions from ../attrib-prompting]`  — DONE 2026-09-03

- [x] **Target.** Fill the prompting group of Table 1's Qwen3.5-9B and
  DeepSeek-R1-Distill-Llama-8B blocks: the six prompt-based methods with the
  backbone itself as the judge, so every row of a block runs on one model.
- **Scope.** All five subsets, both open judges, without-GT. Nothing runs here:
  the predictions live in `../attrib-prompting/outputs-nogt/` (judges
  `qwen3.5-9b`, `deepseek-8b`, one JSON per trajectory, the prompting schema);
  `scripts/prompting/evaluate.py` scores them on the frozen triples with its
  usual rules (SOAP's test splits, missing prediction = wrong) into
  `results-prompting/`, and `scripts/tables/open_backbone_rows.py` formats the
  twelve rows into `tables/open_backbones_{step,agent}_without_gt.tsv`.
- **With-GT coverage is partial.** The open judges carry with-GT predictions only
  for All-at-Once / Step-by-Step / Binary Search on four columns (no CE), so
  `tab:main-gt` gains no open-judge prompting rows yet; the scored partial cells
  sit in `results-prompting/by_column.tsv` (`with_gt=True`).
- **Results** — step acc %, mean over the triple (full precision in
  `results-prompting/by_column.tsv`; SOAP rows repeat Table 1 for reference):

  | qwen3.5-9b judge | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | All-at-Once | 17.46 | 11.49 | 33.47 | 4.65 | 1.45 |
  | Step-by-Step | 10.58 | 16.09 | 31.15 | 14.73 | 10.87 |
  | Binary Search | 2.65 | 13.79 | 59.92 | 0.78 | 5.07 |
  | CORRECT | 29.10 | 3.45 | 36.35 | 7.75 | 2.90 |
  | CHIEF | 28.57 | 3.45 | 29.39 | 3.88 | 2.17 |
  | RAFFLES | 46.03 | 27.59 | 73.14 | 29.46 | 23.91 |
  | SOAP | 47.62 | 34.48 | 61.78 | 35.66 | 23.19 |

  | deepseek-8b judge | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | All-at-Once | 7.41 | 5.75 | 21.95 | 4.65 | 2.90 |
  | Step-by-Step | 16.93 | 9.20 | 9.04 | 11.63 | 4.35 |
  | Binary Search | 7.94 | 12.64 | 34.15 | 10.08 | 13.77 |
  | CORRECT | 20.11 | 11.49 | 32.82 | 3.88 | 0.72 |
  | CHIEF | 17.46 | 2.30 | 8.40 | 7.75 | 9.42 |
  | RAFFLES | 28.04 | 8.05 | 35.84 | 20.16 | 10.87 |
  | SOAP | 45.50 | 28.74 | 64.68 | 42.64 | 30.43 |

  **Agent acc %, without GT** (`tables/open_backbones_agent_without_gt.tsv`):

  | qwen3.5-9b judge | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | All-at-Once | 56.61 | 48.28 | 81.40 | 62.79 | 62.32 |
  | Step-by-Step | 19.05 | 47.13 | 35.85 | 51.16 | 65.94 |
  | Binary Search | 38.10 | 62.07 | 70.70 | 22.48 | 59.42 |
  | CORRECT | 59.79 | 47.13 | 78.60 | 62.02 | 74.64 |
  | CHIEF | 50.26 | 54.02 | 74.81 | 65.12 | 74.64 |
  | RAFFLES | 60.85 | 60.92 | 77.27 | 65.89 | 68.84 |

  | deepseek-8b judge | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | All-at-Once | 42.33 | 45.98 | 38.04 | 54.26 | 63.04 |
  | Step-by-Step | 44.97 | 37.93 | 15.23 | 33.33 | 38.41 |
  | Binary Search | 28.57 | 48.28 | 53.37 | 44.96 | 69.57 |
  | CORRECT | 44.97 | 32.18 | 58.67 | 38.76 | 60.87 |
  | CHIEF | 37.04 | 52.87 | 56.71 | 51.16 | 63.77 |
  | RAFFLES | 38.10 | 51.72 | 41.01 | 54.26 | 65.94 |

  Reading: RAFFLES is the only prompting method that competes — it overtakes SOAP
  on two of the ten step cells, Qwen CE (73.14 vs 61.78) and Qwen TE-Mag (23.91
  vs 23.19), and lands within 1.6 points on Qwen WW-AG; SOAP keeps every DeepSeek
  cell by wide margins (min gap 16.7, WW-AG). Every other method sits far below
  its GPT-4o strand, so judge capacity, not the protocol, carries them.
- **Manuscript edits — APPLIED 2026-09-03.** Table 1: the twelve `\PH` prompting
  rows filled; best/second markers recomputed within each open-backbone block —
  RAFFLES takes best on CE and TE-Mag in the Qwen block (SOAP drops to second
  there), and the training-based rows lose the seconds they held while the
  prompting group was empty. `tab:main-gt` untouched (partial coverage, above).

### B3 — ErrorProbe on the open backbones (`tab:main`, `tab:main-gt`)  `[CPU; predictions from ../attrib-prompting]`  — DONE 2026-09-06

- [x] **Target.** The ErrorProbe~\citep{errorprobe} row of Tables 1–2 (Li et al.,
  Findings of ACL 2026; reproduced in `../attrib-prompting/baselines/errorprobe/`,
  modes documented in its README and IMPLEMENTATION.md).
- **Three modes, one reported.** `errorprobe` (truncated-history, the vendored
  default: an Analyzer reads the last 15 turns, a Verifier reviews without veto),
  `errorprobe_bt` (the vendored backward walk), and `errorprobe_paper` (the paper's
  own pipeline, rebuilt: MAST tagging -> dependency graph + backward trace ->
  Strategist/Investigator/Arbiter; probes are prompts, no code runs, no memory).
  The manuscript row reported the truncated mode from 2026-09-05; **since
  2026-09-06 it reports the paper mode**, which is stronger nearly everywhere on
  Qwen and on DeepSeek WW-AG/WW-HC/TE-Cap, weaker on Qwen WW-HC (12.64 vs 18.39)
  and DeepSeek TE-Mag (8.70 vs 17.39).
- **Scoring.** Same path as B2: predictions in `../attrib-prompting/outputs-nogt/`
  (without GT) and `outputs/` (with GT); `scripts/prompting/evaluate.py` (method
  list now includes `errorprobe_paper`) scores them on the frozen triples into
  `results-prompting/`; `scripts/tables/open_backbone_rows.py` adds the
  "ErrorProbe (paper)" row to `tables/open_backbones_*.tsv`. Zero missing and zero
  null-step predictions in every paper-mode cell. With-GT coverage: WW + TE only
  (no CE), both judges, all three modes.
- **Results** — step acc %, mean over the triple; SOAP repeats Table 1:

  **Without GT, paper mode (the reported row)**

  | | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | qwen3.5-9b | 42.33 | 12.64 | 62.42 | 29.46 | 25.36 |
  | SOAP (qwen) | 47.62 | 34.48 | 61.78 | 35.66 | 23.19 |
  | deepseek-8b | 24.34 | 10.34 | 35.97 | 17.05 | 8.70 |
  | SOAP (deepseek) | 45.50 | 28.74 | 64.68 | 42.64 | 30.43 |

  **Without GT, the unreported modes** (kept in .tex comments)

  | | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | truncated / qwen | 20.63 | 18.39 | 53.40 | 21.71 | 21.74 |
  | truncated / deepseek | 14.29 | 6.90 | 36.25 | 13.18 | 17.39 |
  | backward / qwen | 5.82 | 11.49 | — | 3.10 | 7.25 |
  | backward / deepseek | 1.06 | 13.79 | — | 0.00 | 3.62 |

  **With GT, paper mode** (Table 2 shows Qwen WW only)

  | | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | qwen3.5-9b | 43.92 | 16.09 | 31.78 | 23.19 |
  | deepseek-8b | 18.52 | 14.94 | 21.71 | 11.59 |

  Agent accuracies sit in `tables/open_backbones_agent_*.tsv` and
  `results-prompting/by_column.tsv`.
- **Ranking changes the swap caused.** Table 1, Qwen block: ErrorProbe takes
  second on CE from SOAP (62.42 vs 61.78) and best on TE-Mag (25.36 > RAFFLES
  23.91 > SOAP 23.19); on TE-Cap it ties RAFFLES exactly (38/129 = 29.46), both
  underlined. DeepSeek block: TE-Mag second returns to Binary Search (13.77).
  Table 2: WW-AG now reads RAFFLES 46.56 best, ErrorProbe 43.92 second, SOAP
  43.39 third.
- **Manuscript edits — APPLIED 2026-09-06.** `tab:main` (both open blocks) and
  `tab:main-gt`: ErrorProbe rows now carry the paper mode, markers recomputed as
  above; truncated/backward numbers preserved as dated comments; GPT-4o row stays
  `\PH` (no GPT-4o ErrorProbe run exists). **Prose now stale, NOT edited**: the
  tab:main-gt paragraph still calls SOAP "second best on WW-AG, where only
  RAFFLES surpasses it"; the tab:main paragraph still claims SOAP is best "on all
  five subsets with both backbones" (already false since B2's RAFFLES fill);
  the Baselines paragraph still describes ErrorProbe as an "analyzer--verifier
  diagnoser", which names the truncated mode, not the reported paper pipeline.
  Appendix (std/agent tables, `tab:gt-full`, GPT-5, `app:baselines` prose) still
  lacks ErrorProbe entirely, as before.
- **GPT-4o landed — 2026-09-07.** The paper-mode run now covers the GPT-4o
  judge too (full decode budget; still no truncated/backward GPT-4o runs):
  without GT all five subsets, with GT WW + TE (no CE). Scored the same way;
  zero missing and zero null-step predictions. Step acc %:

  | gpt-4o, paper mode | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | without GT | 38.62 | 21.84 | 59.52 | 27.13 | 28.26 |
  | with GT | 34.92 | 28.74 | — | 30.23 | 26.81 |

  Manuscript edit — APPLIED 2026-09-07: `tab:main`'s GPT-block ErrorProbe row
  filled (was `\PH`). With RAFFLES dropped from the comparison (2026-09-06),
  ErrorProbe is best on all five columns of that block, so the block now
  carries full markers (seconds: CHIEF WW-AG/WW-HC, Step-by-Step CE,
  All-at-Once TE-Cap, CORRECT TE-Mag), computed against the PRINTED rows — the
  kept pre-reconciliation numbers; see the RAFFLES discrepancy note in
  `documents/main.md`. `tab:main-gt` untouched (a Qwen-judge table); the with-GT
  GPT-4o cells await `app:gt`. Newly stale prose: the main-comparison
  paragraph's example "47.62 vs 35.98 for RAFFLES" now cites a dropped row —
  the strongest GPT prompt-based method is ErrorProbe at 38.62.

## Datasets

### D1 — AgenTracer TracerTraj-code (`agentracer/code`)  `[GPU]`  — NUMBERS DONE 2026-09-07; FOUND TO BE AN ARTIFACT 2026-09-10 — do not report

- [x] **Target.** A fourth benchmark: the code split of AgenTracer's TracerTraj test
  set (Zhang et al., "AgenTracer: Who Is Inducing Failure in the LLM Agentic
  Systems?"). Numbers first, wiring later: `results-nogt/agentracer/` and
  `results-gt/agentracer/` are populated and verified; nothing reaches `tables/`
  or the manuscript yet, and the manuscript label is undecided (`AT-Code` is the
  candidate; note `make_main_tables.py` already reserves an `AgenTracer` ROW for
  the AgenTracer-8B *method*, so the dataset needs a distinct name).
- **Data.** 127 trajectories, 2,642 steps (mean 20.8, min 2, max 151), from a
  MetaGPT-style software team (Team Leader, Product Manager, Architect, Engineer,
  Data Analyst). Downloaded from `github.com/bingreeky/AgenTracer/data/
  tracertraj-code-test.parquet` and converted by `scripts/build_agentracer.py`
  (`name` -> `role`, per-turn `step` counter dropped, labels copied verbatim and
  asserted: `history[mistake_step]["role"] == mistake_agent` for 127/127).
  `data/agentracer/_provenance.json` records the rest. Two properties matter:
  the **errors are injected** (one agent turn perturbed per trajectory) rather than
  naturally occurring, and **only the test split is public** — the repo ships no
  train parquet — so the reference split comes from these 127, like every other
  subset. Three of 127 are degenerate: two trajectories have 2 steps, one has its
  error at step 0 (no predecessors, so rescoring cannot lift it).
- **Scope.** Both backbones, both GT settings, SOAP only (base score plus the
  three strategies). No prompting baselines, no API spend. The with-GT block here
  is the full problem statement plus a reference Python solution: 456 tokens mean,
  1,123 max on the Qwen3.5 tokenizer, against ~83 for WW-AG and TE-Cap. Because
  the block is pinned, truncation drops real predecessor turns to fit it.
- **Splits.** 30/20/50 -> 37 / 26 / 64 trajectories; test accuracy is quantized to
  1/64 = 1.6 points. Triples chosen by the standard sweep — 48 triples (seeds 1-50),
  `pick_triple.py`, frozen by `sync_seeds.py --write`; `--rule sum` and `--rule
  sum-diff` agree — NOT hand-picked, unlike the five reported subsets:

  | Arm | Triple | margin over runner-up | runners-up |
  |---|---|---|---|
  | without GT | 20, 21, 22 | 0.021 | 47, 16, 46 |
  | with GT    | 15, 16, 17 | 0.010 | 46, 16, 47 |

  Worst-to-best across the 48 triples spans 0.59 -> 0.85 (summed backprop step
  accuracy over both backbones), so the seeds matter as much as on the other
  datasets; read the margins as "a pick", not "the pick".
- **Procedure.** `python -m main extract` for both configs (no `outputs/` tree
  exists to seed from) -> `scripts/check_gt_parity.py --dataset agentracer --roots
  results` (new flag; passes file-for-file and step-for-step) ->
  `scripts/main/sweep_triples.py --datasets agentracer --plan agentracer:1-50`
  (96 units) -> `collect.py --force` -> `pick_triple.py` -> `sync_seeds.py --write`
  -> `main sweep` + `select` + `reproduce --row backprop` on both configs ->
  `check_sweep_repro.py` (48/48 cells reproduce).
- **Cost.** Extraction 4 GPU-hours on H200s: DeepSeek-8B 38 min per arm, Qwen3.5-9B
  82 min per arm, run four-way in parallel; 4 GB on disk. Triple sweep ~1 h on four
  GPUs with 12 workers (three per GPU triple the per-unit time; `UNIT_COST` says
  150 s for one worker per GPU). Reported run: minutes.
- **Results** — step@1 on test, mean over the frozen triple (per-seed sd for
  backprop in brackets), `results-{nogt,gt}/agentracer/select/selection.tsv`:

  | Arm | Backbone | SVD (base) | backprop | succ-strong | succ-near |
  |---|---|---|---|---|---|
  | without GT | Qwen3.5-9B   | 33.33 | **42.71** (sd 2.7) | 41.15 | 40.10 |
  | without GT | DeepSeek-8B  | 38.54 | 42.71 (sd 3.2) | **43.75** | 43.23 |
  | with GT    | Qwen3.5-9B   | 33.33 | **45.31** (sd 3.8) | 44.79 | 44.79 |
  | with GT    | DeepSeek-8B  | 35.42 | **43.75** (sd 5.1) | 43.75 | 43.23 |

  Rescoring lifts the base score by 4-12 points in every cell, and the with-GT
  arm adds 1-3 more. Random step@1 is ~4.8 % (1/20.8). Agent@1 is 81-87 %, but
  the Engineer owns 92 of the 127 decisive errors, so always-guess-Engineer already
  scores ~72 %; step@1 is the only informative headline here.
- **2026-09-10 audit — the SOAP number is a structural artifact, not error detection.**
  Prompted by every baseline in `../attrib-prompting` (`tracertraj`, byte-aligned with
  this corpus) scoring near random on the same frozen splits (best: CORRECT/Qwen 16.7,
  RAFFLES 12.5; SOAP 42.7). Findings, all on the no-GT frozen test splits:
  1. **58 % of gold turns (74/127) are scaffold dumps** — the MetaGPT prompt template
     (`# Past Experience / # Tool State / # Current Plan`, "No Plan", "Open file:
     None") with no code and no decision — while `mistake_reason` describes a code
     change that appears in the NEXT turn (`history[L+1]`, same agent in 122/127).
     The label sits on the prompt turn preceding the faulty action. In 50/127 that
     turn is also the trajectory's first handoff away from the Team Leader.
  2. **SOAP hits only those.** Qwen backprop: 68.5 % on scaffold-gold, **0.0 %** on
     action-gold; DeepSeek 64.7 / 6.8. SOAP/Qwen predicts a scaffold turn 99.5 % of
     the time, an agent's first turn 92 %, the first handoff 86 %. The zero-parameter
     rule "predict the first handoff turn" scores **41.7 %**, equal to SOAP's 42.7.
     This is why `embed` was selected: a new role name and an empty template are a
     lexical novelty, visible before any attention.
  3. **The baselines localise the faulty action and are penalised for it.** Under
     action-aligned labels (gold+1 when gold is a scaffold): RAFFLES/Qwen 54.7,
     ErrorProbe/Qwen 38.5, RAFFLES/DeepSeek 37.5 — SOAP 0.0 / 3.1. Within ±1 step,
     RAFFLES/Qwen 58.9 vs SOAP 52.1. The "one step late" pattern noted on 09-10 is the
     judges being right about the content and the label being on the template turn.
  4. Baseline evaluation artifact, agent level only: judges answer with MetaGPT
     persona names (Alex = Engineer, Alice = Product Manager, Mike = Team Leader,
     Bob = Architect, David = Data Analyst) and `_resolve_agent` cannot map them to
     roles; DeepSeek All-at-Once agent@1 8.7 -> 50.4 with the map. step@1 unaffected.
     Not run for this corpus: GPT-4o/GPT-5 judges, `errorprobe_paper`, the weak judge.
  5. **Root cause, from AgenTracer's own pipeline (2026-09-10; `MetaGPT/Who_When_Data_
     Pipeline/attack_monitor.py`, `universal_framework.py:1046-1068`).** The failure is
     a PROMPT INJECTION: at the first "think step" after a detected pre-attack turn, a
     "CRITICAL SECURITY ALERT ... REQUIRED IMPLEMENTATION" block is prepended to the
     agent's prompt (`PromptInjectionInterceptor._modify_prompt`); `mistake_step` is
     that step number and `mistake_agent` the name logged at it. The injected text is
     NOT kept in the stored history (markers in 1/127 trajectories, none at the label),
     so the labelled turn's content is the ORIGINAL prompt-template dump; the agent's
     compliant, faulty response lands in the following entry. `scripts/
     build_agentracer.py` is faithful — 2,642/2,642 turns identical to the parquet,
     `step` == list index throughout — the offset is the benchmark's convention.
  Verdict: unusable as a SOAP benchmark under its shipped labels. If it is ever used,
  relabel to the action turn first (then SOAP must be re-selected from scratch — its
  current configs encode the artifact) and treat the sweep below as void. The
  numbers that follow are kept as the record of what was measured.
- **Three things to know before reporting these** (written 2026-09-07, before the audit).
  1. **The selected position is `embed` in 3 of 4 cells** (the token-embedding
     layer, before any attention), with DeepSeek without-GT on `act/29`. On the
     five reported subsets the picks are mid-to-late layers. An injected error is
     a surface edit of one turn, and a bag-of-embeddings signal is enough to see
     it — consistent with the injected-error caveat, and a reason not to read
     this corpus as a peer of Who&When.
  2. **Validation accuracy sits 10-15 points below test** (e.g. Qwen without-GT:
     test 42.7, val 30.8). Test selection is the protocol everywhere, but the gap
     is wider here than on the reported subsets, so a switch to val-selection
     would move these cells more than the others.
  3. Selected rescoring configs: without GT Qwen `embed[9,18)`, layers 2-4,
     gamma 1.0, w 2; DeepSeek `act/29[4,7)`, layers 24-32, gamma 0.8, w all.
     With GT Qwen `embed[6,15)`, layers 2-4, gamma 1.0, w 2; DeepSeek
     `embed[7,20)`, layers 16-24, gamma 0.8, w 1.
- **Wiring, when wanted** (out of scope now, all hardcoded lists): `COLUMNS` in
  `scripts/tables/make_main_tables.py`, `make_appendix_tables.py` (+ its `ds_of`
  map), `open_backbone_rows.py`, `dataset_stats.py`; `scripts/prompting/
  evaluate.py` `DATASETS`/`COLUMNS` and `verify.py` `CORPUS`; the sibling repo's
  corpus copy and per-baseline configs if prompting baselines are ever run; then
  `tab:main`/`tab:main-gt` and the appendix mirrors by hand.

## Deferred (in the plan, blocked on inputs)

- (none — E2 unblocked 2026-08-24: the synthetic corpora landed in
  `../datagen/data/synthetic/`; concrete setup above.)

## Tracked outside this plan

- **Scalability** (`fig:scale`): now planned as S1 above (2026-08-27). The
  baseline side landed 2026-08-24 — OAT and StepFinder on Qwen3.5-4B/9B, Qwen3-14B,
  Qwen3.5-27B, WW only, both GT settings; the SOAP side (14B/27B extraction) is
  the open GPU item.
- **Baseline rows**: AgenTracer, GraphTracer (dashes in Tables 1–2). OAT and
  StepFinder are scored under B1; RAFFLES landed with the prompting rows.
  AgenTracer's own test data is now public and staged as `data/agentracer/`
  (D1); the AgenTracer-8B tracer itself is still not run.
- **With-GT SOAP adaptation**: announced in Setup, not yet described or planned here.

## Execution order

1. **Free reads** — A4, A6(a′), A6(b) — DONE 2026-08-17.
2. **CPU batch** — A5, A3, A6(a), E1, A7 — DONE 2026-08-17.
3. **GPU batch** — A1 — DONE 2026-08-17.
4. **E2** — DONE 2026-08-25 (stage → real-row assertion → extraction → fit →
   re-select → score; results above).
5. **S1** — DONE 2026-08-28 (extract 14B ∥ 27B into `results-nogt/ww/` → sweep
   + `select --force` → merged with the prefilled baseline rows into
   `results-ablations/s1_scale.tsv`; figure and manuscript edits still pending).
6. **D1** — NUMBERS DONE 2026-09-07 (extract both arms → parity → 96-unit triple
   sweep → pick + freeze → reported run; no table or manuscript wiring).

Environment note: the venv's torchvision/torchaudio were compiled against a different
torch and crashed transformers' lazy imports; `a1_scorefn.py --stage nll` blocks both
modules before importing (`sys.modules[...] = None`). FIXED 2026-09-07: torchvision
-> 0.28.0+cu130, torchaudio uninstalled (no cu130 build exists for torch 2.13; the
cu130 index would downgrade torch to get one). `main extract` now runs unmodified;
the a1 workaround is harmless and can stay.

## Manuscript edits this plan implies — APPLIED 2026-08-18 on user request

All edits below are now in `manuscript/sections/experiments.tex` (v2.2 header note
there lists them); old numbers stay as dated comments, and every pre-existing
comment / blue note / `\note{}` survives. Figure PDFs (`fig:transfer`, `fig:gamma`,
`fig:layers`, `fig:datasize`) are still placeholders — the TSVs to plot from are
named in each placeholder.

- Main tables: the "SOAP (w/o rescoring)" row moves into `tab:weights`, which widens
  to all five subsets and gains a base-score row (see A2).
- `tab:scorefn`: gains L1 and L2 rows; keeps random subspace; no uncertainty row;
  orientation sentence changes from "selected on validation" to the fixed
  orientations of A1.
- `tab:position`: temporal bias becomes two rows (z-scored / raw).
- `tab:attnsel`: w column widens to {1, 2, 3, 4, 5, all}.
- `fig:gamma`: γ grid densifies to 0.1 steps.
- `fig:transfer`: 5×5 → 4×4 (CE dropped), one grid per backbone.
- `tab:synth`: five subsets → WW-AG + WW-HC; generator rows renamed to Qwen3.5-9B and
  GPT-4o.
- Ablation anchor numbers and `tab:synth`'s stale "real" row must be refreshed to the
  current Table-1 protocol once the runs land.

## Manuscript presentation — APPLIED 2026-08-24

The ablations now follow the REDE layout (`artifacts/rede.pdf`): a 2x4 bar strip
`fig:sensitivity` (γ, w, representation layer, attention band; dashed orange line =
base score, dark bar = selected config) replaces `tab:gamma` / `tab:layers` /
`tab:attnsel`; `fig:transfer` (test-selected 4x4 heatmap + reference-data lines)
replaces `tab:transfer` / `tab:datasize`; `tab:scorefn` / `tab:weights` restyled
(bold best, shaded ours row, row groups). Retired tables stay in comments. Mirrors
(DeepSeek, TE, val-selected grids, full data-size table) live in
`manuscript/sections/appendix_ablations.tex` (`app:deepseek-ablations`, wired into
`main.tex`). Figures: `scripts/ablations/plot_figures.py` → `artifacts/ablations/`
→ `manuscript/assets/`; `--print-tables` dumps the hand-typed table bodies. No
number changed; the val-selected grids carry the Table-1 diagonal per E1's rule.

## Manuscript layout pass — DONE 2026-08-25

Compiled locally with tectonic (downloaded to the scratchpad; not in the repo):
zero overfull boxes. Fixes: `tab:main`/`tab:main-gt` wrapped in `\resizebox`
(were 50/45pt too wide); `tab:scorefn` wrap widened to 0.46\linewidth;
`tab:weights` un-wrapped (six columns overflow any wrap column); `fig_transfer_datasize`
regenerated at wrap size (3.2in, REDE Fig. 5 style); the appendix sensitivity strip
split into `fig_sensitivity_appendix_{qwen,deepseek}` (the 8-row strip was taller
than the page). `fig:overview` and `fig:qualitative` (scores_captain_traj{1,2}) are
in. Remaining undefined refs are the pre-existing appendix stubs (app:datasets,
app:metrics, app:implementation, app:baselines, app:prompting-gpt5, app:gt, app:anchors).

## Method overview figure — DONE 2026-08-25

`src/analysis/method_figure.py` → `manuscript/assets/soap_overview.pdf` (preview
`artifacts/method_figure/soap_overview.png`), placed in `method.tex` as
`fig:overview` where the blue "we need a method figure" note stood. Three stages
(frozen proxy → spectral base score → attention-guided rescoring), schematic:
bars and weights are illustrative, not data.

## Qualitative examples for the wide-window cells — DONE 2026-08-25

`scripts/ablations/qualitative_wide_w.py` → `artifacts/ablations/qualitative_wide_w/`
(gitignored). For the three cells whose Table-1 anchor selects a large window —
Qwen TE-Cap (w=5), Qwen TE-Mag (w=4), DeepSeek TE-Cap (w=all) — it re-runs the
anchor on the frozen triples' test splits (accuracies re-verified) and records
only the FLIPS: base argmax wrong, rescored argmax = gold. Per-seed flips 3/8/3,
one example per trajectory 3/4/1. Figures: base + SOAP curves, argmax markers,
gold dashed; no title, no grid. `MANIFEST.md` indexes all; `--replot` redraws
without rescoring. Most robust (flip on all three seeds): Qwen TE-Mag traj 30
(9 → gold 6) and DeepSeek TE-Cap traj 36 (4 → gold 2); Qwen TE-Cap traj 44
(14 → gold 12, seed 22).
