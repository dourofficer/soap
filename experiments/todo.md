# Experimental plan

Every pending experiment for the manuscript, made concrete: data, splits, backbones,
anchor configs, procedure, and cost. Agreed 2026-08-13; revised 2026-08-17 (coverage,
naming, orientations, grids, A7 re-selection); 2026-08-23 (B1 baseline rows);
2026-08-25 (E2 concretized on the gathered synthetic corpora); 2026-08-27 (S1 scalability
planned). The two main experiments fill
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

Environment note: the venv's torchvision/torchaudio are compiled against a different
torch and crash transformers' lazy imports; `a1_scorefn.py --stage nll` blocks both
modules before importing (`sys.modules[...] = None`). Fix the venv if extraction is
ever needed elsewhere.

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
