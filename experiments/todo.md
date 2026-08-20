# Experimental plan

Every pending experiment for the manuscript, made concrete: data, splits, backbones,
anchor configs, procedure, and cost. Agreed 2026-08-13; revised 2026-08-17 (coverage,
naming, orientations, grids, A7 re-selection). The two main experiments fill
`fig:transfer` and `tab:synth`; the seven ablations fill `tab:scorefn`, `tab:weights`,
`tab:position`, `tab:attnsel`, `fig:gamma`, `fig:layers`, `fig:datasize`.

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
  `a6a_rep_layer`, `a6b_attn_band`, `a7_datasize`, `e1_transfer`, `e2_synthfit`.
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

### E1 — Cross-distribution transfer (`fig:transfer`)  `[CPU]`  — DONE 2026-08-17; PROTOCOL REVISED + RERUN 2026-08-18

- [x] **Target.** Is a fitted SOAP specific to the distribution it was tuned on?
- **Procedure (revised 2026-08-18).** 4×4 source→target grid over {WW-AG, WW-HC,
  TE-Cap, TE-Mag}, one grid per backbone. For each pair: fit R on the SOURCE's train
  split, then RE-SELECT the full config from scratch on the TARGET — dense base grid
  (position × band, ens-mid3 included), then the backprop rescore grid on the winning
  base config — test-selected by the standard rule, exactly like every other
  experiment. Each cell is "the best a source-fitted reference can do on the target".
  Dependency weights always come from the target trajectories' own attention. Seeds
  pair positionally; report the 3-seed mean. (The original protocol froze the
  source's anchor config; that grid is preserved in
  `results-ablations/e1_transfer_frozen-anchor.tsv`.)
- **Sanity check.** The diagonal repeats the main sweep's protocol verbatim, so it
  must reproduce Table 1's base AND SOAP rows. Verified on both backbones.
- **Deliverable.** Two 4×4 step-accuracy heatmaps (one per backbone).
- (CE dropped from the grid with the pooled-source design; can be reinstated later if
  a 5×5 is wanted.)
- **Results** — `results-ablations/e1_transfer.tsv` (`scripts/ablations/e1_transfer.py`;
  base and soap rows per pair with the selected configs). SOAP step acc %, rows =
  source:

  | qwen3.5-9b | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **47.62** | 32.18 | 34.11 | 21.01 |
  | WW-HC | 35.45 | **34.48** | 34.11 | 22.46 |
  | TE-Cap | 26.98 | 33.33 | **35.66** | 21.74 |
  | TE-Mag | 28.57 | 29.89 | 34.88 | **23.19** |

  | deepseek-8b | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **45.50** | 25.29 | 34.88 | 30.43 |
  | WW-HC | 40.74 | **28.74** | 42.64 | 33.33 |
  | TE-Cap | 29.63 | 32.18 | **42.64** | 28.99 |
  | TE-Mag | 35.45 | 29.89 | 30.23 | **30.43** |

  (Bold marks the in-domain diagonal, not the column best.)

  Reading: with the config re-selected per target, the fitted reference travels far
  better than under the frozen-anchor protocol (WW-AG→WW-HC: 32.18 vs 3.45 before).
  On Qwen the diagonal still wins every column, by a wide margin only on WW-AG
  (47.62 vs 35.45); elsewhere the best foreign source lands within ~1 point. On
  DeepSeek the diagonal is not always best: WW-HC→TE-Cap ties in-domain (42.64) and
  WW-HC→TE-Mag beats it (33.33 vs 30.43); DeepSeek WW-HC's own cell (28.74, γ=0) is
  beaten by three foreign sources. Conclusion: the distribution-specific part of
  SOAP is chiefly the hyperparameter configuration; the spectral reference itself is
  broadly reusable.

### E2 — Synthetic reference trajectories (`tab:synth`)  `[GPU, DEFERRED]`

- [ ] **Target.** Does SOAP work when no corpus from the target system exists to fit R?
- **Deferred because** the synthetic trajectories do not exist yet; generators will be
  **Qwen3.5-9B** and **GPT-4o** (Qwen3.5-35B-A3B is dropped everywhere).
- **Scope.** WW-AG and WW-HC only — `tab:synth` shrinks from five subsets to two.
- **Data.** Validation and test splits unchanged (frozen triples). Only the reference
  corpus is replaced by synthetic trajectories, used as-is regardless of task success.
  One synthetic corpus per generator, shared across the triple's seeds (per-seed
  variance comes from the val/test splits alone).
- **Procedure.** Extract activations for the synthetic corpus (GPU) → fit R on it →
  RE-SELECT hyperparameters (base grid, then rescore grid) by the standard protocol →
  report S and +SOAP. Each row is "the best that reference corpus can do", matching
  the optimistic protocol of the real-corpus row.
- **Rows.** Real (= Table 1) / synthetic Qwen3.5-9B / synthetic GPT-4o.

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

## Deferred (in the plan, blocked on inputs)

- **E2 — synthetic reference fit**: blocked until the synthetic trajectories arrive
  (generators Qwen3.5-9B and GPT-4o).

## Tracked outside this plan

- **Scalability** (`fig:scale`): Qwen3.5-9B/14B/27B with All-at-Once, AgenTracer, OAT.
  Pending — the expensive item (new extraction at 14B/27B plus three baselines).
- **Baseline rows**: OAT, StepFinder, AgenTracer, GraphTracer, RAFFLES (dashes in
  Tables 1–2).
- **With-GT SOAP adaptation**: announced in Setup, not yet described or planned here.

## Execution order

1. **Free reads** — A4, A6(a′), A6(b) — DONE 2026-08-17.
2. **CPU batch** — A5, A3, A6(a), E1, A7 — DONE 2026-08-17.
3. **GPU batch** — A1 — DONE 2026-08-17.
4. **E2** — when the synthetic trajectories arrive (extraction → fit → re-select →
   score).

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
