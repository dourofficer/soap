# Extended experiments

Experiments beyond `todo.md`, in its format and under its global conventions (frozen
triples, 30/20/50 splits, both backbones, step accuracy as the headline metric).
Written 2026-08-27.

## E3 — Validation-selected configs, in- and cross-distribution (`tab:main`, `tab:transfer`)  `[CPU]`  — DONE 2026-08-27

- [x] **Target.** Replace the optimistic test-selection of Table 1 with the protocol
  the manuscript already claims: within a frozen triple, per (backbone, subset), the
  reported config is the one that maximizes mean VALIDATION step accuracy over the
  three seeds; the number reported is its TEST accuracy. Measure what that costs
  in-distribution, and whether the transfer picture of E1 holds under it.

- **Selection rule (the one deviation from the global conventions).** `select_config`
  as in `main.sweep`, with the val columns in place of the test columns: argmax of
  mean `step_acc_val@1` over the triple's seeds, config present in all three seeds,
  tiebreak mean `agent_acc_val@1`, then the highest key (12 dp). Applied in the same
  two stages: base grid → (position, c_begin, c_end); rescore grid, expanded ONLY for
  the val-winning base config → (layer_range, γ, w) per strategy. Because the stored
  sweep tables expanded the rescore grid for the TEST-winning base config, the
  rescore stage must be re-run wherever the two base winners differ; the base stage
  is free (its val metrics are already in `results-*/<ds>/sweep/*/sweep.tsv`).

- **Part A — in-distribution.** All five reported subsets (WW-AG, WW-HC, CE's seven
  subsets, TE-Cap, TE-Mag), both backbones, BOTH trees (without-GT and with-GT), on
  the frozen triples of `configs-main/<ds>[-gt].yaml`. Base row and one row per
  strategy (`backprop`, `succ-strong`, `succ-near`), as in `select/selection.tsv`.
  Record test and val metrics for every selected row, and the test-selected row
  beside it, so the optimism gap (test-sel − val-sel, on test) is one subtraction.

- **Part B — cross-distribution transfer.** E1's 4×4 source→target grid over {WW-AG,
  WW-HC, TE-Cap, TE-Mag}, one grid per backbone, without-GT. As in E1: R is fit on
  the SOURCE's train split, dependency weights come from the target's own attention,
  seeds pair positionally, the target's test split is unchanged, and the full config
  is re-selected per pair (dense base grid, then the backprop rescore grid on the
  winning base config). Two changes from E1: selection is val-only, and the target's
  val is its MAIN-EXPERIMENT val split (20%), not train+val (40%) — so every cell,
  diagonal included, selects on the same data as Part A. (E1's val-convention rows,
  selected on the 40% pool, stay in `e1_transfer.tsv` for comparison.)

- **Sanity checks (asserted in the runner).**
  1. Part A's base-stage val winner equals a direct val-argmax over the stored
     `sweep.tsv`; where it coincides with the test winner, the rescore rows must
     reproduce the stored table bit-for-bit.
  2. Part B's diagonal must reproduce Part A's without-GT `backprop` row exactly
     (same splits, same rule) — the analogue of E1's diagonal check.
  3. On the base (`svd`) row, the val-selected config's val accuracy ≥ the
     test-selected one's, and its test accuracy ≤ the test-selected one's. Strategy
     rows carry no such guarantee: their rescore grids sit on DIFFERENT base configs
     (staging), so they are recorded, not asserted.

- **Deliverable.**
  - Part A: two main-table candidates (without-GT, with-GT), val-selected, laid out as
    Tables 1–2 (SOAP + base per backbone × subset, 3-seed mean ± std), plus the
    optimism-gap table.
  - Part B: two 4×4 tables (qwen3.5-9b, deepseek-8b), SOAP step acc %, rows = source,
    diagonal = Part A in-distribution. Read against E1's val-selected tables.

- **Code and results.**
  - `main/sweep.py`: a `select_rule: test|val` config key (default `test`, so nothing
    existing changes) that switches the metric columns in both `select_config` calls
    and in `run_select`. Val-rule outputs go to a sibling tree, never over the frozen
    ones: `results-nogt-valsel/`, `results-gt-valsel/`.
  - `scripts/ablations/e3_valsel_indist.py` (Part A): re-reads the frozen base grid,
    recomputes the rescore grid where the val winner differs (it differed in 41 of 44
    cells), writes the val-rule trees, runs `run_select` under `select_rule=val`.
  - `scripts/ablations/e3_valsel_transfer.py` (Part B): `e1_transfer.py`'s grid
    builders with the main val split and val-only selection; shardable by
    `--models` / `--sources` as E1 (4 shards, ~1 h each).
  - `results-ablations/e3_valsel_indist.tsv` (Part A; columns of `selection.tsv` plus
    `rule` and the paired test-selected metrics) and `results-ablations/
    e3_valsel_transfer.tsv` (Part B; columns of `e1_transfer.tsv`).

- **Cost.** No forward passes. Part A: ~2 h (the CE rescore passes dominate). Part
  B: ~1 h per shard, 4 shards.

- **Not decided here.** Whether the manuscript switches Tables 1–2 to Part A. The
  run produces both rules; the choice is made after reading the optimism gap. If it
  switches, every ablation anchor in `todo.md` (A1–A7, E2, S1) inherits the val rule
  in the "one sweep" the global conventions promise.

- **Results** — `results-ablations/e3_valsel_indist.tsv` (Part A, 176 rows: 44 cells ×
  4 rows, val-selected config and metrics beside the test-selected ones) and
  `results-ablations/e3_valsel_transfer.tsv` (Part B, 64 rows, merged from
  `e3_parts/`). Val-rule trees: `results-{nogt,gt}-valsel/<ds>/`. All sanity checks
  passed; the 8 Part B diagonals reproduce Part A exactly.

  **Part A — SOAP (`backprop`) step acc %, val-selected | test-selected (Table 1/2).
  CE = mean over its 7 subsets.**

  | | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | qwen3.5-9b, no-GT | 37.57 \| 47.62 | 25.29 \| 34.48 | 59.35 \| 61.78 | 22.48 \| 35.66 | 8.70 \| 23.19 |
  | deepseek-8b, no-GT | 40.74 \| 45.50 | 24.14 \| 28.74 | 59.15 \| 64.68 | 31.78 \| 42.64 | 23.19 \| 30.43 |
  | qwen3.5-9b, GT | 39.68 \| 43.39 | 24.14 \| 34.48 | 57.92 \| 60.59 | 22.48 \| 36.43 | 5.80 \| 21.01 |
  | deepseek-8b, GT | 34.92 \| 44.97 | 24.14 \| 29.89 | 61.61 \| 65.19 | 28.68 \| 42.64 | 33.33 \| 36.96 |

  Base (`svd`) row, same layout:

  | | WW-AG | WW-HC | CE | TE-Cap | TE-Mag |
  |---|---|---|---|---|---|
  | qwen3.5-9b, no-GT | 32.28 \| 39.15 | 26.44 \| 33.33 | 59.47 \| 61.38 | 19.38 \| 33.33 | 8.70 \| 21.01 |
  | deepseek-8b, no-GT | 34.39 \| 38.62 | 24.14 \| 28.74 | 58.47 \| 64.19 | 31.01 \| 40.31 | 23.19 \| 29.71 |
  | qwen3.5-9b, GT | 38.10 \| 41.27 | 28.74 \| 33.33 | 58.41 \| 60.30 | 19.38 \| 32.56 | 5.07 \| 15.94 |
  | deepseek-8b, GT | 34.92 \| 40.21 | 24.14 \| 29.89 | 61.57 \| 64.81 | 21.71 \| 41.09 | 33.33 \| 35.51 |

  **Part B — SOAP step acc %, val-selected, rows = source, diagonal (bold) = Part A.**

  qwen3.5-9b:

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **37.57** | 22.99 | 17.83 | 4.35 |
  | WW-HC | 23.81 | **25.29** | 26.36 | 4.35 |
  | TE-Cap | 15.87 | 28.74 | **22.48** | 7.97 |
  | TE-Mag | 16.93 | 25.29 | 25.58 | **8.70** |

  deepseek-8b:

  | | →WW-AG | →WW-HC | →TE-Cap | →TE-Mag |
  |---|---|---|---|---|
  | WW-AG | **40.74** | 8.05 | 28.68 | 14.49 |
  | WW-HC | 24.34 | **24.14** | 32.56 | 28.99 |
  | TE-Cap | 19.58 | 18.39 | **31.78** | 20.29 |
  | TE-Mag | 24.87 | 14.94 | **31.78** | 23.19 |

  Reading. (1) The optimism gap is large: val selection costs SOAP 2–15 points of test
  accuracy on the small subsets (WW, TE) and 2–5 on CE, whose 7 subsets are big
  enough for the val split to generalize. The worst cell is qwen TE-Mag (8.70 vs
  23.19): TE-Mag's val split holds ~18 trajectories, so val accuracy moves in steps of
  5.6 points and the val argmax is close to a coin toss. (2) SOAP still beats the base
  score on 14 of 20 cells under val selection, ties on 4 (γ=0 selected) and loses on
  2 (both WW-HC, qwen), so the rescoring gain survives the honest protocol, smaller.
  (3) Cross-distribution: with the 20 % val split the diagonal no longer wins every
  column — a foreign reference selected on the same tiny val split is as good or
  better in 5 of 8 columns (e.g. deepseek WW-HC→TE-Cap 32.56 vs 31.78 in-dist; qwen
  TE-Cap→WW-HC 28.74 vs 25.29). E1's 40 %-pool version shows the same ranking with
  higher numbers throughout. The honest conclusion is that at these subset sizes
  selection noise, not distribution shift, dominates the off-diagonal.

- **Implication for the manuscript.** Switching Tables 1–2 to Part A is defensible
  but would drop most headline numbers by 5–15 points and make TE-Mag unreportable
  for qwen. If the switch is made, the ablation anchors (A1–A7, E2, S1) must be
  re-selected under the val rule; `select_rule: val` in the config does that.
