# Review-response experiments

Four experiments that answer Weaknesses 2–4 of `SOAP_ICLR2027_review.pdf`. Drafted and
approved 2026-09-21; all four ran the same day. Each section ends with a **Results**
block; "Summary of findings" at the end collects what they mean for the paper.
Two more, E5 and E6, stand in for the success-only reference control; they were
planned and run later the same day and report three selection modes each.

## Conventions

These follow `experiments/todo.md`'s global conventions, with the points that matter
restated.

- **Coverage.** Both backbones (qwen3.5-9b, deepseek-8b) × WW-AG, WW-HC, TE-Cap,
  TE-Mag, without GT, on the frozen seed triples. A number is the mean over the triple.
- **Selection rule: TEST-selected, as in Table 1.** Anchors are the `svd` (base) and
  `backprop` (SOAP) rows of `results-nogt/<ds>/select/selection.tsv`. Any knob a control
  selects uses the same rule: mean test step accuracy over the triple, tiebreak agent
  accuracy (`main.sweep.select_config`). So every row stands on its most optimistic
  setting, like SOAP's. Each runner takes `--select-rule {test,val}` (default `test`);
  the later conversion to validation selection is that one flag.
- **Self-check.** Every runner first reproduces the anchor's base and SOAP accuracy to
  1e-9, and writes no control row until it does.
- **Code and output.** `scripts/ablations/<exp>_<slug>.py` →
  `results-ablations/<exp>_<slug>.tsv`. `manuscript/` is not edited by these runs.
- **Step length** means `n_tokens`, the step's own token count under the backbone's
  tokenizer, read from `results-ablations/a1_scorefn/nll/<ds>-<subset>-<model>.tsv`
  (A1 already computed it for all eight cells). No new GPU pass.

## A10 — Length baselines (`a10_trivial`)  `[CPU]`

- **Target.** Weakness 4: the base score is an unnormalized projection of a mean-pooled
  vector, so it may track step length. Does a length heuristic alone find the decisive
  step?
- **Rows.** Nothing is selected.
  1. *Longest step* — predict the step with the most tokens.
  2. *Shortest step* — predict the step with the fewest.
  3. *Random* — expected accuracy of a uniform guess, the mean of 1/T over test
     trajectories (T = scored steps).
  4. *Base score* and *SOAP* — the anchors, for reference.
- **Optional diagnostics — keep or strike.** They are not baselines, but they answer
  the same reviewer point directly:
  - the mean within-trajectory Spearman correlation of the base score with `n_tokens`
    and with the vector norm ‖v‖, one number per cell;
  - a *unit-norm base score* row: scale every step vector to ‖v‖ = 1 (reference and
    test alike) before the projection, anchor band unchanged. If accuracy holds, the
    score does not live on the norm.
- **Reading.** If longest or shortest approaches the base score on a subset, length
  explains part of the signal there, and the paper must say so. If both sit near
  random, the confound is answered.
- **Output.** `results-ablations/a10_trivial.tsv`.
- **Results** — DONE 2026-09-21 (`scripts/ablations/a10_trivial.py`, log
  `logs/a10_trivial.log`; every base row reproduced its selection accuracy). Both
  optional diagnostics were kept, and one row was added after the first run:
  *unit-norm (band re-selected)*. The base score chose its band on unnormalized
  vectors, so holding that band fixed understates the unit-norm score; the added row
  re-selects the band over all 210 at the anchor layer by the standard test rule.
  Step accuracy %, test:

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Longest step | 19.05 | 19.54 | 10.08 | 11.59 |
  | Shortest step | 0.53 | 0.00 | 4.65 | 2.90 |
  | Random | 12.10 | 5.08 | 8.58 | 5.16 |
  | Unit-norm, anchor band | 35.45 | 16.09 | 24.81 | 16.67 |
  | Unit-norm, band re-selected | 35.45 | 22.99 | 30.23 | 23.19 |
  | Base score | 39.15 | 33.33 | 33.33 | 21.01 |
  | SOAP | 47.62 | 34.48 | 35.66 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Longest step | 15.87 | 19.54 | 8.53 | 12.32 |
  | Shortest step | 0.53 | 0.00 | 4.65 | 1.45 |
  | Random | 12.10 | 5.08 | 8.58 | 5.16 |
  | Unit-norm, anchor band | 29.10 | 6.90 | 25.58 | 10.87 |
  | Unit-norm, band re-selected | 33.86 | 22.99 | 27.13 | 22.46 |
  | Base score | 38.62 | 28.74 | 40.31 | 29.71 |
  | SOAP | 45.50 | 28.74 | 42.64 | 30.43 |

  Mean within-trajectory Spearman correlation of the base score, test split:

  | | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | qwen3.5-9b, with token count | 0.42 | 0.65 | 0.64 | −0.26 |
  | qwen3.5-9b, with ‖v‖ | −0.30 | −0.86 | −0.94 | −0.92 |
  | deepseek-8b, with token count | 0.37 | 0.73 | 0.72 | 0.44 |
  | deepseek-8b, with ‖v‖ | −0.34 | −0.82 | −0.93 | −0.70 |

  Reading. Length alone does not find the decisive step: the longest step scores
  10–20 %, above random but 12–30 points below the base score, and the shortest step
  sits at or below random. So the base score is not a length heuristic. The reviewer's
  norm point, however, holds in part. Wherever the anchor band starts at component 0
  (every cell but WW-AG), the base score is close to a monotone function of the vector
  norm (ρ from −0.70 to −0.94): the top singular vector is the mean direction, so the
  energy along it is nearly the squared norm, and "low energy" means "small vector".
  Removing the norm costs 3–10 points on Qwen (TE-Mag gains 2) and 5–13 on DeepSeek
  even after the band is re-selected; what remains, 23–35 %, still clears every length
  and random row by a wide margin. WW-AG, whose band [1, k) skips the mean direction,
  is the cell least tied to the norm (|ρ| ≈ 0.3) and loses the least (39.15 → 35.45 on
  Qwen). For the paper: state that the score has a norm component and an angular
  component, report the unit-norm row, and drop any wording that implies the signal is
  purely directional. A1's L2-norm row (26.44 on Qwen WW-HC against 33.33) already
  showed the norm alone is not enough; this table shows the direction alone is not
  either.

## A11 — Rescoring controls (`a11_rescore_controls`)  `[CPU]`

- **Target.** Weakness 2: the reviewer argues rescoring works as a last-step penalty
  (the final step collects nothing, every other step gets a near-equal boost), and
  that with w=1 it mostly credits the next step's strongest predecessor. Is the
  attention routing doing anything a cruder rule does not?
- **Procedure.** The anchor's base configuration is fixed. Every row that has a γ
  co-selects it on the shared grid {0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0} by the test rule,
  so each control stands on its best γ, as SOAP does.
- **Rows.**
  1. *Base score.*
  2. *SOAP* — anchor attention band and w.
  3. *Next-step shift* — `S̃_i = S_i + γ·S_{i+1}`; the last step gets +0. The
     reviewer's named control for w=1.
  4. *Constant boost* — `S̃_i = S_i + γ·mean(S)` for every non-final step, +0 for the
     final one. A pure last-step penalty with no routing at all.
  5. *Exclude final step* — base score with the last scored step removed from the
     argmax. No γ.
  6. *Uniform (normalized)* — each step collects the mean of all its successors' base
     scores; A2's row, repeated for one table.
  7. *SOAP, length-normalized attention* — divide the attention mass landing in each
     predecessor by that predecessor's `n_tokens`, renormalize, then proceed as SOAP.
     Tests the bias toward long steps.
  8. *SOAP, first step removed* — drop step 1 (task description, attention sink) from
     every step's dependency weights, renormalize.
- **Diagnostics.** Among test trajectories where SOAP's prediction differs from the
  base score's: the share whose base prediction was the final step, and the mean
  distance (in steps) the prediction moved. Per cell, the share of dependency weight
  that lands on step 1.
- **Checks.** At γ=0 every row equals base. Row 7 with every `n_tokens` set to 1 equals
  SOAP exactly.
- **Caveat.** DeepSeek WW-HC's anchor has γ=0; rows 3, 4, 6, 7, 8 may still select
  γ>0 there, since γ is co-selected per row.
- **Reading.** Where rows 3–5 match SOAP, the honest claim is "a last-step correction",
  not "blame routed along dependencies"; keep the routing claim only on subsets where
  SOAP beats all three. Rows 7–8 within a point of SOAP answer the length and sink
  objections; a gain from either is a free improvement worth adopting.
- **Output.** `results-ablations/a11_rescore_controls.tsv` (every γ recorded,
  `selected` marks the winner, as in A3).
- **Results** — DONE 2026-09-21 (`scripts/ablations/a11_rescore_controls.py`, log
  `logs/a11_rescore_controls.log`; diagnostics in
  `results-ablations/a11_rescore_controls_diag.tsv`). Every base and SOAP row reproduced
  its selection accuracy, and every row equals base at γ=0. Step accuracy %, test, with
  the co-selected γ in brackets. DeepSeek WW-HC's anchor chose γ=0, so it has no
  attention band or w to inherit and its three SOAP rows are blank.

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Base score | 39.15 | 33.33 | 33.33 | 21.01 |
  | Exclude final step | 40.74 | 22.99 | 34.88 | 19.57 |
  | Next-step shift | 39.68 (0.1) | 33.33 (0) | 35.66 (0.1) | 21.01 (0) |
  | Constant boost | 40.74 (1.0) | 34.48 (0.1) | 34.88 (1.0) | 21.01 (0) |
  | Uniform (normalized) | 41.27 (0.8) | 34.48 (0.1) | 35.66 (0.1) | 21.01 (0) |
  | **SOAP** | **47.62** (0.6) | 34.48 (0.1) | 35.66 (0.1) | **23.19** (1.0) |
  | SOAP, length-normalized | 40.74 (0.1) | 34.48 (0.1) | 33.33 (0.1) | 21.01 (0) |
  | SOAP, first turn removed | 40.74 (0.2) | 34.48 (0.1) | 35.66 (0.1) | 21.74 (1.0) |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Base score | 38.62 | 28.74 | 40.31 | 29.71 |
  | Exclude final step | 38.62 | 16.09 | 42.64 | 27.54 |
  | Next-step shift | 38.62 (0) | 28.74 (0) | 40.31 (0.1) | 29.71 (0) |
  | Constant boost | 38.62 (1.0) | 28.74 (0) | 42.64 (1.0) | 29.71 (0) |
  | Uniform (normalized) | 38.62 (0) | 28.74 (0) | 42.64 (0.1) | 29.71 (0) |
  | **SOAP** | **45.50** (1.0) | — | 42.64 (0.2) | **30.43** (0.1) |
  | SOAP, length-normalized | 38.62 (0) | — | 42.64 (0.2) | 29.71 (0) |
  | SOAP, first turn removed | 40.21 (0.8) | — | 42.64 (0.2) | 30.43 (0.1) |

  Where SOAP's prediction differs from the base score's (test split, three seeds pooled):

  | Cell | Flips / trajectories | Base prediction was the final step | Mean steps moved earlier | Fixed / broke | Weight on first turn (uniform would give) |
  |---|---|---|---|---|---|
  | Qwen WW-AG | 65 / 189 | 6 % | 1.5 | 23 / 7 | 0.44 (0.34) |
  | DeepSeek WW-AG | 75 / 189 | 1 % | 1.6 | 28 / 15 | 0.42 (0.34) |
  | Qwen WW-HC | 9 / 87 | 44 % | 21.3 | 1 / 0 | 0.03 (0.07) |
  | Qwen TE-Cap | 6 / 129 | 100 % | 4.0 | 3 / 0 | 0.27 (0.17) |
  | DeepSeek TE-Cap | 7 / 129 | 100 % | 9.4 | 3 / 0 | 0.18 (0.17) |
  | Qwen TE-Mag | 52 / 138 | 29 % | 8.9 | 8 / 5 | 0.16 (0.13) |
  | DeepSeek TE-Mag | 14 / 138 | 29 % | 2.4 | 3 / 2 | 0.17 (0.13) |

  Reading. The answer differs by subset, and the paper should say so.
  - **WW-AG: the attention routing is real.** No attention-free control comes within 6
    points of SOAP (best: uniform 41.27 on Qwen, nothing above base on DeepSeek).
    Almost none of the displaced predictions were the final step (6 % and 1 %), and the
    prediction moves 1.5 steps on average, so the reviewer's "last-step penalty" reading
    is wrong here, and so is "next-step shift", which gains nothing. The net effect is
    +16 and +13 correct trajectories.
  - **WW-AG: but the gain rides on raw attention mass.** Dividing the mass by the
    predecessor's length, or dropping the first turn, removes almost all of it (47.62 →
    40.74 on Qwen; 45.50 → 38.62 / 40.21 on DeepSeek). The first turn draws 0.42–0.44
    of the dependency weight where a uniform spread would give 0.34. So at w=1 much of
    the lift goes to steps that are long or that open the trajectory. That may be the
    right prior for WW-AG, whose decisive steps come early, but it means the reviewer's
    length and sink objection is NOT answered by these rows: the two variants are not
    harmless, they are where the gain lives. One caveat: both variants inherit the
    anchor's band and w, which were selected for raw attention; re-selecting them would
    be the fair upper bound and is a small follow-up.
  - **TE-Cap: rescoring is a last-step penalty.** Every flip displaces a final-step
    prediction, and the constant boost or plain exclusion of the final step matches
    SOAP on DeepSeek (42.64) and sits one trajectory below it on Qwen (34.88 vs 35.66).
  - **WW-HC: no evidence for routing.** Uniform averaging and the constant boost tie
    SOAP (34.48), which is one trajectory above base; DeepSeek selects γ=0.
  - **TE-Mag: a small routing gain.** SOAP is the only row above base (+2.2 and +0.7
    points, a net of three and one trajectories); excluding the final step hurts.
  - Excluding the final step outright is harmful on WW-HC (−10 to −13 points): the
    final step is often the decisive one there, which also explains why a soft penalty
    (γ=0.1) is the most any row selects.

  For the paper: keep the dependency-routing claim for WW-AG, where the controls
  support it, and describe the other subsets as they are — a correction of the base
  score's bias toward the final step, worth 0–2 points. Replace the featured qualitative
  example (the 0.88 vs 0.87 last-step case) with a WW-AG trajectory where the prediction
  moves one or two steps between non-final steps. Report the length-normalized and
  first-turn rows rather than omit them.

### A11 on CORRECT-Error — added 2026-09-21 on request

- **Scope.** The same runner on all seven CORRECT-Error subsets, both backbones
  (`--configs configs-main/correct-error.yaml`; output
  `results-ablations/a11_rescore_controls_ce.tsv` and `..._ce_diag.tsv`; log
  `logs/a11_rescore_controls_ce.log`). Base and SOAP rows reproduced the selection table
  in every cell, and the macro-averages reproduce Table 1 (61.38 → 61.78 on Qwen,
  64.19 → 64.68 on DeepSeek).
- **Two differences from the main A11 run.** A1 never computed token counts for
  CORRECT-Error, so the length-normalized row tokenizes each turn's bare content with
  the backbone's tokenizer on CPU (`common.load_ntokens`), not A1's serialized,
  truncated step. And only 5 of the 14 anchors selected γ > 0 (Qwen arc, gaia, hotpot,
  wikimqa; DeepSeek wikimqa); elsewhere SOAP is the base score, the three SOAP rows are
  blank, and the macro-average fills them with the base score, as Table 1 does.
- **Results.** Step accuracy %, test; co-selected γ in the TSV.

  | qwen3.5-9b | arc | gaia | hotpot | math500 | mmlu_pro | musique | wikimqa | macro |
  |---|---|---|---|---|---|---|---|---|
  | Base score | 81.36 | 68.00 | 45.67 | 61.18 | 79.71 | 43.59 | 50.14 | 61.38 |
  | Exclude final step | 2.63 | 8.00 | 13.49 | 9.28 | 8.70 | 17.95 | 13.71 | 10.54 |
  | Next-step shift | 81.58 | 72.00 | 45.91 | 61.18 | 79.71 | 43.59 | 50.59 | **62.08** |
  | Constant boost | 81.36 | 70.67 | 45.67 | 61.18 | 79.71 | 43.59 | 50.14 | 61.76 |
  | Uniform (normalized) | 81.36 | 69.33 | 45.67 | 61.18 | 79.71 | 43.59 | 50.14 | 61.57 |
  | SOAP | 81.58 | 69.33 | 46.48 | — | — | — | 50.59 | 61.78 |
  | SOAP, length-normalized | 81.36 | 69.33 | 46.83 | — | — | — | 50.59 | 61.80 |
  | SOAP, first turn removed | 81.80 | 69.33 | 46.48 | — | — | — | 50.59 | 61.81 |

  | deepseek-8b | arc | gaia | hotpot | math500 | mmlu_pro | musique | wikimqa | macro |
  |---|---|---|---|---|---|---|---|---|
  | Base score | 87.72 | 72.00 | 52.71 | 59.92 | 80.43 | 44.66 | 51.86 | 64.19 |
  | Exclude final step | 5.26 | 6.67 | 20.65 | 8.86 | 9.42 | 16.67 | 14.35 | 11.70 |
  | Next-step, constant, uniform | 87.72 | 72.00 | 52.71 | 59.92 | 80.43 | 44.66 | 51.86 | 64.19 |
  | SOAP | — | — | — | — | — | — | **55.31** | 64.68 |
  | SOAP, length-normalized | — | — | — | — | — | — | 51.86 | 64.19 |
  | SOAP, first turn removed | — | — | — | — | — | — | 55.31 | 64.68 |

  Flips on the five cells with γ > 0 (test, three seeds pooled): Qwen arc 14 of 456
  trajectories (3 fixed, 2 broke); gaia 2 of 75 (1 / 0); hotpot 103 of 867 (27 / 20);
  wikimqa 111 of 1,101 (20 / 15); DeepSeek wikimqa 309 of 1,101 (94 / 56). The
  displaced prediction was the final step in 0–29 % of flips, and the first turn draws
  LESS weight than a uniform spread would give it (0.12–0.28 against 0.21–0.34).

  Reading.
  - **On Qwen, rescoring on CORRECT-Error is indistinguishable from the crude
    controls.** Every row sits between 61.57 and 62.08, and the next-step shift — the
    reviewer's named control — is the best row, above SOAP. The +0.4 that Table 1
    reports cannot be credited to attention routing.
  - **One cell shows real routing: DeepSeek wikimqa,** +3.45 points (a net of 38
    trajectories out of 1,101), which no attention-free control recovers at any γ.
    As on WW-AG, length normalization erases it (55.31 → 51.86), while removing the
    first turn does not — so here the gain rides on long predecessors, not on an
    attention sink.
  - **A new confound, and a larger one than anything else in this file: the decisive
    step on CORRECT-Error is usually the final step.** Excluding the final step drops
    the base score from 61–64 % to 11 %. Counted over the whole corpus, the gold step
    is the last turn in 81.6 % of arc, 71.7 % of mmlu_pro, 65.0 % of math500, 64.0 % of
    gaia, 42.2 % of wikimqa, 41.7 % of hotpot and 36.5 % of musique trajectories — a
    macro-average of 57.5 %. "Always predict the final step" therefore scores about
    57.5 on the CE column, against 61.78 / 64.68 for SOAP and 55.03–62.42 for the
    strongest judges. (This is the corpus-wide share, not the mean over the three test
    splits; a test-split number needs a one-line runner.) The reviewer asked for a
    fixed-index baseline, and this plan had dropped it; on CORRECT-Error it must go
    back in, because it shows every method in that column is within a few points of a
    rule that reads nothing.

  For the paper: add a "final step" row to the CE column of Table 1, and do not cite
  CORRECT-Error as evidence for rescoring. Its value is the base score's margin over
  the final-step rule on hotpot, musique and wikimqa, where the final step is decisive
  in under half the trajectories.

## E4 — Reference controls (`e4_reference_controls`)  `[one GPU extraction, then CPU]`

- **Target.** Weakness 3: is the reference matrix R capturing failure structure, or
  generic representation geometry that any text would give?
- **Procedure.** E2's design: fit R on the alternative corpus; score the real test
  split; dependency weights from the target trajectories' own attention. Each
  reference is reported twice — at the frozen anchor configuration, and with the full
  configuration re-selected (test rule).
- **Rows.**
  1. *Benchmark train split* — the anchor; must reproduce Table 1.
  2. *Synthetic, mixed* — E2's references (`ag-*`, `hc-*`), recomputed for one table.
     WW-AG and WW-HC only.
  3. *Synthetic, failure-only* — the same corpora filtered to `outcome == "fail"`
     (110/126 and 93/124 for WW-AG; 53/55 and 49/55 for WW-HC). Activations exist; no
     extraction. WW-AG and WW-HC only.
  4. *Unrelated text* — WikiText-103 paragraphs, 8 per pseudo-trajectory, 120 files,
     staged as `data/synthetic/wikitext/` and extracted once per backbone
     (`python -m main extract --config configs-main/synthetic.yaml --stage activations
     --subset wikitext`). The extractor wraps each paragraph as `[assistant] - Step i:`,
     so the control keeps the log format and changes only the content. All four cells.
  5. *Random orthonormal basis* — A1's floor, dimension-matched to the anchor band. All
     four cells.
- **Excluded.** A success-only reference (see below).
- **Reading.** If unrelated text comes close to the benchmark reference, R encodes the
  backbone's generic geometry and the "failed trajectories" premise must be rewritten
  as "a reference for the common mode of step representations". If it falls toward
  random while synthetic references hold, in-distribution agent logs are what matter —
  still short of "failures matter", which only the success-only control can settle.
- **Output.** `results-ablations/e4_reference_controls.tsv`.
- **Results** — DONE 2026-09-21 (`scripts/ablations/e4_reference_controls.py`, staging
  `scripts/ablations/e4_stage_wikitext.py`; four runs merged from
  `results-ablations/e4_parts_*.tsv`; logs `logs/e4_*.log`). The real reference
  reproduced Table 1's base and SOAP rows in all eight cells, under both modes. WikiText
  activations: `results-nogt/synthetic/activations/<model>/wikitext/` (120 files each;
  `wikitext` added to `configs-main/synthetic.yaml`). Corpus sizes: real = the 30 %
  split; synthetic 124 / 126 (WW-AG) and 55 / 55 (WW-HC), failure-only 93 / 110 and
  49 / 53; WikiText 120. SOAP step accuracy %, test (base rows are in the TSV).

  **Configuration re-selected per reference (test rule — the protocol of Table 1)**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 47.62 | 34.48 | 35.66 | 23.19 |
  | Synthetic (Qwen), mixed | 41.80 | 31.03 | — | — |
  | Synthetic (Qwen), failures only | 41.27 | 31.03 | — | — |
  | Synthetic (GPT-4o), mixed | 46.56 | 29.89 | — | — |
  | Synthetic (GPT-4o), failures only | 43.92 | 29.89 | — | — |
  | WikiText | 35.45 | 34.48 | 35.66 | 27.54 |
  | Random basis, worst of 11 draws (WW) | 33.33 | 25.29 | 30.23 | 23.91 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 45.50 | 28.74 | 42.64 | 30.43 |
  | Synthetic (Qwen), mixed | 39.68 | 29.89 | — | — |
  | Synthetic (Qwen), failures only | 37.57 | 26.44 | — | — |
  | Synthetic (GPT-4o), mixed | 38.10 | 25.29 | — | — |
  | Synthetic (GPT-4o), failures only | 38.10 | 25.29 | — | — |
  | WikiText | 41.80 | 28.74 | 31.01 | 23.91 |
  | Random basis, worst of 11 draws (WW) | 36.51 | 27.59 | 31.01 | 22.46 |

  **Table 1's configuration, frozen (only the reference changes)**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 47.62 | 34.48 | 35.66 | 23.19 |
  | Synthetic (Qwen), mixed / failures only | 30.16 / 30.69 | 24.14 / 24.14 | — | — |
  | Synthetic (GPT-4o), mixed / failures only | 34.39 / 35.45 | 17.24 / 17.24 | — | — |
  | WikiText | 24.34 | 31.03 | 11.63 | 16.67 |
  | Random basis, worst of 11 draws (WW) | 14.81 | 3.45 | 8.53 | 15.22 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 45.50 | 28.74 | 42.64 | 30.43 |
  | Synthetic (Qwen), mixed / failures only | 20.11 / 22.22 | 22.99 / 22.99 | — | — |
  | Synthetic (GPT-4o), mixed / failures only | 26.98 / 25.93 | 13.79 / 13.79 | — | — |
  | WikiText | 12.17 | 11.49 | 2.33 | 8.70 |
  | Random basis, worst of 11 draws (WW) | 14.29 | 2.30 | 19.38 | 8.70 |

  Reading. This is the result that most needs attention before the rebuttal.
  - **Under Table 1's protocol, a reference with no agent content does about as well as
    the benchmark's failed trajectories.** With the configuration re-selected on test,
    WikiText ties the real reference on Qwen WW-HC and TE-Cap and beats it on TE-Mag
    (27.54 vs 23.19); a random basis — no reference at all — stays within 1–2 points
    of the real reference on DeepSeek WW-HC even at its worst draw (27.59 vs 28.74).
    The real reference keeps a clear lead over the worst random draw on WW-AG (+9 to
    +14) and on Qwen WW-HC (+9), and over WikiText only on DeepSeek TE-Cap and TE-Mag
    (+7 to +12) and WW-AG (+4 to +12).
  - **The random-basis rows for WW-AG and WW-HC are the WORST of 11 draws** (added
    2026-09-22 on request: draw 0 is the main run's basis, draws 1–10 are new;
    `results-ablations/e4_random_draws_<model>_<target>.tsv`, logs
    `logs/e4_random_draws_*.log`; the TraceElephant cells keep the single draw). The
    first run's single draw was lucky: it was the best of 11 on Qwen WW-AG (43.92,
    median 39.15) and on DeepSeek WW-HC (34.48, median 31.03). Re-selected, the draws
    span 33.33–43.92 (Qwen WW-AG), 36.51–47.09 (DeepSeek WW-AG), 25.29–33.33 (Qwen
    WW-HC) and 27.59–34.48 (DeepSeek WW-HC).
    The real reference beats every draw on Qwen and 9 of 11 on DeepSeek WW-AG, and
    loses to 10 of 11 on DeepSeek WW-HC. A 10–14 point spread across bases that
    carry no information is itself a measure of what test selection over the grid
    can produce.
  - **Two things explain it, and neither is failure structure.** First, A10 showed the
    score has a large norm component, and the energy of a vector in any 20-d basis
    grows with its norm, so any basis inherits that signal. Second, the base grid holds
    2,310 (Qwen) to 7,350 (DeepSeek) configurations selected on 29–63 test
    trajectories; the best of that many draws is high whatever the basis. The
    re-selected table therefore measures the protocol's optimism as much as any
    reference. It cannot support the claim that failed trajectories are what matter,
    and it shows that the claim cannot be tested under test selection at all.
  - **Failure-only versus mixed makes no difference** (within 0–3 points, both signs).
    The corpora are 74–96 % failures to begin with, so this row says little; the
    success-only control remains the real test and remains excluded.
  - **The frozen-configuration table looks like support for the real reference, but
    read it with care.** Every other reference collapses at Table 1's configuration,
    often to random. That shows the selected layer and band are specific to the basis
    they were selected with — the components of two different bases are not aligned, so
    band [1, 7) means something else in each — not that the real reference carries more
    signal. The one exception proves the point: WikiText at Qwen WW-HC's band [0, 5)
    keeps 31.03 of 34.48, because the top component of any text corpus is the mean
    direction, which is the norm.
  - **Follow-up, validation rule — DONE 2026-09-21.** The same runner under
    `--select-rule val` (`results-ablations/e4_reference_controls_valsel.tsv`, logs
    `logs/e4_valsel_*.log`): anchors from `results-nogt-valsel/`, every reference's
    configuration re-selected on the validation split, test accuracy reported. The real
    row reproduced the validation-selected tree in all eight cells. This removes the
    best-of-thousands optimism, so it is the fair test of the reference. SOAP step
    accuracy %, test:

    | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
    |---|---|---|---|---|
    | Benchmark train split | 37.57 | 25.29 | 22.48 | 8.70 |
    | Synthetic (Qwen), mixed / failures only | 27.51 / 28.04 | 12.64 / 12.64 | — | — |
    | Synthetic (GPT-4o), mixed / failures only | 38.10 / 26.46 | 13.79 / 13.79 | — | — |
    | WikiText | 30.69 | 32.18 | 26.36 | 21.01 |
    | Random basis | 23.28 | 12.64 | 24.81 | 18.12 |

    | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
    |---|---|---|---|---|
    | Benchmark train split | 40.74 | 24.14 | 31.78 | 23.19 |
    | Synthetic (Qwen), mixed / failures only | 29.63 / 25.93 | 14.94 / 18.39 | — | — |
    | Synthetic (GPT-4o), mixed / failures only | 32.28 / 33.33 | 13.79 / 8.05 | — | — |
    | WikiText | 30.69 | 17.24 | 32.56 | 18.12 |
    | Random basis | 30.69 | 14.94 | 23.26 | 14.49 |

    Reading. With the optimism gone the real reference separates from the controls in
    some cells and not in others. It leads on WW-AG with both backbones (+7 to +14 over
    WikiText and the random basis) and on DeepSeek WW-HC and TE-Mag (+5 to +9). On Qwen
    it loses to WikiText on WW-HC, TE-Cap and TE-Mag (by 4 to 12 points), and DeepSeek
    TE-Cap is a tie. The random basis falls behind the real reference in six of eight
    cells, so a fitted reference does beat no reference; but "fitted on unrelated prose"
    is about as good as "fitted on the benchmark's failures" half the time. The
    validation splits hold 12–26 trajectories, so single cells are noisy (the synthetic
    rows swing by 10 points between mixed and failures-only on Qwen WW-AG); the pattern
    across cells is what to trust.

  For the paper: the premise "failed trajectories reveal where agents go wrong" is not
  supported as stated. What the evidence supports is narrower — a step's energy in a
  low-rank basis, largely its norm, marks the decisive step, and an in-distribution
  reference helps on WW-AG and on DeepSeek, not consistently elsewhere. The A7 result
  (six to twelve trajectories suffice) and the transfer result fit this reading.

## B4 — Supervised probe on the validation labels (`b4_probe`)  `[CPU]`

- **Target.** Weakness 4: SOAP reads validation labels to select its configuration.
  What does a supervised method do with the same labels?
- **Procedure.** Logistic regression, one example per step, label = "is the decisive
  step", trained on the VALIDATION split only (12–25 trajectories). Balanced class
  weights, standardized features. Prediction: the within-trajectory argmax of the
  logit, scored on the test split like every other row.
- **Rows.**
  1. *Probe, spectral coordinates* — the step's 20 projections onto the reference
     split's singular vectors. Twenty features, so it can fit a dozen trajectories.
  2. *Probe, hidden state* — the raw mean-pooled vector (4096-d).
  3. *Probe, shuffled labels* — row 1 with labels permuted within trajectory; should
     fall to A10's random row. A sanity check, not a result.
  4. *Base score* and *SOAP*, for reference.
- **Selection.** The probe's layer and regularization strength C ∈ {0.01, 0.1, 1, 10}
  are chosen by leave-one-trajectory-out cross-validation inside the validation split.
  This is the one place the test rule does not apply: a probe tuned on test labels
  would be meaningless as a "same labels" comparison. Note the asymmetry when reading
  — SOAP's rows are test-selected, so the comparison favors SOAP until the validation
  conversion lands.
- **Reading.** A probe that beats SOAP on a subset must be reported, and the claim
  narrowed to the subsets where SOAP holds. A probe that loses supports the argument
  that a dozen labeled trajectories are too few to train on, yet enough to select with.
- **Output.** `results-ablations/b4_probe.tsv`.
- **Results** — DONE 2026-09-21 (`scripts/ablations/b4_probe.py`, log
  `logs/b4_probe.log`; base rows reproduced). Candidate layers: every stored layer on
  Qwen (9), every third on DeepSeek (11). Validation trajectories per seed: 26 (WW-AG),
  12 (WW-HC), 17 (TE-Cap, TE-Mag). The TSV records the layer and C chosen per seed;
  they change from seed to seed, a sign of how little a dozen trajectories pin down.
  Step accuracy %, test. The two SOAP rows come from `tables/appendix_valsel.tsv`:
  the test-selected row is Table 1, the validation-selected row is the like-for-like
  comparison, since the probe never reads a test label.

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Probe, spectral coordinates | 29.63 | 14.94 | 31.01 | 16.67 |
  | Probe, hidden state | 43.92 | 20.69 | 18.60 | 17.39 |
  | Probe, shuffled labels | 6.35 | 4.60 | 13.95 | 1.45 |
  | SOAP, validation-selected | 37.57 | 25.29 | 22.48 | 8.70 |
  | SOAP, test-selected (Table 1) | 47.62 | 34.48 | 35.66 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Probe, spectral coordinates | 32.28 | 12.64 | 32.56 | 24.64 |
  | Probe, hidden state | 35.45 | 19.54 | 20.93 | 20.29 |
  | Probe, shuffled labels | 6.35 | 4.60 | 11.63 | 2.90 |
  | SOAP, validation-selected | 40.74 | 24.14 | 31.78 | 23.19 |
  | SOAP, test-selected (Table 1) | 45.50 | 28.74 | 42.64 | 30.43 |

  Reading. Against Table 1's test-selected SOAP, no probe wins a cell; the closest is
  the hidden-state probe on Qwen WW-AG (43.92 vs 47.62). That comparison flatters SOAP.
  Like for like — both methods touching only validation labels — the picture is mixed:
  on Qwen a probe beats validation-selected SOAP on WW-AG (43.92 vs 37.57), TE-Cap
  (31.01 vs 22.48) and TE-Mag (17.39 vs 8.70), and loses on WW-HC; on DeepSeek SOAP
  wins WW-AG and WW-HC clearly and the two are within a trajectory or two on
  TraceElephant. Neither probe is reliable: which feature set wins flips by subset, and
  the cross-validated accuracy overstates test accuracy by 5–15 points. The shuffled
  probe falls to or below A10's random row, so the pipeline leaks nothing.
  For the paper: a supervised probe on the same labels is a competitive baseline, not a
  beaten one. Report it, and rest the case for SOAP on WW-HC and on DeepSeek, on its
  needing no training, and on the validation-selected numbers once they are the
  headline. This result also raises the priority of shrinking SOAP's configuration
  grid: most of the gap between its two rows is selection noise that a probe with two
  knobs does not pay.

## E5 — Success versus failure reference, MCP-Atlas (`e5_success_reference`)  `[one GPU extraction, then CPU]`  — DONE 2026-09-21

- **Target.** Weakness 3, the control E4 left open: does a reference built from FAILED
  trajectories score better than one built from SUCCESSFUL ones? No benchmark in our
  question pools ships successes, and generating them is costly and uncertain. MCP-Atlas
  has both outcomes from one agent on one task pool, so it gives the clean contrast, at
  the price of a cross-distribution fit.
- **Corpus.** `../attrib-prompting/vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B/` — 198
  single-agent tool-use runs. Outcome follows OAT's split
  (`data_pipeline.load_mcp_atlas_trajectories`): a run is a success when its `errors`
  list is empty. That gives 103 successes — the corpus OAT trains on — and 95 failures.
  OAT drops a failure whose errors carry no valid step; we drop the same ones. The two
  pools are alike in size: median 13 messages per run in both, 1,436 vs 1,435 messages
  in all.
- **Rows.** Two references, nothing else:
  1. *Successes* — 50 runs sampled from the 103.
  2. *Failures* — 50 runs sampled from the failures.
  One draw, seed 0, without replacement; the runner takes `--draw-seed` so a second draw
  costs one CPU run. It records each arm's step count; if the two differ by more than
  5 %, redraw the larger arm to match before reading anything.
  For context the tables repeat E4's benchmark, WikiText and random rows; they are not
  recomputed.
- **Staging.** `scripts/ablations/e5_stage_mcp_atlas.py` writes all 198 runs to
  `data/synthetic/mcp-atlas/<id>.json` in `main.data`'s minimal schema, as
  `e4_stage_wikitext.py` does, plus `"outcome": "success" | "fail"` (the field E4's
  `failed_files` already reads). One step = one message of `raw_conversation_history`,
  with OAT's serialization so SOAP and OAT read the same text: reasoning content, then
  content, then tool calls; tool output cut at 4,096 characters. `role` carries the
  message role. `mistake_step = -1`: the error annotations are not read.
- **Extraction.** Add `mcp-atlas` to `configs-main/synthetic.yaml`, then once per
  backbone: `python -m main extract --config configs-main/synthetic.yaml --stage
  activations --subset mcp-atlas`. No attention pass — dependency weights come from the
  target's own trajectories.
- **Procedure.** E4's, unchanged: fit R on the arm, score the target's real val and
  test splits on the frozen triple. The runner imports `Reference`, `base_grid` and
  `rescore_grid` from `e2_synthfit.py` and builds the two arms as static references
  with `files=` the sampled list. All eight cells (both backbones × WW-AG, WW-HC,
  TE-Cap, TE-Mag), base and SOAP rows.
- **Three tables per backbone**, one per selection mode:
  1. *Frozen* — Table 1's configuration, nothing re-tuned.
  2. *Validation-selected* — the full configuration re-selected per reference on mean
     validation accuracy; test accuracy reported (`--select-rule val`).
  3. *Test-selected* — re-selected on mean test accuracy, Table 1's protocol
     (`--select-rule test`).
- **Self-check.** The `real` reference is run alongside and must reproduce
  `selection.tsv` to 1e-9 under the test rule, and E4's validation rows under the val
  rule, before either arm is written. Staged file count must equal the kept runs.
- **Reading.** Decide before the run:
  - *Failures beat successes* by more than two test trajectories in most cells, under
    the frozen AND validation tables → first evidence for "failures matter".
  - *The two tie* (the likely outcome, given E4) → the defensible claim is that failed
    trajectories SUFFICE, and they are what a practitioner has. Rewrite the premise so.
  - *Successes beat failures* → the mixture story of Section 2 is wrong as stated.
  - The test-selected table is reported for continuity with Table 1 only; E4 showed it
    cannot separate references. A gap that appears there alone is not evidence.
- **Caveat.** Both arms are fit out of distribution, so they may both sit near E4's
  WikiText row, leaving no room for a difference. E6 is the in-distribution complement.
- **Output.** `results-ablations/e5_success_reference.tsv` (columns as E4, plus
  `n_steps` and `draw_seed`).
- **Results** — DONE 2026-09-21 (`scripts/ablations/e5_success_reference.py`, staging
  `scripts/ablations/e5_stage_mcp_atlas.py`; four runs, one per backbone and rule,
  merged from `results-ablations/e5_parts_*.tsv` with a `select_rule` column; logs
  `logs/e5_*.log`). Staging kept 191 runs: 103 successes, 88 failures; OAT's rule
  dropped 7 failures whose errors name no valid step. Activations:
  `results-nogt/synthetic/activations/<model>/mcp-atlas/`. The seed-0 draw holds 715
  success steps and 751 failure steps, 4.8 % apart, so no redraw. The real reference
  reproduced the selection table in all eight cells under both rules. E4's loop was
  made reusable for this (`e4_reference_controls.main(build, order, out, extra_args)`);
  E4 itself was not rerun. SOAP step accuracy %, test (base rows are in the TSV):

  **Frozen — Table 1's configuration**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 47.62 | 34.48 | 35.66 | 23.19 |
  | MCP-Atlas successes | 25.93 | 24.14 | 32.56 | 20.29 |
  | MCP-Atlas failures | 33.86 | 24.14 | 31.01 | 19.57 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 45.50 | 28.74 | 42.64 | 30.43 |
  | MCP-Atlas successes | 31.75 | 22.99 | 26.36 | 24.64 |
  | MCP-Atlas failures | 30.69 | 18.39 | 28.68 | 25.36 |

  **Validation-selected**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 37.57 | 25.29 | 22.48 | 8.70 |
  | MCP-Atlas successes | 24.34 | 19.54 | 27.13 | 10.14 |
  | MCP-Atlas failures | 37.04 | 17.24 | 17.83 | 22.46 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 40.74 | 24.14 | 31.78 | 23.19 |
  | MCP-Atlas successes | 27.51 | 20.69 | 26.36 | 17.39 |
  | MCP-Atlas failures | 35.45 | 20.69 | 28.68 | 21.74 |

  **Test-selected — the protocol of Table 1**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 47.62 | 34.48 | 35.66 | 23.19 |
  | MCP-Atlas successes | 38.10 | 28.74 | 33.33 | 22.46 |
  | MCP-Atlas failures | 42.86 | 28.74 | 33.33 | 22.46 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Benchmark train split | 45.50 | 28.74 | 42.64 | 30.43 |
  | MCP-Atlas successes | 40.21 | 29.89 | 40.31 | 29.71 |
  | MCP-Atlas failures | 44.44 | 32.18 | 35.66 | 28.99 |

  Reading, by the rule set before the run. "Two test trajectories" is 3.2 points on
  WW-AG, 6.9 on WW-HC, 4.7 on TE-Cap and 4.3 on TE-Mag.
  - **The two arms tie.** Failures beat successes by more than two trajectories in the
    frozen AND the validation table in one cell of eight: Qwen WW-AG (+7.9 and +12.7).
    Everywhere else the frozen gap is within ±5 points with both signs, and the
    validation gaps swing from −9.3 (Qwen TE-Cap) to +12.3 (Qwen TE-Mag) — the size of
    the noise E4 already measured for validation selection on these splits. The rule's
    bar, "most cells, both tables", is not met.
  - **WW-AG leans toward failures, and only there.** Failures lead on WW-AG in five of
    the six tables (+4 to +13); the exception is DeepSeek, frozen (−1.1). WW-AG is also
    the one subset whose band skips the mean direction (A10), so it is the cell most
    able to show a difference in what the reference holds. One draw of 50 runs cannot
    carry a claim; a second `--draw-seed` would say whether the lean survives.
  - **Distribution matters more than outcome.** At the frozen configuration both
    MCP-Atlas arms fall 10–20 points below the benchmark reference on WW-AG and 5–16
    elsewhere on DeepSeek, far more than they differ from each other. Re-selected on
    test, both climb back to within a few points of Table 1, as every reference in E4
    did.
  - **For the paper:** the evidence supports "failed trajectories suffice", not
    "failed trajectories are what matter". A success reference from the same corpus
    does as well on seven cells of eight.

## E6 — Reference from the steps before the decisive error (`e6_prefix_reference`)  `[CPU]`  — DONE 2026-09-21

- **Target.** The same question, in distribution. By the paper's own account the steps
  before the decisive step `t*` are ordinary: nothing has gone wrong yet. A reference
  fit on them alone is the nearest thing to a success-only reference that the
  benchmarks allow. It is a proxy — these steps come from runs that later fail — and it
  reads step labels on the reference split, so it is a DIAGNOSTIC, never a variant of
  the method.
- **Rows.**
  1. *Full reference* — every step of the seed's train split. Table 1; must reproduce.
  2. *Prefix* — the train split's steps with `step_idx < mistake_step` only.
  3. *Random subset* — steps drawn uniformly without replacement from the same train
     split, as many as row 2 holds for that seed. Draw seed = the split seed. This
     separates "fewer rows" from "earlier, cleaner rows".
- **Procedure.** A `Reference` subclass masks the rows of the train store before
  `main.score.fit_svd`; the ensemble position's z-score statistics come from the masked
  rows too. Validation and test splits, scoring, and dependency weights are untouched.
  The mask uses the store's own `(traj_idx, step_idx)` index; the runner asserts that
  the store's labeled step equals the JSON's `mistake_step` for every train trajectory
  (`documents/CONVENTIONS.md` on step indexing) before masking. A trajectory with
  `mistake_step = 0` contributes no row to the prefix arm. No extraction: every
  activation exists.
- **Coverage and tables.** All eight cells, base and SOAP rows, and the same three
  tables as E5: frozen, validation-selected, test-selected.
- **Recorded per cell and seed.** Rows in each arm. On WW-AG the mean `t*` is 3.0 over
  37 train trajectories, so expect about 110 rows against 20 fitted components — enough,
  but thin. If an arm has fewer than 40 rows in any seed, flag the cell.
- **Self-check.** Row 1 reproduces `selection.tsv` to 1e-9; row 3 drawn at the full
  size reproduces row 1.
- **Reading.**
  - *Prefix ≈ random subset ≈ full* → the decisive and later steps add nothing to R and
    remove nothing from it; the reference captures the common mode of steps, and the
    contamination in Section 2's mixture is too small a share of the rows to shape the
    top directions.
  - *Prefix above random subset* → cleaner rows help; failure steps pollute R. This
    argues AGAINST "failures matter".
  - *Prefix below random subset* → the later steps carry something the fit needs — but
    the prefix is also all early steps, so position explains it as well as failure
    does. State both readings; this design cannot separate them.
- **Output.** `results-ablations/e6_prefix_reference.tsv` (columns as E4, plus
  `n_rows`).
- **Results** — DONE 2026-09-21 (`scripts/ablations/e6_prefix_reference.py`; four
  runs merged from `results-ablations/e6_parts_*.tsv` with a `select_rule` column; logs
  `logs/e6_*.log`). The store's labeled step matched the JSON's `mistake_step` in every
  train trajectory, and the full reference reproduced the selection table in all eight
  cells under both rules. The random subset is one fixed draw per split, seeded by the
  split's file list. Rows in the prefix arm, mean over the triple (minimum; full
  split): WW-AG 109 (93; 327), WW-HC 221 (178; 784), TE-Cap 157 (150; 539), TE-Mag 341
  (216; 802) — no seed fell below 40. SOAP step accuracy %, test:

  **Frozen — Table 1's configuration**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 47.62 | 34.48 | 35.66 | 23.19 |
  | Steps before t* | 33.33 | 33.33 | 34.11 | 23.91 |
  | Random subset, same size | 40.74 | 31.03 | 34.88 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 45.50 | 28.74 | 42.64 | 30.43 |
  | Steps before t* | 35.45 | 26.44 | 28.68 | 28.99 |
  | Random subset, same size | 39.15 | 25.29 | 41.86 | 28.26 |

  **Validation-selected**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 37.57 | 25.29 | 22.48 | 8.70 |
  | Steps before t* | 25.40 | 25.29 | 24.81 | 12.32 |
  | Random subset, same size | 40.74 | 21.84 | 22.48 | 8.70 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 40.74 | 24.14 | 31.78 | 23.19 |
  | Steps before t* | 37.57 | 10.34 | 34.88 | 23.19 |
  | Random subset, same size | 25.93 | 24.14 | 31.78 | 22.46 |

  **Test-selected — the protocol of Table 1**

  | qwen3.5-9b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 47.62 | 34.48 | 35.66 | 23.19 |
  | Steps before t* | 36.51 | 33.33 | 35.66 | 25.36 |
  | Random subset, same size | 39.68 | 33.33 | 36.43 | 23.19 |

  | deepseek-8b | WW-AG | WW-HC | TE-Cap | TE-Mag |
  |---|---|---|---|---|
  | Full reference | 45.50 | 28.74 | 42.64 | 30.43 |
  | Steps before t* | 44.97 | 31.03 | 34.88 | 31.16 |
  | Random subset, same size | 39.15 | 27.59 | 42.64 | 29.71 |

  Reading.
  - **On WW-HC, TE-Cap (Qwen) and TE-Mag the three references tie** at the frozen
    configuration, within one test trajectory. There the decisive and later steps add
    nothing to R and take nothing from it: the first outcome of the reading rule. A
    third of the rows, chosen either way, fits the same common mode.
  - **Nowhere does the prefix beat the random subset by a margin that holds in two
    tables.** So there is no sign that failure steps pollute R, and none for a cleaner
    "ordinary-step" reference.
  - **The prefix falls below the random subset in three frozen cells:** WW-AG on both
    backbones (−7.4, −3.7) and DeepSeek TE-Cap (−13.2). This is the third outcome, and
    the design cannot say why. On WW-AG the prefix is the first three turns of each
    trajectory — the task statement and the opening plan — so "too few kinds of step"
    explains the drop as well as "the later, failing steps carry signal" does.
  - **Size alone costs WW-AG 6–7 points** (random subset 40.74 and 39.15 against 47.62
    and 45.50), in line with A7's data-quantity curve.
  - **Validation selection adds noise larger than any of these effects**: the random
    subset beats the full reference on Qwen WW-AG (40.74 vs 37.57) and loses 15 points
    to it on DeepSeek; the prefix drops to 10.34 on DeepSeek WW-HC. Read the frozen
    table; the other two are reported for the record.
  - **For the paper:** with E5, this closes the success-only question as far as the
    available data allows. R captures what typical steps of the target system look
    like; whether those steps come before or after the error, or from runs that fail
    or succeed, changes little.

## Summary of findings

What the four experiments say about the three claims the reviewer challenged, and what
to do about each.

1. **"Attention propagation resolves downstream contamination" — true on WW-AG, not
   shown elsewhere (A11).** On WW-AG no attention-free control comes within 6 points of
   SOAP and the corrected predictions are not final-step predictions. On TE-Cap the
   whole gain is a final-step penalty; on WW-HC uniform averaging ties SOAP; TE-Mag
   keeps a gain of one to three trajectories. On CORRECT-Error (A11 extension) every
   Qwen row sits within half a point and the next-step shift beats SOAP (62.08 vs
   61.78); only DeepSeek wikimqa shows routing (+3.45), and length normalization erases
   it. And on WW-AG the gain depends on raw
   attention mass: normalizing by step length or dropping the first turn removes it.
   *Action:* scope the claim to WW-AG, describe the rest as a final-step correction,
   report the length and first-turn rows, replace the featured example. *Small
   follow-up:* re-select the attention band and w for the two variants.
2. **"Failed trajectories are what matter" — not supported (E4).** Under Table 1's
   protocol WikiText reaches the headline numbers and a random basis comes within a few
   points of them (worst of 11 draws: 9–14 below on WW-AG, 1 below on DeepSeek WW-HC).
   Under validation
   selection the benchmark reference leads on WW-AG and on DeepSeek, and loses to
   WikiText on three of four Qwen cells. Failure-only filtering changes nothing.
   *Action:* reframe the premise as a low-rank reference for the common mode of step
   representations, helped by in-distribution data on some subsets; drop the Huber
   mixture story or label it motivation only. The two stand-ins for the success-only
   control agree (E5, E6): MCP-Atlas successes and failures tie on seven cells of
   eight, with a lean toward failures on WW-AG from a single draw, and a reference cut
   to the steps before the decisive error does no better than a random subset of the
   same size. Claim that failed trajectories SUFFICE, not that they are required.
3. **"The score is not a length or norm artifact" — half true (A10).** Length
   baselines sit near random. But wherever the band starts at component 0 the score is
   nearly a function of the vector norm (ρ −0.70 to −0.94), and removing the norm costs
   3–13 points. *Action:* say so, and report the unit-norm row.
4. **Same labels, supervised (B4).** No probe beats test-selected SOAP; against
   validation-selected SOAP a probe wins three of four Qwen cells and roughly ties on
   DeepSeek TraceElephant. *Action:* report the probe as a competitive baseline.

5. **Found on the way: on CORRECT-Error the final step is usually the decisive one
   (A11 extension).** Excluding the final step drops the base score from 61–64 % to
   11 %, and over the whole corpus the gold step is the last turn in 57.5 % of
   trajectories (macro; 81.6 % on arc, 36.5 % on musique). "Always predict the final
   step" therefore lands within 4–7 points of SOAP and of the best judges in the CE
   column. *Action:* add a final-step row to Table 1 — this restores the fixed-index
   baseline dropped below, for the one column where it matters — and stop citing
   CORRECT-Error as evidence for rescoring. *Small follow-up:* score the rule on the
   three test splits; the 57.5 is a corpus-wide share.

The thread through all four: SOAP's headline numbers lean on selecting among thousands
of configurations with a few dozen labeled trajectories. E4's random-basis row measures
that lean directly — on Qwen WW-AG, 33.33 to 43.92 across 11 random bases under test
selection (median 39.15), 23.28 under validation selection. Shrinking the grid (a handful of layers, band families [0, k) and [1, k),
fixed w) is the change most likely to move the validation-selected numbers toward the
test-selected ones, and it is the work to do next, ahead of any rewriting.

## Excluded by decision (2026-09-21)

- **Success-only reference in our own question pools** — successful trajectories exist
  only in the synthetic corpora, and obtaining a clean, size-matched set is nontrivial.
  Replaced by two stand-ins: E5 (MCP-Atlas successes vs failures, cross-distribution)
  and E6 (steps before the decisive error, in distribution).
- **Few-shot judge** with validation demonstrations — not needed.
- **Fixed-index, first-non-orchestrator, agent-prior baselines** — dropped; A3's
  temporal-bias rows already cover position. The CORRECT-Error run reopens one of them:
  a final-step baseline (Summary, item 5).
- **Probe trained on train+val labels** — dropped; the probe sees validation labels only.

## Order

A10 and A11 first (CPU, minutes to run); then B4; then E4, whose WikiText extraction
needs a free GPU (0–3 were idle on 2026-09-20). E6 next (CPU only, no staging); then
E5: stage MCP-Atlas, extract once per backbone, run the two arms under both rules.
