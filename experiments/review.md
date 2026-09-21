# Review-response experiments

Four experiments that answer Weaknesses 2–4 of `SOAP_ICLR2027_review.pdf`. Drafted
2026-09-21 for review: **nothing here has run**. Strike or edit any row, then approve;
each finished experiment gets a **Results** block in this file.

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

## Excluded by decision (2026-09-21)

- **Success-only reference** — successful trajectories exist only in the synthetic
  corpora, and obtaining a clean, size-matched set is nontrivial. Deferred.
- **Few-shot judge** with validation demonstrations — not needed.
- **Fixed-index, first-non-orchestrator, agent-prior baselines** — dropped; A3's
  temporal-bias rows already cover position.
- **Probe trained on train+val labels** — dropped; the probe sees validation labels only.

## Order

A10 and A11 first (CPU, minutes to run); then B4; then E4, whose WikiText extraction
needs a free GPU (0–3 were idle on 2026-09-20).
