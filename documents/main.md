## The method, in brief

SOAP localizes the decisive-error step in a failed multi-agent trajectory, using
only step-unlabeled failure trajectories and a frozen proxy model — no step labels,
no fine-tuning, no success-only corpus. Two components:

1. **Spectral base score.** Encode each step in its generation context (one forward
   pass, mean-pool the step's own tokens at a chosen layer). Fit an SVD on the
   pooled reference-split steps; score each step by its mean squared projection onto
   a contiguous band `C = [c_begin, c_end)` of right-singular directions. Errors
   project smaller, so the score is inverted: `S = 1/(π_C + ε)`.
2. **Attention-guided rescoring.** Dependency weights `w_{i,t}` = fraction of step
   `t`'s attention mass landing in predecessor `i` (head-averaged over a layer band,
   renormalized over predecessors). Each step collects the dependency-weighted
   average of its downstream dependents' base scores: `S̃(s_i) = S(s_i) + γ·B_i`.
   Single-pass over the original scores; γ=0 recovers the base score. This corrects
   **downstream contamination** — a per-step scorer's argmax otherwise lands after
   the true error, because every later step conditions on the corrupted context.

Prediction: within-trajectory argmax of `S̃`; the responsible agent is the agent of
the predicted step.

The implementation's reported strategy is top-w `backprop` (each step keeps only its
w strongest predecessors before propagation). The manuscript's method section
describes the `w="all"` form and **deliberately stays as-is for now** — a known
revision item, not a misunderstanding.

## What produces the reported numbers

Everything in the paper comes from **`main/`** (the frozen-axis runner), not `src/`
(the fully-sweepable reference implementation). The chain:

`main/` sweep on the frozen triples → `results-nogt/` / `results-gt/` →
`scripts/prompting/evaluate.py` (baselines on the same test splits) →
`scripts/tables/make_main_tables.py` → `tables/*.tsv` → LaTeX.

## Evaluation protocol, as actually practiced now

- **Data**: 5 reported subsets — WW-AG (126), WW-HC (58), CE (2,226; macro-average
  over its 7 internal subsets on one shared triple), TE-Cap (85), TE-Mag (91).
  Trajectory-level 30/20/50 reference/val/test split. A sixth, **AgenTracer
  TracerTraj-code (`agentracer/code`, 127)**, was extracted, swept and frozen on
  2026-09-07 (`experiments/todo.md`, D1) but is NOT in any table or the manuscript.
  The 2026-09-10 audit in D1 found its SOAP number to be a label artifact (58 % of
  gold turns are prompt-template dumps; SOAP scores 0 % on the rest) — do not report.
- **Seed triples are DONE.** The selection phase is finished; the triples are
  hand-picked by the author, frozen in `configs-main/<ds>.yaml`, one per subset,
  shared by both backbones. The per-triple sweep machinery (`scripts/main/`)
  produced candidates, but the final triples are the author's call — including the
  WW-HC manual override [13, 14, 15]. Do not re-run triple selection. The one
  exception is `agentracer`, whose triples ([20, 21, 22] without GT, [15, 16, 17]
  with GT) came straight from `pick_triple.py` on 2026-09-07 — `sum` and `sum-diff`
  agree — because it is a new cell, not a re-selection.
- **Config selection is test-selected, deliberately.** Within the frozen triple, the
  winning config maximizes mean TEST step accuracy. This is an optimistic interim
  choice; the plan is to convert to validation-selected later (`pick_triple.py
  --rule val` plus the `select_config` rule change covers both layers). The
  manuscript's "validation labels for hyperparameter selection" sentence describes
  the intended final protocol, not the current numbers.
- **Reporting**: mean over the triple's three seeds. Headline metric: step-level
  accuracy (exact match); agent-level and ranking metrics go to the appendix.
- **Backbones**: prompt-based baselines run on GPT-4o (GPT-5 counterpart destined
  for the appendix), scored on SOAP's exact frozen test splits; SOAP and other
  non-prompt methods on Qwen3.5-9B and DeepSeek-R1-Distill-Llama-8B.

## Manuscript state (work in progress)

- **Experiments section prose complete** (2026-08-31, v2.5 header note in
  `sections/experiments.tex`): every `\TODO{Results and analysis.}` filled in
  REDE style — tab:main / tab:main-gt / synthetic written fresh; the commented
  drafts for scale, transfer, scorefn, weights, sensitivity activated. Setup
  fixed: two baseline families matching the table groups (AgenTracer/GraphTracer
  dropped), StepFinder described, with-GT adaptation stated (gold answer
  appended to the task description in the proxy context), judge = GPT-4o.
  "Different metrics." stub retired to comments.
- **Tables 1 and 2 and the ablation/experiment numbers are current** (tables
  refreshed 2026-08-11 from `tables/*.tsv`; ablations/E1/E2/S1 filled from
  `results-ablations/` per `experiments/todo.md`).
- **Only uncommented .tex text is the paper.** Commented blocks are dead history
  (CRR/ReCAP naming, `contamination.tex`, the old method formulation) kept for
  reference; ignore them when reading or revising.
- **Empty cells are pending work, not omissions**: OAT, StepFinder (placeholder bib
  entry), AgenTracer, GraphTracer, RAFFLES rows; the with-GT SOAP adaptation
  description; the method figure and all figure placeholders.
- **Main-text figures merged 2026-09-02** to match the reorganized experiments
  section: `fig_scale_transfer.pdf` (scale WW-AG, scale WW-HC, transfer heatmap,
  synthetic reference) and `fig_ablations.pdf` (gamma, representation layer,
  attention band on WW-AG, plus reference-data quantity) replace the four
  separate figures; the qualitative example (`scores_captain_traj1.pdf`) stands
  alone in a wrapfigure. Generator: `scripts/ablations/plot_figures.py`
  (`--only fig_scale_transfer --only fig_ablations`); the unmerged predecessors
  stay in the script. New PDFs live in `artifacts/ablations/` and
  `manuscript/assets/` but must be uploaded to Overleaf by hand (`pull.sh` only
  pulls).

- **Appendix rewritten 2026-09-03** (`sections/appendix.tex`, REDE structure A/B/C).
  Every `app:*` label the main text cites now resolves: datasets + statistics
  (`tables/appendix_datasets.tsv`), input format and truncation, implementation
  details with the selected configurations (`tables/appendix_anchors.tsv`),
  baselines (prompting paragraph added to the OAT/StepFinder text), the selection
  protocol with the test-vs-validation-selected table (`tables/appendix_valsel.tsv`,
  from `results-ablations/e3_valsel_indist.tsv`), the with-GT grid
  (`appendix_with_gt_full.tsv`), compute, synthetic generation; ablation mirrors
  (restored from commit `2caa168`) plus two NEW ablations — last-token vs mean
  pooling (`results-ablations/a8_pooling.tsv`, `scripts/ablations/a8_pooling.py`)
  and accuracy at k / MRR for SOAP, OAT and StepFinder
  (`results-ablations/a9_ranked.tsv`, `scripts/ablations/a9_ranked.py`; `mrr` added
  to `main/metrics.py`); agent-level, per-seed std, GPT-5, scale, transfer, synthetic
  and qualitative appendices. Builder: `scripts/tables/make_appendix_tables.py`.
- **Main-text edits made with it**: `tab:main-gt` prompting rows filled with the
  Qwen3.5-9B judge (results-prompting/ was regenerated 2026-09-03 with those cells;
  RAFFLES 46.56 now beats SOAP 43.39 on WW-AG with GT, prose adjusted); the setup and
  implementation-details sentences now state test selection and point to
  `app:selection`; method section states the top-w trimming; the blue/red notes
  covered by the appendix removed.
- **Textual qualitative examples added 2026-09-11** (inline in `sections/appendix.tex`,
  `app:qualitative`; two `figure[p]` pages, two trajectories each, one
  per subset, Qwen3.5-9B on the frozen triples): WW-AG 65 (seed 3), WW-HC 23 (seed
  15, the ONLY base-wrong/SOAP-right flip on that triple, 73 steps, set at 6pt with
  the orchestrator cycles collapsed), TE-Cap 50 (seed 23), TE-Mag 30 (seed 15; flips
  on all three seeds). Scores are `base`/`final` x 10^3 from
  `results-nogt/<ds>/reproduce/qwen3.5-9b/<subset>/backprop_seed-<s>_test.steps.tsv`;
  `traj_idx` there is the JSON file stem. The TraceElephant reproduce tree was
  regenerated on the frozen seeds for this (captain 22-24, magentic 15-17); the stale
  pre-freeze files (captain 14-16, magentic 8-10) still sit beside them. The main
  text's qualitative paragraph now points at `app:qualitative` (was an empty `\ref{}`).
  Titles name only subset and agent system. Preview:
  `artifacts/qualitative_examples/textual_examples_preview.pdf`.
- **`app:transfer` rewritten to the validation convention only 2026-09-08**: the
  test-selected grids and the two-convention comparison are commented out;
  `fig_transfer_appendix.pdf` regenerated as a 1x2 val-only figure
  (`scripts/ablations/plot_figures.py`, old 2x2 layout noted in a comment) and
  copied to `manuscript/assets/` — upload to Overleaf by hand. Off-diagonals come
  from `e1_transfer.tsv` conv=val (target val pools reference+val, 40%); diagonals
  stay Table-1 per E1's rule. NOT the E3 protocol (`e3_valsel_transfer.tsv`, main
  20% val split) — swap the TSV if the headline converts to val selection. The
  main text's DeepSeek transfer pointer now refs `app:transfer` (was
  `app:deepseek-ablations`).
- **ErrorProbe added 2026-09-05, switched to the paper mode 2026-09-06**
  (`\citep{errorprobe}`, Li et al., Findings of ACL 2026; reproduced in
  `../attrib-prompting/baselines/errorprobe/`). The row in `tab:main` (Qwen and
  DeepSeek blocks) and `tab:main-gt` now reports `errorprobe_paper` — the paper's
  tag -> backward-trace -> three-role pipeline (see B3 in `experiments/todo.md`);
  the truncated-history and backward-tracing modes live on in .tex comments. The
  swap moved markers: Qwen block, ErrorProbe is second on CE (62.42 vs SOAP
  61.78), best on TE-Mag (25.36), and ties RAFFLES on TE-Cap (29.46, both
  underlined); DeepSeek TE-Mag second returned to Binary Search; with-GT WW-AG
  reads RAFFLES 46.56 / ErrorProbe 43.92 / SOAP 43.39. Three prose spots are now
  stale: the with-GT paragraph ("second best on WW-AG"), the main-comparison
  claim ("best on all five subsets with both backbones", false since the RAFFLES
  fill), and the Baselines description ("analyzer--verifier diagnoser" names the
  truncated mode). The GPT-4o paper-mode run landed and fills the GPT block row
  (2026-09-07: 38.62 / 21.84 / 59.52 / 27.13 / 28.26, best on all five columns
  of that block, which now carries full markers). NOTE the block states above
  predate two 2026-09-06 changes made in the .tex directly: RAFFLES was dropped
  from the comparison (kept in related work; rows commented out) and the
  Qwen-block/`tab:main-gt` ErrorProbe rows moved to the `qwen3.5-9b-weak` judge
  (same checkpoint, handicapped decoding) — the dated .tex comments and B3 in
  `experiments/todo.md` are current. `tab:main-gt`'s six prompting rows were refilled from
  `tables/open_backbones_step_with_gt.tsv` (Qwen3.5-9B judge); RAFFLES 46.56 beats
  SOAP 43.39 on WW-AG and the prose says so. Named in the Baselines paragraph. NOT yet
  in the appendix: `app:baselines` prose, `tab:gt-full`, the std/agent tables, GPT-5.
- **ErrorProbe Qwen rows switched to the `qwen3.5-9b-weak` judge 2026-09-06** (user
  request): the same Qwen3.5-9B checkpoint decoded on a handicapped budget (512 new
  tokens, temperature 1.0, top-p 0.95, 16k window clipped from the front; config
  comment in `../attrib-prompting/baselines/errorprobe/configs/ww.yaml`). Paper mode,
  both `tab:main` (Qwen block) and `tab:main-gt`. `scripts/prompting/evaluate.py`
  now scores that judge; `open_backbone_rows.py` writes it as the "Qwen3.5-9B (weak)"
  block of `tables/open_backbones_*.tsv`. The full-budget rows it replaced are kept
  in .tex comments. The table labels the row "Qwen3.5-9B" with no mention of the
  reduced decoding budget; disclose it in the caption or `app:baselines` before
  submission.
- **RAFFLES dropped from the comparison 2026-09-06** (user decision; it stays in
  related work only). Every RAFFLES table row in `experiments.tex` and `appendix.tex`
  (tab:main, tab:main-gt, tab:gt-full, tab:agent, tab:std, tab:gpt5) is commented
  out under a "RAFFLES dropped" comment, best/second marks were recomputed per block,
  and the seven prose sentences that cited RAFFLES numbers were rewritten with the
  old text kept in comments (ErrorProbe is now the strongest prompt-based baseline in
  the Qwen block). The prompting pipeline and `tables/*.tsv` still carry RAFFLES; only
  the LaTeX changed. The RAFFLES discrepancy item below is therefore moot for the
  manuscript.
- **`app:compute` rewritten 2026-09-15** (user decision: report the extraction
  difference faithfully, no single-pass argument). Three tables: `tab:cost-setup` (once
  per subset: corpus, labels, extraction passes/tokens/time/peak memory, then training or
  SVD time), `tab:cost-inference` (per trajectory: the same extraction columns, then the
  scorer in ms on CPU / GPU), `tab:cost-judges` (one trajectory at a time on the
  Qwen3.5-9B judge vs SOAP on the same in-budget sample). Every number is timed as run
  through each method's own code (C4 in `experiments/todo.md`; scripts under
  `scripts/ablations/c4_*.py` and `../attrib-prompting/scripts/cost_rb_*.py`; joined
  TSVs `results-ablations/c4_cost_*.tsv`). Message: SOAP's per-step-in-context
  extraction is the most expensive stage (5.5 s / 161 s per WW-AG / WW-HC trajectory vs
  0.4 / 2.1 for OAT) but its peak memory is bounded by the 8,192 budget (25 GB max)
  while OAT's grows with the trajectory (31.6 GB on the longest); after extraction SOAP
  is cheapest (SVD 0.15--0.30 s vs 255--577 s of training; scorer in ms). The old
  subsection (batched judge throughput, StepFinder mislabeled as single-pass, July
  mtimes) is kept commented.
- **`app:compute` reduced to two tables and revised 2026-09-17** (user decisions):
  `tab:cost-inference` (fitting s/seed, one extraction forward pass with peak GB, CPU
  scoring ms/trajectory at 4 pinned threads) and `tab:cost-judges` (time per trajectory
  on the in-budget sample). SOAP is charged ONE forward pass per trajectory (the
  one-pass convention, stated in the text: causal attention makes one pass yield every
  step's states; the per-step extractor is an implementation choice), timed by
  `scripts/ablations/c4_onepass_cost.py`: 0.38 / 0.76 s (all), 0.36 / 0.62 s (sample),
  plus 0.7 / 5.5 ms scoring. StepFinder's 63.6 / 88.2 ms CPU scoring of 09-15 was a
  torch lazy-thread-pool artifact; pinned it is 1.8 / 5.0 ms (C4 in
  `experiments/todo.md`). The three-table 09-15 version stays commented in the .tex.
- **Open discrepancy**: `tab:main`'s GPT-4o RAFFLES row (35.98 / 18.39 / 53.05 /
  23.26 / 23.91) and DeepSeek RAFFLES CE (35.84) do not match
  `results-prompting/by_column.tsv` and `tables/table1_without_gt.tsv`
  (42.86 / 25.29 / 64.22 / 24.81 / 17.39; 41.60). The author chose to keep the
  manuscript numbers (2026-09-03); the appendix's std/agent tables use the TSVs, so
  the two disagree until the source of the manuscript row is reconciled.

- **Cost table and appendix fill 2026-09-13** (plan: one backbone qwen3.5-9b, WW-AG +
  WW-HC; no validation-selected headline; context-budget ablation and Qwen3.5-4B
  dropped by decision).
  - `tab:cost` added to `app:compute` (placeholder cells until the TSV lands; lead-in
    prose written). Numbers: `results-ablations/c1_cost.tsv` from
    `scripts/ablations/c1_cost.py`, which joins `../attrib-prompting/reports/cost_ww.tsv`
    (`../attrib-prompting/scripts/cost_report.py`: exact offline replay of every
    method's generator program with the stored responses, then tokenized) with wall-clock
    harvested from the predictors' `wrote ... (N/N files, Xs)` log lines and SOAP's own
    token count per trajectory. Outstanding GPU runs (ErrorProbe paper-mode timing, the
    six legacy with-GT open-judge prompting cells, SOAP's timed extraction) are listed
    with commands in `../attrib-prompting/scripts/TODO.md`; C1 in `experiments/todo.md`.
  - **The `qwen3.5-9b-weak` judge never existed as a directory.** The handicap is applied
    in place to the `qwen3.5-9b` spec of the ErrorProbe config, so
    `outputs*/ww/*/qwen3.5-9b/errorprobe_paper/` IS the reported weak run; the
    full-budget run was overwritten 2026-09-06 and lives only in .tex comments.
    `scripts/prompting/evaluate.py` no longer lists the phantom judge; the "lost weak
    rows" of 2026-09-07 were a naming artifact, no data was lost.
  - ErrorProbe rows added to `tab:agent`, `tab:std`, `tab:gt-full`, `tab:gpt5` (builders
    `make_appendix_tables.py` / `make_main_tables.py` now carry `errorprobe_paper`; TSVs
    rebuilt, existing cells unchanged). Marker moves: GPT-4o with-GT block, ErrorProbe
    best on WW-AG/WW-HC/TE-Cap/TE-Mag; GPT-5 table, ErrorProbe best on WW-AG in both
    settings (49.21 vs SOAP 47.62 without GT; 53.44 with), prose rewritten. GPT-5 blocks
    removed from `tab:agent` and `tab:std` (kept in comments). Handicap disclosure
    sentence added to `app:baselines`; the "neither baseline predicts an agent" sentence
    now names OAT/StepFinder (the Who&When prompts do ask for an agent name).
  - Dangling refs fixed (`tab:seeds` -> prose; `fig:sensitivity` -> `fig:datasize`);
    the "Additional analysis" paragraph now cites the pooling, remaining-cell ablation,
    and input-format appendices; the context-budget comment in `method.tex` retired.

- **`tab:gt-full` complete 2026-09-16.** The last blank cell, ErrorProbe (paper mode) with
  GPT-4o on CORRECT-Error with the answer, landed in
  `../attrib-prompting/outputs/correct-error/*/gpt-4o/errorprobe_paper` (all seven pools,
  2,226 predictions, none missing). `scripts/prompting/evaluate.py` rescored every cell and
  `make_appendix_tables.py` rebuilt `tables/appendix_with_gt_full.tsv`; the cell reads
  62.26 and takes the CE best mark in the GPT-4o block from Step-by-Step (59.29), so
  ErrorProbe is now best on all five columns of that block. The prose's "one blank cell"
  sentence and the caption's "a blank cell was not run" are gone. Every printed cell of the
  table was checked against the TSV (26 rows, 0 mismatches).

## Known items for later

- ~~Method section: state the top-w trimming~~ — done 2026-09-03 (one sentence after
  Eqn. recap, details in `app:implementation`).
- Convert config selection from test to validation (the appendix now reports both;
  `app:selection` must be rewritten if the headline numbers switch).
- Reconcile the RAFFLES row (above).
- `tests/test_main.py::test_configs_main_declare_frozen_triples` fails on
  `configs-main/synthetic.yaml` (extraction-only, no seeds) — pre-existing; skip that
  file in the test or give it an empty seeds block.
- Documentation drift: `configs-main/ww.yaml`'s comment says the triples came from
  the `sum-diff` rule while `documents/SELECTION.md` describes `sum`; moot since the
  triples are hand-picked, but fix when touched.
- ~~Refresh ablation/synthetic numbers~~ — done; analysis text written 2026-08-31.
