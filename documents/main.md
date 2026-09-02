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
  Trajectory-level 30/20/50 reference/val/test split.
- **Seed triples are DONE.** The selection phase is finished; the triples are
  hand-picked by the author, frozen in `configs-main/<ds>.yaml`, one per subset,
  shared by both backbones. The per-triple sweep machinery (`scripts/main/`)
  produced candidates, but the final triples are the author's call — including the
  WW-HC manual override [13, 14, 15]. Do not re-run triple selection.
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

## Known items for later

- Method section: state the top-w trimming (or confirm `w="all"` configs).
- Convert config selection from test to validation.
- Documentation drift: `configs-main/ww.yaml`'s comment says the triples came from
  the `sum-diff` rule while `documents/SELECTION.md` describes `sum`; moot since the
  triples are hand-picked, but fix when touched.
- ~~Refresh ablation/synthetic numbers~~ — done; analysis text written 2026-08-31.
