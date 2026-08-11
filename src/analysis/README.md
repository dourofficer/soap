# `src/analysis/` — diagnostics and manuscript figures

Everything here reads **cached representations** (`outputs/<ds>/…`, produced by the
extract stage) and the pipeline's own **selection tables** (which frozen config reached
the manuscript) — nothing re-runs the proxy model except the per-step scoring itself.
All commands run from the repo root, same CLI conventions as every other stage
(`--config configs/datasets/<ds>.yaml`, `--set`, `--model/--subset/--seed`, `--force`).

Outputs land in `artifacts/` (git-ignored, regenerable) except the contamination
figure, which writes `manuscript/assets/` directly.

## Prerequisites

1. Extracted activations for the dataset: `outputs/<ds>/activations/<model>/<subset>/`
   (extract stage; expensive, GPU).
2. The selection tables produced by the protocol itself:
   - `outputs/manuscript-tables/table1_main_selection.tsv` (anchor config per cell) —
     read by `contamination_figure`, `qualitative`;
   - `outputs/<ds>/tables/325/triples_selection.tsv` (per-window configs) —
     read by `score_dist_triples`.

   Both are written by `src.reports.{triples,manuscript}`. They pin the analyses to the
   exact configs behind the reported numbers; every script re-derives the recorded
   accuracy and **asserts** it before writing anything. A mismatch means wrong
   config/window, not noise. Point elsewhere with `--set selection_tsv=…` (e.g. at a
   with-GT tree under `outputs-gt/`) — the SVD row schema is identical.

## Modules

| module | what it produces | where |
|---|---|---|
| `geometry` | the geometry probe: cos(u₀, mean), energy fractions, rank-AUC of norm vs ‖Pv‖² vs sin²θ — the empirical basis for "signal is alignment, not magnitude" | stdout tables |
| `score_dist_triples` | per-step base scores for **all 18 seed windows** (each with its own protocol config), tagged `is_mistake` | `artifacts/score-dist/<ds>/triples/steps.tsv` + `configs.tsv` |
| `plot_score_dist` | drawing library for the above: error-vs-ordinary densities per cell (`raw` + `znorm`), AUC helpers. Importable; also runnable on a `steps.tsv` directly | `artifacts/score-dist/<ds>/plots/` |
| `plot_score_dist_triples` | per-seed + pooled density figures per window, and `ranking.tsv` (pick a window by AUC, not by browsing) | `artifacts/score-dist/<ds>/triples/plots/` |
| `qualitative` | every "flip" where base argmax lands downstream of gold and SOAP recovers it: trajectory JSON + per-stage scores + before/after figure per example | `artifacts/qualitative_examples/<model>/<subset>/` |
| `qualitative_plot` | drawing code for `qualitative` (two stacked panels: before/after scores + the promotion stems); not run directly | — |
| `contamination_figure` | the intro's transcript-card figure (hand-curated prose, computed scores) | `manuscript/assets/` + cached `artifacts/contamination_figure/scores.json` |

> `qualitative` re-runs a frozen config per-step, so it imports the reproduction
> primitives from `main.reproduce` (reproduction lives in `main/`, not `src/`).

## Reproducing the manuscript figures

The four PDFs in `manuscript/assets/`:

**`contamination_example.pdf`** (intro, downstream-contamination transcript card):

```bash
python -m src.analysis.contamination_figure --config configs/datasets/ww.yaml \
    --model qwen3.5-9b --subset hand-crafted
```

Writes straight into `manuscript/assets/`. Scores are cached in
`artifacts/contamination_figure/scores.json`, so restyling reruns (`--score-source`,
`--only 21`) are instant and CPU-only; `--force` recomputes.

**`qwen3.5-9b_algorithm-generated_seed-3.pdf`, `deepseek-8b_hand-crafted_seed-15.pdf`,
`qwen3.5-9b_arc_seeds-6-7-8_seed-7.pdf`** (score-distribution densities, one seed each):

```bash
# 1. score every seed window (needs the cached representations; slow first time)
python -m src.analysis.score_dist_triples --config configs/datasets/ww.yaml

# 2. draw all windows + the ranking
python -m src.analysis.plot_score_dist_triples \
    --steps artifacts/score-dist/ww/triples/steps.tsv
```

Figures land under `artifacts/score-dist/ww/triples/plots/{raw,znorm}/` as
`<model>_<subset>_seeds-A-B-C_seed-N.pdf`. The manuscript copies are the `znorm`
variant — pick the window from `plots/ranking.tsv` (best pooled AUC first), then copy.
The ww/traceelephant copies drop the window part from the name; the correct-error one
keeps it:

```bash
cp artifacts/score-dist/ww/triples/plots/znorm/qwen3.5-9b_algorithm-generated_seeds-1-2-3_seed-3.pdf \
   manuscript/assets/qwen3.5-9b_algorithm-generated_seed-3.pdf
```

**Qualitative flip examples** (browse, then pin the headline example):

```bash
python -m src.analysis.qualitative --config configs/datasets/ww.yaml --model qwen3.5-9b
# pin which examples make combined.pdf, without touching code:
python -m src.analysis.qualitative --config configs/datasets/ww.yaml --model qwen3.5-9b \
    --set pick.hand-crafted=[13,42] --set seeds_scope=manuscript
```

## Other entry points

```bash
# geometry probe
python -m src.analysis.geometry --config configs/datasets/correct-error.yaml \
    --seed 1 --set poolings=[mean] --set bands=[1,5,20]
```

Swap `ww.yaml` for `traceelephant.yaml` / `correct-error.yaml` to cover the other
datasets.
