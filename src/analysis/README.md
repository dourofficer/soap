# `src/analysis/` — diagnostics and manuscript figures

Everything here reads **cached representations** (`outputs/<ds>/…`, produced by the
extract stage) and the **exp-august selection tables** (which frozen config reached the
manuscript) — nothing re-runs the proxy model except the per-step scoring itself.
All commands run from the repo root, same CLI conventions as every other stage
(`--config configs/datasets/<ds>.yaml`, `--set`, `--model/--subset/--seed`, `--force`).

Outputs land in `artifacts/` (git-ignored, regenerable) except the contamination
figure, which writes `manuscript/assets/` directly.

## Prerequisites

1. Extracted activations for the dataset: `outputs/<ds>/<subset>/<model>/…` (extract
   stage; expensive, GPU).
2. The Protocol-2 selection tables from `exp-august/`:
   - `exp-august/outputs/manuscript-tables/table1_main_selection.tsv` (anchor config
     per cell) — read by `score_dist`, `contamination_figure`;
   - `exp-august/outputs/<ds>/tables/325/triples_selection.tsv` (per-window configs) —
     read by `score_dist_triples`, `qualitative`.

   These pin the analyses to the exact configs behind the reported numbers; every
   script re-derives the recorded accuracy and **asserts** it before writing anything.
   A mismatch means wrong config/window, not noise.

## Modules

| module | what it produces | where |
|---|---|---|
| `geometry` | the geometry probe: cos(u₀, mean), energy fractions, rank-AUC of norm vs ‖Pv‖² vs sin²θ — the empirical basis for "signal is alignment, not magnitude" | stdout tables |
| `score_dist` | raw base score per step (anchor config, one per cell), tagged `is_mistake` | `artifacts/score-dist/<ds>/scores.tsv` |
| `plot_score_dist` | error-vs-ordinary score densities per cell (`raw` + `znorm` variants) + `summary.tsv` of AUCs | `artifacts/score-dist/<ds>/plots/` |
| `score_dist_triples` | same per-step scores, but for **all 18 seed windows** (each with its own Protocol-2 config) | `artifacts/score-dist/<ds>/triples/steps.tsv` + `configs.tsv` |
| `plot_score_dist_triples` | per-seed + pooled density figures per window, and `ranking.tsv` (pick a window by AUC, not by browsing) | `artifacts/score-dist/<ds>/triples/plots/` |
| `qualitative` | every "flip" where base argmax lands downstream of gold and SOAP recovers it: trajectory JSON + per-stage scores + before/after figure per example | `artifacts/qualitative_examples/<model>/<subset>/` |
| `qualitative_plot` | drawing code for `qualitative` (two stacked panels: before/after scores + the promotion stems); not run directly | — |
| `contamination_figure` | the intro's transcript-card figure (hand-curated prose, computed scores) | `manuscript/assets/` + cached `artifacts/contamination_figure/scores.json` |

## Reproducing the manuscript figures

The four PDFs in `manuscript/assets/`:

**`contamination_handcrafted_traj1.pdf`, `contamination_handcrafted_traj21.pdf`**
(intro, downstream-contamination transcript cards):

```bash
python -m src.analysis.contamination_figure --config configs/datasets/ww.yaml \
    --model qwen3.5-9b --subset hand-crafted
```

Writes straight into `manuscript/assets/`. Scores are cached in
`artifacts/contamination_figure/scores.json`, so restyling reruns (`--score-source`,
`--only 21`) are instant and CPU-only; `--force` recomputes.

**`qwen3.5-9b_algorithm-generated_seed-3.pdf`, `deepseek-8b_hand-crafted_seed-15.pdf`**
(score-distribution densities, one seed each):

```bash
# 1. score every seed window (needs the cached representations; slow first time)
python -m src.analysis.score_dist_triples --config configs/datasets/ww.yaml

# 2. draw all windows + the ranking
python -m src.analysis.plot_score_dist_triples \
    --steps artifacts/score-dist/ww/triples/steps.tsv
```

Figures land under `artifacts/score-dist/ww/triples/plots/{raw,znorm}/` as
`<model>_<subset>_seeds-A-B-C_seed-N.pdf`. The manuscript copies are the `znorm`
variant, renamed to drop the window part — pick the window from
`plots/ranking.tsv` (best pooled AUC first), then copy, e.g.:

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
# single-cell score distributions under the one anchor config (simpler than triples)
python -m src.analysis.score_dist --config configs/datasets/ww.yaml
python -m src.analysis.plot_score_dist --scores artifacts/score-dist/ww/scores.tsv

# geometry probe
python -m src.analysis.geometry --config configs/datasets/correct-full.yaml \
    --seed 1 --set poolings=[mean] --set bands=[1,5,20]
```

Swap `ww.yaml` for `traceelephant.yaml` / `correct-error.yaml` / `correct-full.yaml` to
cover the other datasets (score-dist artifacts for all four already exist under
`artifacts/score-dist/`).
